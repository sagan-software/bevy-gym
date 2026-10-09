/** Own one worker and one outstanding request. The page never drives its optimizer directly. */
export class Training {
  #workerFactory;
  #onChange;
  #port = null;
  #state = Object.freeze({phase: 'idle', seed: 7, mode: 'recovery', curriculum: null, updates: 0, metrics: null, recipe: null, evaluation: null, error: null});

  constructor(onChange, workerFactory = () => new Worker(new URL('./worker.js', import.meta.url), {type: 'module'})) {
    this.#onChange = onChange;
    this.#workerFactory = workerFactory;
  }

  get state() { return this.#state; }

  #publish(change) {
    this.#state = Object.freeze({...this.#state, ...change});
    this.#onChange(this.#state);
  }

  /** Validate before replacing state. Initialization and late replies belong to this worker only. */
  async start(seed, mode = 'recovery') {
    if (!['recovery', 'curriculum'].includes(mode)) throw new RangeError('Choose recovery or curriculum training.');
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new RangeError('Seed must be a whole number from 0 to 4294967295.');
    }
    if (!['idle', 'failed'].includes(this.#state.phase)) throw new Error('Discard the current run before starting another.');
    this.#publish({phase: 'starting', seed, mode, curriculum: null, updates: 0, metrics: null, recipe: null, evaluation: null, error: null});
    let port;
    try {
      const worker = this.#workerFactory();
      port = {worker, pending: null};
      this.#port = port;
      const ready = new Promise((resolve, reject) => { port.pending = {event: 'ready', resolve, reject}; });
      worker.onmessage = event => this.#receive(port, event);
      worker.onerror = event => this.#fail(port, new Error(event.message || 'Training worker could not start.'));
      worker.onmessageerror = () => this.#fail(port, new Error('Training worker message could not be read.'));
      await ready;
      const result = await this.#request(port, {command: mode === 'curriculum' ? 'start_curriculum' : 'start', seed}, 'started');
      if (result.seed !== seed || result.updates !== 0 || result.transitions !== 0 || result.update_limit !== (mode === 'curriculum' ? 1200 : 260)
          || result.actor_learning_rate !== 0.0003 || result.critic_learning_rate !== 0.001) {
        throw new Error('Training worker returned an incompatible recipe.');
      }
      if (mode === 'curriculum') validateCurriculum(result.curriculum, null, 0, 'training');
      if (this.#port !== port) return;
      this.#publish({phase: 'running', recipe: Object.freeze(result), curriculum: result.curriculum ?? null});
      void this.#run(port);
    } catch (error) {
      if (port === undefined) this.#publish({phase: 'failed', error: error.message});
      else this.#fail(port, error);
    }
  }

  pause() {
    if (this.#state.phase === 'running') this.#publish({phase: 'pausing'});
  }

  resume() {
    if (this.#state.phase !== 'paused') return;
    this.#publish({phase: 'running'});
    void this.#run(this.#port);
  }

  /** Cancel pending promises before releasing the old worker. Its callbacks become inert. */
  discard() {
    this.#close(new DOMException('The training run was discarded.', 'AbortError'));
    this.#publish({phase: 'idle', curriculum: null, updates: 0, metrics: null, recipe: null, evaluation: null, error: null});
  }

  /** Freeze and score weights only when collection has stopped. */
  async checkpoint() {
    const phase = this.#state.phase;
    if (!['paused', 'complete', 'exhausted'].includes(phase) || this.#state.updates === 0) {
      throw new Error('Pause training before using a checkpoint.');
    }
    const port = this.#port;
    const updates = this.#state.updates;
    this.#publish({phase: 'inspecting'});
    try {
      const policy = await this.#request(port, {command: 'export'}, 'policy');
      if (!Array.isArray(policy.bytes) || policy.bytes.length === 0 || policy.bytes.length > 1_048_576
          || !policy.bytes.every(value => Number.isInteger(value) && value >= 0 && value <= 255)) {
        throw new Error('Training worker returned an invalid checkpoint.');
      }
      const evaluation = await this.#request(port, {command: 'evaluate', bytes: policy.bytes}, 'evaluation');
      validateScores(evaluation.episodes);
      validateScores(evaluation.baseline);
      if (this.#port !== port) throw new DOMException('The training run was discarded.', 'AbortError');
      this.#publish({phase, evaluation: {...evaluation, updates}});
      return {bytes: Uint8Array.from(policy.bytes), updates, seed: this.#state.seed, mode: this.#state.mode};
    } catch (error) {
      this.#fail(port, error);
      throw error;
    }
  }

  async #run(port) {
    try {
      while (this.#port === port && this.#state.phase === 'running') {
        const progress = await this.#request(port, {command: 'advance'}, 'progress');
        if (this.#port !== port) return;
        const updates = this.#state.updates + 1;
        const status = this.#state.mode === 'curriculum' ? progress.status : updates === 260 ? 'complete' : 'training';
        if (this.#state.mode === 'curriculum') validateCurriculum(progress.curriculum, this.#state.curriculum, updates, status);
        const values = ['actor_loss', 'critic_loss', 'entropy', 'approximate_kl', 'actor_learning_rate', 'critic_learning_rate'];
        if (progress.updates !== updates || progress.transitions !== updates * 512 || progress.status !== status
            || !Number.isSafeInteger(progress.optimizer_steps) || progress.optimizer_steps <= 0
            || !values.every(key => Number.isFinite(progress[key]))) {
          throw new Error('Training worker returned invalid progress.');
        }
        const phase = ['complete', 'exhausted'].includes(status) ? status : this.#state.phase === 'pausing' ? 'paused' : 'running';
        this.#publish({phase, updates, metrics: Object.freeze(progress), curriculum: progress.curriculum ?? null});
      }
    } catch (error) { this.#fail(port, error); }
  }

  #request(port, command, event) {
    if (this.#port !== port) return Promise.reject(new DOMException('The training run was discarded.', 'AbortError'));
    if (port.pending) return Promise.reject(new Error('A training request is already pending.'));
    return new Promise((resolve, reject) => {
      port.pending = {event, resolve, reject};
      try { port.worker.postMessage(JSON.stringify(command)); }
      catch (error) { port.pending = null; reject(error); }
    });
  }

  #receive(port, event) {
    if (this.#port !== port) return;
    try {
      if (typeof event.data !== 'string') throw new Error('Training worker returned a non-text message.');
      if (event.data.length > 1_048_576) throw new Error('Training worker message exceeds 1 MiB.');
      const result = JSON.parse(event.data);
      if (!result || typeof result !== 'object' || !port.pending) throw new Error('Training worker returned an unexpected message.');
      if (result.event === 'error') throw new Error(result.message || 'Training worker failed.');
      if (result.event !== port.pending.event || (result.event === 'ready' && result.protocol !== 1)) {
        throw new Error('Training worker returned an incompatible response. Reload to retry.');
      }
      const pending = port.pending;
      port.pending = null;
      pending.resolve(result);
    } catch (error) { this.#fail(port, error); }
  }

  #fail(port, error) {
    if (this.#port !== port) return;
    this.#close(error);
    this.#publish({phase: 'failed', error: error.message});
  }

  #close(error) {
    const port = this.#port;
    this.#port = null;
    if (port) {
      port.pending?.reject(error);
      port.pending = null;
      port.worker.terminate();
    }
  }
}

/** The five independent episodes have fixed order and lossless decimal-string seeds. */
function validateScores(scores) {
  const seeds = ['0', '1', '2', '42', '18446744073709551615'];
  if (!Array.isArray(scores) || scores.length !== seeds.length || !scores.every((score, index) =>
    score && score.seed === seeds[index] && Number.isInteger(score.steps) && score.steps >= 1 && score.steps <= 500
    && Number.isFinite(score.reward) && Number.isFinite(score.final_distance) && score.final_distance >= 0
    && typeof score.survived === 'boolean')) {
    throw new Error('Training worker returned invalid evaluation scores.');
  }
}

/** Verify every lesson transition against the native completion rule. */
function validateCurriculum(course, previous, total, status) {
  const fail = () => { throw new Error('Training worker returned invalid curriculum progress.'); };
  if (!course || course.lesson_limit !== 600 || !['hover', 'recovery'].includes(course.lesson)
      || !Number.isInteger(course.lesson_updates) || course.lesson_updates < 0 || course.lesson_updates > 600
      || !Number.isInteger(total) || total < 0 || total > 1200) fail();
  if (!previous) {
    if (total !== 0 || course.lesson !== 'hover' || course.lesson_updates !== 0
        || course.evaluation !== null || status !== 'training') fail();
    return;
  }
  const count = previous.lesson_updates + 1;
  let lesson = previous.lesson, nextCount = count, nextStatus = 'training';
  if (count % 20 === 0) {
    const evaluation = course.evaluation;
    if (!evaluation || evaluation.lesson !== lesson) fail();
    validateScores(evaluation.episodes);
    const scores = evaluation.episodes;
    const passed = scores.every(score => score.survived)
      && scores.reduce((sum,score) => sum + score.reward,0) / 5 >= 400
      && scores.reduce((sum,score) => sum + score.final_distance,0) / 5 <= 0.5;
    if (evaluation.passed !== passed) fail();
    if (passed && lesson === 'hover') { lesson = 'recovery'; nextCount = 0; }
    else if (passed) nextStatus = 'complete';
    else if (count === 600) nextStatus = 'exhausted';
  } else if (JSON.stringify(course.evaluation) !== JSON.stringify(previous.evaluation)) fail();
  if (course.lesson !== lesson || course.lesson_updates !== nextCount || status !== nextStatus) fail();
}
