import { Training } from './training.mjs';

/** Keep training state separate from the frozen policy selected in the scene. */
export function mountTraining(watchCheckpoint) {
  const element = id => document.getElementById(id);
  const form = element('training-form');
  const seed = element('training-seed');
  const mode = element('training-mode');
  const toggle = element('training-toggle');
  const discard = element('training-discard');
  const watch = element('watch-checkpoint');
  const download = element('download-checkpoint');
  const training = new Training(render);
  form.closest('.training').addEventListener('keydown', event => event.stopPropagation());

  function render(state) {
    const labels = {idle:'Start training',starting:'Loading trainer…',running:'Pause training',pausing:'Pausing…',paused:'Resume training',inspecting:'Evaluating…',complete:'Training complete',exhausted:'Lesson budget reached',failed:'Retry training'};
    const status = {idle:'Training starts from random weights.',starting:'Loading the training worker…',running:'Training',pausing:'Pausing after this update…',paused:'Paused',inspecting:'Testing five independent episodes…',complete:'Training complete. Test the checkpoint before using it.',exhausted:'Lesson did not pass within 600 updates. Inspect the checkpoint or discard this run.',failed:state.error};
    toggle.textContent = labels[state.phase];
    toggle.disabled = ['starting','pausing','inspecting','complete','exhausted'].includes(state.phase);
    seed.disabled = !['idle','failed'].includes(state.phase);
    mode.disabled = seed.disabled;
    discard.hidden = state.phase === 'idle';
    element('training-status').textContent = status[state.phase];
    const course = state.curriculum ?? (['idle','failed'].includes(state.phase) && mode.value === 'curriculum'
      ? {lesson:'hover',lesson_updates:0,evaluation:null} : null);
    element('training-count').textContent = course
      ? `${course.lesson === 'hover' ? 'Calm hover' : 'Disturbed recovery'}: ${course.lesson_updates} / 600 updates`
      : `${state.updates} / 260 updates`;
    element('training-progress').max = course ? 600 : 260;
    element('training-progress').value = course ? course.lesson_updates : state.updates;
    const lessonScore = element('lesson-score');
    lessonScore.hidden = !course;
    if (course) {
      const result = course.evaluation;
      lessonScore.textContent = result
        ? `${result.lesson === 'hover' ? 'Calm hover' : 'Disturbed recovery'} ${result.passed ? 'passed' : 'not passed'}: ${result.episodes.filter(e => e.survived).length}/5 survived; mean return ${(result.episodes.reduce((sum,e) => sum + e.reward,0)/5).toFixed(1)}; distance ${(result.episodes.reduce((sum,e) => sum + e.final_distance,0)/5).toFixed(2)} m.`
        : 'Evaluates every 20 updates. Pass: 5/5 survive, mean return ≥400, mean distance ≤0.5 m.';
    }
    element('training-steps').textContent = `${(state.updates * 512).toLocaleString()} simulation steps`;
    const recipe = state.metrics ?? state.recipe;
    element('policy-rate').textContent = recipe?.actor_learning_rate ?? 'Not started';
    element('value-rate').textContent = recipe?.critic_learning_rate ?? 'Not started';
    for (const [id,key] of [['policy-loss','actor_loss'],['value-loss','critic_loss'],['entropy','entropy'],['kl','approximate_kl']]) {
      element(id).textContent = state.metrics ? state.metrics[key].toPrecision(5) : 'Not started';
    }
    const canInspect = ['paused','complete','exhausted'].includes(state.phase) && state.updates > 0;
    watch.disabled = !canInspect;
    download.disabled = !canInspect;
    element('checkpoint-hint').hidden = canInspect || state.phase === 'inspecting';
    const score = element('checkpoint-score');
    score.hidden = !state.evaluation;
    if (state.evaluation) {
      const result = state.evaluation;
      const survived = result.episodes.filter(episode => episode.survived).length;
      const mean = result.episodes.reduce((sum,episode) => sum + episode.reward,0) / 5;
      const baseline = result.baseline.reduce((sum,episode) => sum + episode.reward,0) / 5;
      score.textContent = `Update ${result.updates}: ${survived} of 5 episodes survived. Mean return ${mean.toFixed(1)}; constant thrust ${baseline.toFixed(1)}.`;
    }
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    if (training.state.phase === 'running') training.pause();
    else if (training.state.phase === 'paused') training.resume();
    else if (['idle','failed'].includes(training.state.phase) && form.reportValidity()) {
      element('checkpoint-status').textContent = '';
      void training.start(Number(seed.value), mode.value).catch(error => { element('training-status').textContent = error.message; });
    }
  });
  discard.addEventListener('click', () => training.discard());
  mode.addEventListener('change', () => render(training.state));

  async function useCheckpoint(action) {
    element('checkpoint-status').textContent = '';
    try {
      const checkpoint = await training.checkpoint();
      if (action === 'watch') {
        watchCheckpoint(checkpoint.bytes);
        const canvas = element('drone-canvas');
        canvas.focus({preventScroll: true});
        canvas.scrollIntoView({block: 'start', behavior: 'instant'});
        element('checkpoint-status').textContent = `Watching update ${checkpoint.updates} from seed ${checkpoint.seed}.`;
      } else {
        const url = URL.createObjectURL(new Blob([checkpoint.bytes], {type:'application/octet-stream'}));
        const link = document.createElement('a');
        link.href = url;
        link.download = `drone-${checkpoint.mode}-seed-${checkpoint.seed}-update-${checkpoint.updates}.mpk`;
        link.click();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
    } catch (error) {
      if (error?.name !== 'AbortError') element('checkpoint-status').textContent = `Checkpoint could not be used: ${error?.message ?? String(error)}`;
    }
  }
  watch.addEventListener('click', () => { void useCheckpoint('watch'); });
  download.addEventListener('click', () => { void useCheckpoint('download'); });
  window.addEventListener('pagehide', () => training.discard());
  render(training.state);
}
