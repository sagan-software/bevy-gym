/** Read one selected local checkpoint and discard superseded asynchronous reads. */
export class CheckpointFile {
  #generation = 0;
  #watch;
  #changed;

  constructor(watch, changed) {
    this.#watch = watch;
    this.#changed = changed;
  }

  clear() {
    this.#generation++;
    this.#changed({phase:'idle'});
  }

  async watch(file, kind) {
    const generation = ++this.#generation;
    if (!file) {
      this.#changed({phase:'failed', message:'Choose a checkpoint file.'});
      return;
    }
    if (!['recovery', 'damage'].includes(kind) || typeof this.#watch[kind] !== 'function') {
      this.#changed({phase:'failed', message:'Choose a supported policy type.'});
      return;
    }
    if (file.size > 1048576) {
      this.#changed({phase:'failed', message:'Choose a checkpoint no larger than 1 MiB.'});
      return;
    }
    this.#changed({phase:'reading'});
    let bytes;
    try {
      bytes = new Uint8Array(await file.arrayBuffer());
    } catch {
      if (generation === this.#generation) {
        this.#changed({phase:'failed', message:'Could not read this file. Select it again.'});
      }
      return;
    }
    if (generation !== this.#generation) return;
    try {
      // The synchronous WASM callback validates before replacing the pending policy.
      this.#watch[kind](bytes);
      this.#changed({phase:'watching', name:file.name});
    } catch {
      this.#changed({phase:'failed', message:'Could not load this checkpoint. Check the policy type or choose another file.'});
    }
  }
}
