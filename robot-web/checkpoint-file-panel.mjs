import { CheckpointFile } from './checkpoint-file.mjs';

/** Keep local file selection independent of the browser's training run. */
export function mountCheckpointFile(watch) {
  const form = document.getElementById('checkpoint-file-form');
  const file = document.getElementById('checkpoint-file');
  const kind = document.getElementById('checkpoint-kind');
  const button = document.getElementById('watch-file');
  const status = document.getElementById('file-status');
  const input = new CheckpointFile(watch, state => {
    button.disabled = state.phase === 'reading' || !file.files.length;
    button.textContent = state.phase === 'reading' ? 'Reading file…' : 'Watch file';
    status.textContent = state.phase === 'watching' ? `Watching ${state.name}.`
      : state.phase === 'failed' ? state.message : '';
    if (state.phase === 'watching') document.getElementById('drone-canvas').focus({preventScroll:true});
  });
  file.addEventListener('change', () => input.clear());
  kind.addEventListener('change', () => input.clear());
  form.addEventListener('submit', event => {
    event.preventDefault();
    void input.watch(file.files[0], kind.value);
  });
  input.clear();
}
