/** Play silent example recordings only while visible and permitted by the viewer. */
export function initializePreviews({ document, videos, button, preference, Observer }) {
  const visible = new Set();
  const versions = new Map();
  let choice = null;
  const enabled = () => choice ?? !preference.matches;
  const label = () => { button.textContent = enabled() ? 'Pause previews' : 'Play previews'; };
  const refresh = () => {
    label();
    for (const video of videos) {
      const version = (versions.get(video) ?? 0) + 1;
      versions.set(video, version);
      if (!enabled() || document.hidden || !visible.has(video)) {
        video.pause();
        continue;
      }
      video.muted = true;
      if (!video.src) video.src = video.dataset.src;
      // A stale rejection cannot override a newer playback request or pause.
      video.play().catch(() => {
        if (versions.get(video) !== version || !enabled()) return;
        choice = false;
        refresh();
      });
    }
  };
  const observer = new Observer(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting) visible.add(entry.target);
      else visible.delete(entry.target);
    }
    refresh();
  });
  for (const video of videos) observer.observe(video);
  button.hidden = false;
  label();
  button.addEventListener('click', () => { choice = !enabled(); refresh(); });
  document.addEventListener('visibilitychange', refresh);
  preference.addEventListener('change', refresh);
}

