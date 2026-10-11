import { initializePreviews } from './previews.mjs';

initializePreviews({
  document,
  videos: document.querySelectorAll('.example video'),
  button: document.querySelector('#preview-toggle'),
  preference: matchMedia('(prefers-reduced-motion: reduce)'),
  Observer: IntersectionObserver,
});
