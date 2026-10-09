import { activateAudio } from "./audio.js";

activateAudio(globalThis, document);

const loading = document.querySelector("#loading");
try {
  // The generated bindings capture AudioContext while their module evaluates.
  const { default: init } = await import("./drone-pursuit.js");
  await init();
  loading.remove();
} catch (error) {
  loading.textContent = "The arena could not start. Reload to retry.";
  console.error(error);
}
