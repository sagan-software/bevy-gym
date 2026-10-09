import init, { watch_checkpoint } from "./drone-flight.js";
import { mountTraining } from "./training-panel.mjs";

const loading = document.querySelector("#loading");
try {
  await init();
  loading.remove();
  mountTraining(watch_checkpoint);
} catch (error) {
  loading.textContent = "The drone simulation could not start. Reload to retry.";
  console.error(error);
}
