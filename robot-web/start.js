import init, { watch_checkpoint, watch_recovery_file, watch_motor_failure_file } from "./drone-flight.js";
import { mountTraining } from "./training-panel.mjs";
import { mountCheckpointFile } from "./checkpoint-file-panel.mjs";

const loading = document.querySelector("#loading");
try {
  await init();
  loading.remove();
  mountTraining(watch_checkpoint);
  mountCheckpointFile({recovery:watch_recovery_file, damage:watch_motor_failure_file});
} catch (error) {
  loading.textContent = "The drone simulation could not start. Reload to retry.";
  console.error(error);
}
