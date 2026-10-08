import init from "./drone-flight.js";

const loading = document.querySelector("#loading");
try {
  await init();
  loading.remove();
} catch (error) {
  loading.textContent = "The drone simulation could not start. Reload to retry.";
  console.error(error);
}
