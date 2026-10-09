import init from "./drone-pursuit.js";

const loading = document.querySelector("#loading");
try {
  await init();
  loading.remove();
} catch (error) {
  loading.textContent = "The arena could not start. Reload to retry.";
  console.error(error);
}
