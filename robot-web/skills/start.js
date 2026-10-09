import init from "./lesson.js";

const loading = document.querySelector("#loading");
try {
  await init();
  loading.remove();
} catch (error) {
  loading.textContent = `The lesson could not start: ${error}. Reload to retry.`;
  console.error(error);
}
