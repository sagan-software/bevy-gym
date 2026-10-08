// Readiness is posted by Rust after initialization installs the message listener.
import init from "./drone-worker.js";

await init();
