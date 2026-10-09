// Install before WASM creates its audio output. Browsers require a play gesture
// before a suspended Web Audio context may resume. Failed attempts retry later.
export function activateAudio(scope, events) {
  if (typeof scope.AudioContext !== "function") return;
  const contexts = new Set();
  scope.AudioContext = new Proxy(scope.AudioContext, {
    construct(target, args, constructor) {
      const context = Reflect.construct(target, args, constructor);
      contexts.add(context);
      return context;
    },
  });
  const resume = () => {
    for (const context of contexts) {
      if (context.state === "closed") {
        contexts.delete(context);
      } else if (context.state === "suspended") {
        // A gesture can still be rejected by browser policy. Keep it for retry.
        context.resume().catch(() => {});
      }
    }
  };
  events.addEventListener("pointerdown", resume);
  events.addEventListener("pointerup", resume);
  events.addEventListener("keydown", resume);
}
