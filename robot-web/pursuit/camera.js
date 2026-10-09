/** Keep rejected pointer capture recoverable in embedded browsers. */
export function activateCamera(scope, document) {
  const canvas = document.querySelector('#pursuit-canvas');
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  scope.droneCameraCapture = async () => {
    try {
      await canvas.requestPointerLock();
    } catch {
      // Right-button drag still rotates the camera when capture is unavailable.
    }
  };
  scope.droneCameraCaptured = () => document.pointerLockElement === canvas;
  scope.droneCameraRelease = () => document.exitPointerLock();
}
