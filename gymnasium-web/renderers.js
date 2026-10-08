// Geometry and colors ported from Farama Gymnasium (MIT), revision
// 7a1191388aa4aa973d3a5e4b039899cd99cc991f, classic_control/{cartpole,mountain_car,continuous_mountain_car,pendulum}.py.
// Pendulum also uses the original assets/clockwise.png sprite.
function surface(canvas, width = 600, height = 400) {
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, width, height);
  // Match pygame's final vertical flip while retaining its source coordinates.
  ctx.translate(0.5, height - 0.5); ctx.scale(1, -1);
  return ctx;
}
function polygon(ctx, points, color) {
  ctx.fillStyle = color; ctx.beginPath();
  points.forEach(([x, y], index) => { x = Math.trunc(x); y = Math.trunc(y); if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
  ctx.closePath(); ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.stroke();
}
function circle(ctx, x, y, radius, color) {
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, radius, 0, 2 * Math.PI); ctx.fill();
}
export function drawCartPole(canvas, state) {
  const ctx = surface(canvas), scale = 600 / 4.8, cartx = state[0] * scale + 300, carty = 100;
  polygon(ctx, [[-25,-15],[-25,15],[25,15],[25,-15]].map(([x,y]) => [x+cartx,y+carty]), "#000000");
  const angle = -state[2], cosine = Math.cos(angle), sine = Math.sin(angle);
  polygon(ctx, [[-5,-5],[-5,120],[5,120],[5,-5]].map(([x,y]) => [
    x*cosine-y*sine+cartx, x*sine+y*cosine+carty+7.5,
  ]), "#ca9865");
  circle(ctx, Math.trunc(cartx), Math.trunc(carty+7.5), 5, "#8184cb");
  ctx.strokeStyle = "#000000"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0,carty); ctx.lineTo(600,carty); ctx.stroke();
}
export function drawMountainCar(canvas, state, goal = 0.5) {
  const ctx = surface(canvas), scale = 600 / 1.8, position = state[0];
  const height = x => Math.sin(3*x)*0.45+0.55;
  const x = position => (position+1.2)*scale;
  ctx.strokeStyle = "#000000"; ctx.lineWidth = 1; ctx.beginPath();
  for (let i=0;i<100;i++) {
    const position=-1.2+1.8*i/99;
    if (i) ctx.lineTo(x(position),height(position)*scale); else ctx.moveTo(x(position),height(position)*scale);
  }
  ctx.stroke();
  const angle=Math.cos(3*position), cosine=Math.cos(angle), sine=Math.sin(angle);
  const rotate=([a,b]) => [a*cosine-b*sine+x(position),a*sine+b*cosine+10+height(position)*scale];
  polygon(ctx,[[-20,0],[-20,20],[20,20],[20,0]].map(rotate),"#000000");
  for (const offset of [10,-10]) {
    const [a,b]=rotate([offset,0]); circle(ctx,Math.trunc(a),Math.trunc(b),8,"#808080");
  }
  const flagx=Math.trunc(x(goal)), flagy=Math.trunc(height(goal)*scale);
  ctx.beginPath();ctx.moveTo(flagx,flagy);ctx.lineTo(flagx,flagy+50);ctx.stroke();
  polygon(ctx,[[flagx,flagy+50],[flagx,flagy+40],[flagx+25,flagy+45]],"#cccc00");
}

let torqueArrow;
export async function loadPendulumAsset() {
  if (!torqueArrow) {
    const image = new Image();
    image.src = new URL("assets/clockwise.png", import.meta.url).href;
    await image.decode();
    torqueArrow = image;
  }
}
export function drawPendulum(canvas, [angle, velocity, torque]) {
  const ctx = surface(canvas, 500, 500), scale = 500 / 4.4, offset = 250;
  const width = .2 * scale, rotation = angle + Math.PI / 2;
  const cosine = Math.cos(rotation), sine = Math.sin(rotation);
  const rotate = ([x, y]) => [x * cosine - y * sine + offset, x * sine + y * cosine + offset];
  polygon(ctx, [[0, -width / 2], [0, width / 2], [scale, width / 2], [scale, -width / 2]].map(rotate), "#cc4d4d");
  const [x, y] = rotate([scale, 0]);
  circle(ctx, offset, offset, Math.trunc(width / 2), "#cc4d4d");
  circle(ctx, Math.trunc(x), Math.trunc(y), Math.trunc(width / 2), "#cc4d4d");
  const size = Math.trunc(scale * Math.abs(torque) / 2);
  if (size > 0) {
    if (!torqueArrow) throw new Error("Pendulum torque asset has not loaded.");
    // The two vertical flips cancel; retain Pygame's integer blit coordinates.
    const left = offset - Math.trunc(size / 2), top = 500 - (offset - Math.trunc(size / 2)) - size;
    ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.translate(torque > 0 ? left + size : left, top); ctx.scale(torque > 0 ? -1 : 1, 1);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(torqueArrow, 0, 0, size, size);
    ctx.restore();
  }
  circle(ctx, offset, offset, Math.trunc(.05 * scale), "#000000");
}
