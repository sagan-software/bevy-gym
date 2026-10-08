// Geometry and colors ported from Farama Gymnasium (MIT), revision
// 7a1191388aa4aa973d3a5e4b039899cd99cc991f, classic_control/{cartpole,mountain_car}.py.
// These environments draw polygons and circles; upstream uses no sprite assets.
function surface(canvas) {
  const width = 600, height = 400;
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
