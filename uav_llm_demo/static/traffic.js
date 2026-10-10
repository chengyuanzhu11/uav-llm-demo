export function sampleTraffic(model, elapsed) {
  const segment = model.segments.find((item) => elapsed < item.start + item.duration)
    || model.segments[model.segments.length - 1];
  const fraction = Math.min(1, Math.max(0, (elapsed - segment.start) / segment.duration));
  const remaining = 1 - fraction;
  const controls = segment.controls;
  const position = [0, 1, 2].map((axis) => remaining ** 3 * controls[0][axis]
    + 3 * remaining ** 2 * fraction * controls[1][axis]
    + 3 * remaining * fraction ** 2 * controls[2][axis] + fraction ** 3 * controls[3][axis]);
  const velocity = [0, 1, 2].map((axis) => 3 * (remaining ** 2 * (controls[1][axis] - controls[0][axis])
    + 2 * remaining * fraction * (controls[2][axis] - controls[1][axis])
    + fraction ** 2 * (controls[3][axis] - controls[2][axis])) / segment.duration);
  return { position, velocity };
}
