/** Score RGB-D points against the exported scene triangles, using Three's geometry library. */
const fs = require('node:fs');
const THREE = require('three');
const [surfacesPath] = process.argv.slice(2);
const surfaces = JSON.parse(fs.readFileSync(surfacesPath, 'utf8'));
if (surfaces.frame !== 'gazebo_enu_metres') throw new Error('Unexpected surface coordinate frame');
const queries = JSON.parse(fs.readFileSync(0, 'utf8'));
const triangle = new THREE.Triangle(), point = new THREE.Vector3(), closest = new THREE.Vector3();
const scores = queries.map(query => {
  if (!Array.isArray(query.targets) || !query.targets.length) throw new Error('Missing target IDs');
  point.set(query.point_grid[0] * 3, -query.point_grid[1] * 3, query.point_grid[2] * 5);
  if (![point.x, point.y, point.z].every(Number.isFinite)) throw new Error('Invalid point');
  let best = Infinity, target = null;
  for (const id of query.targets) {
    const vertices = surfaces.objects[id];
    if (!Array.isArray(vertices) || !vertices.length || vertices.length % 9) throw new Error('Unknown or invalid target surface: ' + id);
    for (let i = 0; i < vertices.length; i += 9) {
      triangle.a.fromArray(vertices, i); triangle.b.fromArray(vertices, i + 3); triangle.c.fromArray(vertices, i + 6);
      triangle.closestPointToPoint(point, closest);
      const distance = point.distanceTo(closest);
      if (distance < best) { best = distance; target = id; }
    }
  }
  return {distance_m: best, target_id: target};
});
console.log(JSON.stringify(scores));
