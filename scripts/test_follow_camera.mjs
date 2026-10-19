import assert from 'node:assert/strict';
import * as THREE from '../uav_llm_demo/static/vendor/three.core.js';
import { FollowCameraRig } from '../uav_llm_demo/static/follow_camera.js';

const base = new THREE.Vector3(9, 7, 12);
const results = [];
const box = (a, b) => new THREE.Box3(new THREE.Vector3(...a), new THREE.Vector3(...b));
function run(name, rate, path, duration, obstacles = [], horizon = 0.3) {
  const rig = new FollowCameraRig(THREE);
  const start = path(0);
  rig.reset(start, obstacles);
  let previous = rig.position.clone();
  let maxSpeed = 0;
  let maximumAcceleration = 0;
  let previousVelocity = null;
  let maxRelativeStep = 0;
  let maxRelativeDrift = 0;
  let lastTarget = start;
  let maxStep = 0;
  let minimumClearance = Infinity;
  const ray = new THREE.Ray();
  const hit = new THREE.Vector3();
  for (let frame = 1; frame <= duration * rate; frame += 1) {
    const t = frame / rate;
    const target = path(t);
    const previews = Array.from({ length: 6 }, (_, i) => path(Math.min(duration, t + horizon * (i + 1) / 6)));
    const position = rig.update(target, 1 / rate, obstacles, previews);
    const step = position.distanceTo(previous);
    maxStep = Math.max(maxStep, step);
    maxSpeed = Math.max(maxSpeed, step * rate);
    const velocity = position.clone().sub(previous).multiplyScalar(rate);
    if (previousVelocity) maximumAcceleration = Math.max(maximumAcceleration, velocity.distanceTo(previousVelocity) * rate);
    previousVelocity = velocity;
    const beforeOffset = previous.clone().sub(lastTarget);
    const offset = position.clone().sub(target);
    maxRelativeStep = Math.max(maxRelativeStep, offset.distanceTo(beforeOffset));
    maxRelativeDrift = Math.max(maxRelativeDrift, offset.distanceTo(base));
    assert.ok(position.toArray().every(Number.isFinite), `${name}: nonfinite camera`);
    for (const obstacle of obstacles) {
      assert.ok(!obstacle.containsPoint(target), `${name}: invalid fixture flies through a building`);
      assert.ok(!obstacle.containsPoint(position), `${name}: camera inside building at ${t}`);
      minimumClearance = Math.min(minimumClearance, obstacle.distanceToPoint(position));
      const swept = position.clone().sub(previous);
      if (swept.length() > 1e-7) {
        ray.set(previous, swept.clone().normalize());
        const cross = ray.intersectBox(obstacle, hit);
        assert.ok(!cross || hit.distanceTo(previous) >= swept.length() - 1e-5,
          `${name}: camera crossed building between frames at ${t}`);
      }
      ray.set(target, offset.clone().normalize());
      const blocked = ray.intersectBox(obstacle, hit);
      assert.ok(!blocked || hit.distanceTo(target) >= offset.length() - 1e-5,
        `${name}: building blocks camera sightline at ${t}`);
    }
    previous.copy(position);
    lastTarget = target;
  }
  const result = { name, fps: rate, horizon, maxSpeed, maximumAcceleration, maxStep, maxRelativeStep, maxRelativeDrift,
    minimumClearance: Number.isFinite(minimumClearance) ? minimumClearance : null, ...rig.diagnostics };
  results.push(result);
  return result;
}

for (const fps of [30, 60, 120]) {
  const constant = run('constant velocity', fps, t => new THREE.Vector3(t * 3, 12, t), 8);
  assert.ok(constant.maxRelativeDrift < 1e-8);
  assert.ok(Math.abs(constant.maxSpeed - Math.sqrt(10)) < 1e-7);
  assert.equal(constant.emergencyClamps, 0);

  // Previous implementation jumps while this valid flight passes a building corner.
  const tangent = new THREE.Vector3(-0.4587629571022261, 0, 0.8885586920349274);
  const corner = run('original elevated building corner', fps,
    t => tangent.clone().multiplyScalar(-15 + 3 * t).setY(12), 10,
    [box([5.485911569558084, 0, 1.6766781266778708], [10.437978745438159, 29.451486136764288, 8.752997133880854])]);
  assert.equal(corner.emergencyClamps, 0, 'confirmed 300ms preview should anticipate this corner');
  assert.equal(corner.sweepClamps, 0);
  assert.ok(corner.maxSpeed < 20, `corner speed ${corner.maxSpeed}`);
  assert.ok(corner.maximumAcceleration < 140, `corner acceleration ${corner.maximumAcceleration}`);

  const wall = run('near wall then corner', fps,
    t => new THREE.Vector3(0, 12, 12 - 3 * t), 12,
    [box([4, 0, -100], [5, 100, 5])]);
  assert.equal(wall.emergencyClamps, 0);
  assert.equal(wall.sweepClamps, 0);
  assert.ok(wall.maxSpeed < 28, JSON.stringify(wall));
  assert.ok(wall.maximumAcceleration < 140, `wall acceleration ${wall.maximumAcceleration}`);

  const narrow = run('stationary narrow courtyard', fps,
    () => new THREE.Vector3(0, 12, 0), 8,
    [box([3, 0, -30], [8, 30, 30]), box([-8, 0, -30], [-3, 30, 30]),
      box([-8, 0, 3], [8, 30, 8]), box([-8, 0, -8], [8, 30, -3])]);
  assert.equal(narrow.directionChanges, 0);
  assert.ok(narrow.maxSpeed < 1e-8, `stationary camera changed position: ${narrow.maxSpeed}`);
}
const shortPreview = run('short confirmed preview', 60,
  t => new THREE.Vector3(0, 12, 12 - 3 * t), 12,
  [box([4, 0, -100], [5, 100, 5])], 0.15);
assert.equal(shortPreview.emergencyClamps, 0);
assert.equal(shortPreview.sweepClamps, 0);
const emptyRig = new FollowCameraRig(THREE);
emptyRig.reset(new THREE.Vector3());
const frozen = emptyRig.position.clone();
assert.deepEqual(emptyRig.update(new THREE.Vector3(), NaN, [], null), frozen);
assert.ok(emptyRig.update(new THREE.Vector3(), 1 / 60, [], [null, new THREE.Vector3(NaN, 0, 0)]).toArray().every(Number.isFinite));
console.log(JSON.stringify({ passed: true, cases: results.length, results }, null, 2));
