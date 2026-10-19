import assert from "node:assert/strict";
import { performance } from "node:perf_hooks";
import * as THREE from "../uav_llm_demo/static/vendor/three.core.js";
import { FlightTrail } from "../uav_llm_demo/static/flight_trail.js";

let checks = 0;
const test = (name, body) => { body(); checks++; console.log(`PASS ${name}`); };
const point = index => ({ x: index * 0.035, y: Math.sin(index * 0.007) * 10, z: 3 + Math.cos(index * 0.009) });
const toWorld = p => new THREE.Vector3(p.x, p.z, p.y);
const track = count => Array.from({ length: count }, (_, index) => point(index));
const geometrySegments = trail => trail.object.children.reduce((sum, mesh) => sum + mesh.geometry.drawRange.count / 36, 0);

function assertFiniteAndBounded(trail) {
  for (const mesh of trail.object.children) {
    const geometry = mesh.geometry;
    const positions = geometry.getAttribute("position").array;
    assert.ok(positions.every(Number.isFinite));
    assert.ok(Number.isFinite(geometry.boundingSphere.radius));
    assert.ok(geometry.drawRange.start >= 0 && geometry.drawRange.count >= 0);
    assert.ok(geometry.drawRange.start + geometry.drawRange.count <= geometry.index.count);
    assert.ok(geometry.getAttribute("position").updateRanges.length <= 1);
  }
}

test("new telemetry appends only new points; completed chunks stay immutable", () => {
  let conversions = 0;
  const trail = new FlightTrail(THREE, p => { conversions++; return toWorld(p); });
  const path = track(1646);
  trail.update(path.slice(0, 1600));
  const initial = trail.object.children.map(mesh => ({ mesh, version: mesh.geometry.getAttribute("position").version,
    values: mesh.geometry.getAttribute("position").array.slice() }));
  trail.update(path);
  assert.equal(conversions, path.length);
  assert.equal(trail.stats.segments, 1645);
  assert.equal(trail.stats.triangles, 19740);
  assert.equal(geometrySegments(trail), 1645);
  assert.equal(trail.material.depthTest, false);
  for (let i = 0; i < initial.length - 1; i++) {
    assert.equal(trail.object.children[i], initial[i].mesh);
    assert.equal(initial[i].mesh.geometry.getAttribute("position").version, initial[i].version);
    assert.deepEqual(initial[i].mesh.geometry.getAttribute("position").array, initial[i].values);
  }
  trail.update(path.map(p => ({ ...p })));
  assert.equal(conversions, path.length, "unchanged payload must not transform / append history");
  assertFiniteAndBounded(trail);
  trail.dispose();
});

test("rolling 10000-point windows retain geometry, bound memory, and upload only new samples", () => {
  let conversions = 0;
  const trail = new FlightTrail(THREE, p => { conversions++; return toWorld(p); });
  const path = track(12000);
  trail.update(path.slice(0, 10000), 0);
  const retained = trail.object.children[10];
  for (let start = 1; start <= 2000; start++) {
    trail.update(path.slice(start, start + 10000), start);
    assert.equal(trail.stats.points, 10000);
    assert.equal(trail.stats.segments, 9999);
    assert.ok(trail.stats.chunks <= Math.ceil(10000 / 256) + 1);
    assert.ok(trail.stats.allocatedSegments <= 10496);
  }
  assert.equal(conversions, path.length);
  assert.ok(trail.object.children.includes(retained));
  assert.equal(geometrySegments(trail), 9999);
  assertFiniteAndBounded(trail);
  trail.dispose();
});

test("same-length replaced history, shorter history, disjoint windows, and empty updates reset correctly", () => {
  const trail = new FlightTrail(THREE, toWorld);
  const path = track(600);
  trail.update(path);
  let disposed = 0;
  const old = trail.object.children[0];
  old.geometry.addEventListener("dispose", () => disposed++);
  const replaced = path.map(p => ({ ...p }));
  replaced[251].y += 7;
  trail.update(replaced);
  assert.equal(disposed, 1);
  assert.notEqual(trail.object.children[0], old);
  assert.equal(trail.stats.segments, 599);
  trail.update(replaced.slice(0, 13));
  assert.equal(trail.stats.segments, 12);
  trail.update(track(5), 9000);
  assert.equal(trail.stats.segments, 4);
  // An adjacent but non-overlapping window must not retain a segment from
  // its now-missing predecessor, even though the global indices are adjacent.
  trail.update(track(3), 9005);
  assert.equal(trail.stats.segments, 2);
  trail.update([]);
  assert.equal(trail.object.children.length, 0);
  assert.equal(trail.stats.points, 0);
  trail.update(track(3), 0);
  assert.equal(trail.stats.segments, 2);
  trail.dispose();
});

test("invalid samples break the trail; zero-length points and vertical segments remain finite", () => {
  const trail = new FlightTrail(THREE, toWorld);
  trail.update([
    { x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 }, { x: 0, y: 0, z: 1 },
    null, { x: 3, y: 3, z: 1 }, { x: 4, y: 3, z: 1 },
    { x: Infinity, y: 0, z: 0 }, { x: 5, y: 5, z: 1 }, { x: 5, y: 6, z: 1 },
    { x: null, y: 0, z: 0 }, { x: 9, y: 9, z: 9 },
  ]);
  assert.equal(trail.stats.segments, 3);
  assertFiniteAndBounded(trail);
  trail.update(null);
  assert.equal(trail.object.children.length, 0);
  trail.dispose();
  const badWorld = new FlightTrail(THREE, p => p.x === 2 ? new THREE.Vector3(NaN, 0, 0) : toWorld(p));
  badWorld.update([0, 1, 2, 3, 4].map(x => ({ x, y: 0, z: 0 })));
  assert.equal(badWorld.stats.segments, 2);
  assertFiniteAndBounded(badWorld);
  badWorld.dispose();
});

test("tube centers follow each actual straight segment and never smooth through corners", () => {
  const trail = new FlightTrail(THREE, p => new THREE.Vector3(p.x, p.y, p.z));
  const path = [{ x: 0, y: 0, z: 0 }, { x: 5, y: 0, z: 0 }, { x: 5, y: 5, z: 0 }];
  trail.update(path);
  const vertices = trail.object.children[0].geometry.getAttribute("position");
  for (let segment = 0; segment < 2; segment++) {
    for (let end = 0; end < 2; end++) {
      const center = new THREE.Vector3();
      for (let side = 0; side < 6; side++) {
        center.add(new THREE.Vector3().fromBufferAttribute(vertices, segment * 12 + end * 6 + side));
      }
      center.divideScalar(6);
      assert.ok(center.distanceTo(new THREE.Vector3(...Object.values(path[segment + end]))) < 1e-6);
    }
  }
  assert.equal(trail.stats.triangles, 24);
  trail.dispose();
});

test("delayed reveal changes only draw ranges and respects trimmed windows", () => {
  const trail = new FlightTrail(THREE, toWorld);
  const path = track(800);
  trail.setVisibleEnd(127);
  trail.update(path);
  assert.equal(geometrySegments(trail), 127);
  const versions = trail.object.children.map(mesh => mesh.geometry.getAttribute("position").version);
  trail.setVisibleEnd(550);
  assert.equal(geometrySegments(trail), 550);
  trail.setVisibleEnd(99);
  assert.equal(geometrySegments(trail), 99);
  assert.deepEqual(trail.object.children.map(mesh => mesh.geometry.getAttribute("position").version), versions);
  trail.update(path.slice(300), 300);
  assert.equal(geometrySegments(trail), 0);
  trail.setVisibleEnd(410);
  assert.equal(geometrySegments(trail), 110);
  trail.setVisibleEnd(NaN);
  assert.equal(geometrySegments(trail), 0);
  trail.setVisibleEnd();
  assert.equal(geometrySegments(trail), 499);
  assertFiniteAndBounded(trail);
  trail.dispose();
});

test("oversized initial histories and repeated disposal stay bounded", () => {
  const trail = new FlightTrail(THREE, toWorld);
  trail.update(track(25000));
  assert.equal(trail.stats.points, 10000);
  assert.equal(trail.stats.segments, 9999);
  let geometriesDisposed = 0;
  let materialsDisposed = 0;
  const chunks = trail.object.children.length;
  trail.object.children.forEach(mesh => mesh.geometry.addEventListener("dispose", () => geometriesDisposed++));
  trail.material.addEventListener("dispose", () => materialsDisposed++);
  const scene = new THREE.Scene();
  scene.add(trail.object);
  trail.dispose();
  trail.dispose();
  trail.update(track(6));
  assert.equal(geometriesDisposed, chunks);
  assert.equal(materialsDisposed, 1);
  assert.equal(scene.children.length, 0);
  assert.equal(trail.object.children.length, 0);
});

function percentile(values, q) {
  const sorted = [...values].sort((a, b) => a - b);
  return Number(sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * q))].toFixed(3));
}

const path = track(14000);
const benchmark = {};
for (const count of [1646, 10000]) {
  const trail = new FlightTrail(THREE, toWorld);
  const buildStart = performance.now();
  trail.update(path.slice(0, count));
  const initialBuildMs = performance.now() - buildStart;
  const frames = [];
  const originalCount = count;
  for (let next = count + 1; next < count + 500; next++) {
    const start = Math.max(0, next - 10000);
    const payload = path.slice(start, next);
    const begin = performance.now();
    trail.update(payload, start);
    frames.push(performance.now() - begin);
  }
  benchmark[originalCount] = { initialBuildMs: Number(initialBuildMs.toFixed(3)),
    appendMedianMs: percentile(frames, 0.5), appendP95Ms: percentile(frames, 0.95),
    finalSegments: trail.stats.segments, finalChunks: trail.stats.chunks };
  trail.dispose();
}
const legacyMs = [];
const oldPoints = path.slice(0, 1646).map(toWorld);
for (let i = 0; i < 5; i++) {
  const begin = performance.now();
  const curve = new THREE.CatmullRomCurve3(oldPoints, false, "centripetal", 0.25);
  const geometry = new THREE.TubeGeometry(curve, oldPoints.length * 8, 0.14, 8, false);
  legacyMs.push(performance.now() - begin);
  benchmark.legacyTriangles = geometry.index.count / 3;
  geometry.dispose();
}
benchmark.legacyRebuildMedianMs = percentile(legacyMs, 0.5);
console.log(JSON.stringify({ checks, benchmark, note: "CPU geometry timing; excludes browser / GPU frame time" }, null, 2));
