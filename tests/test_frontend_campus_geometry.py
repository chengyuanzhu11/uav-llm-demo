"""The rendered running track follows the navigation geometry, not a second oval."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from uav_llm_demo.campus_layout import CAMPUS_LAYOUT, _landmark_bounds
from uav_llm_demo.safety import _intersects


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for campus texture checks")
class FrontendCampusGeometryTests(unittest.TestCase):
    def test_landscape_meshes_fit_shared_safety_envelopes_and_remain_sensor_geometry(self):
        for index, item in enumerate(CAMPUS_LAYOUT["landmarks"]):
            lo, hi = _landmark_bounds(item)
            for other in CAMPUS_LAYOUT["landmarks"][index+1:]:
                a, b = _landmark_bounds(other)
                self.assertFalse(all(lo[i] < b[i] and hi[i] > a[i] for i in range(3)),
                                 (item["id"], other["id"]))
            for structure in CAMPUS_LAYOUT["buildings"] + CAMPUS_LAYOUT["roofs"]:
                x, X, y, Y = structure["footprint"]
                self.assertFalse(lo[0] < X+.39 and hi[0] > x-.39 and lo[1] < Y+.39 and hi[1] > y-.39,
                                 (item["id"], structure["id"]))
            for line in CAMPUS_LAYOUT["roads"] + CAMPUS_LAYOUT["waterways"]:
                radius = line["width"] / 2
                a, b = [lo[0]-radius, lo[1]-radius, -1], [hi[0]+radius, hi[1]+radius, 1]
                self.assertFalse(any(_intersects([*p, 0], [*q, 0], a, b)
                                     for p, q in zip(line["points"], line["points"][1:])), (item["id"], line["id"]))
        script = r'''
import assert from 'node:assert/strict';
import * as THREE from './uav_llm_demo/static/vendor/three.core.js';
import {buildSemanticCampus} from './uav_llm_demo/static/semantic_campus.js';
globalThis.document = {createElement: () => ({getContext: () => new Proxy({
  createLinearGradient: () => ({addColorStop() {}}),
}, {get: (object,key) => object[key] || (() => {})})})};
const layout = ''' + json.dumps(CAMPUS_LAYOUT) + r''';
const group = buildSemanticCampus(THREE,new THREE.Group(),{grid_size:96,campus_layout:layout});
group.updateMatrixWorld(true);
assert.equal(group.userData.environment,true);
for (const item of layout.landmarks) {
  const object = group.getObjectByName('campus_feature_'+item.id);
  assert.ok(object && object.visible && object.userData.cameraOccluder);
  const bounds = new THREE.Box3().setFromObject(object);
  const solid = layout.ground_obstacles.find(o => o.id==='campus_fixture_'+item.id);
  const [x,y,z]=solid.lower_grid, [X,Y,Z]=solid.upper_grid;
  const envelope = new THREE.Box3(new THREE.Vector3((x-47.5)*3,z*5,(y-47.5)*3),
    new THREE.Vector3((X-47.5)*3,Z*5,(Y-47.5)*3)).expandByScalar(.00001);
  assert.ok(envelope.containsBox(bounds),item.id+' rendering exceeds collision envelope');
  assert.equal(group.getObjectByName(solid.id),undefined,'A solid proxy obscures the visible object');
  assert.ok(bounds.max.y<8,'Furnishing intrudes into minimum flight clearance');
}
const trees = group.children.filter(m => m.name.startsWith('campus_tree_'));
assert.equal(trees.find(m=>m.name==='campus_tree_trunks').count,layout.trees.length);
for (const mesh of trees) {
  assert.ok(!mesh.userData.cameraOccluder,'A campus-wide instance batch cannot be a camera blocker');
  mesh.geometry.computeBoundingBox();
  for (let i=0;i<mesh.count;i++) {
    const m=new THREE.Matrix4();mesh.getMatrixAt(i,m);
    const b=mesh.geometry.boundingBox.clone().applyMatrix4(m);
    const owner=layout.trees.find(t=>b.min.x>=(t.x-47.5)*3-t.radius*3-.00001
      && b.max.x<=(t.x-47.5)*3+t.radius*3+.00001
      && b.min.z>=(t.y-47.5)*3-t.radius*3-.00001
      && b.max.z<=(t.y-47.5)*3+t.radius*3+.00001
      && b.min.y>=-.00001 && b.max.y<=t.height*5+.00001);
    assert.ok(owner, 'Tree part exceeds its safety column');
    const vertices=mesh.geometry.attributes.position, point=new THREE.Vector3();
    for (let v=0;v<vertices.count;v++) {
      point.fromBufferAttribute(vertices,v).applyMatrix4(m);
      assert.ok(Math.hypot(point.x-(owner.x-47.5)*3,point.z-(owner.y-47.5)*3)<=owner.radius*3+.00001,
        'Tree foliage exceeds its circular landing exclusion');
    }
  }
}
'''
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                capture_output=True, text=True, timeout=20, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ground_patches_have_one_visible_owner_without_stretched_textures(self):
        script = r'''
import assert from 'node:assert/strict';
import * as THREE from './uav_llm_demo/static/vendor/three.core.js';
import {buildSemanticCampus} from './uav_llm_demo/static/semantic_campus.js';
globalThis.document = {createElement: () => ({width:0,height:0,getContext: () => new Proxy({
  createLinearGradient: () => ({addColorStop() {}}),
}, {get: (object,key) => object[key] || (() => {})})})};
const layout = ''' + json.dumps(CAMPUS_LAYOUT) + r''';
const sport = new Set(['running_track','stadium','football','basketball','tennis','volleyball']);
const elevation = surface => surface.kind === 'launch' ? .007 : sport.has(surface.kind) ? .1 : .07;
const inside = (s,x,z) => s.footprint[0]-.39 < x && x < s.footprint[1]+.39
  && s.footprint[2]-.39 < z && z < s.footprint[3]+.39;
function check(surfaces) {
  const scenario = {grid_size:96,points_of_interest:[],campus_layout:{...layout,surfaces,places:[]}};
  const group = buildSemanticCampus(THREE,new THREE.Group(),scenario);
  group.updateMatrixWorld(true);
  const meshes = group.children.filter(mesh => mesh.name.startsWith('campus_surface_'));
  const ray = new THREE.Raycaster();
  const down = new THREE.Vector3(0,-1,0);
  const points = [];
  // Check both ordinary interiors and the narrow shared bands around borders.
  for (const a of surfaces) for (const b of surfaces) {
    const left=Math.max(a.footprint[0],b.footprint[0])-.39;
    const right=Math.min(a.footprint[1],b.footprint[1])+.39;
    const north=Math.max(a.footprint[2],b.footprint[2])-.39;
    const south=Math.min(a.footprint[3],b.footprint[3])+.39;
    if (left>=right || north>=south) continue;
    for (const u of [.19,.53,.81]) for (const v of [.23,.57,.87]) {
      points.push([left+(right-left)*u,north+(south-north)*v]);
    }
  }
  for (const [x,z] of points) {
    const covering=surfaces.filter(surface => inside(surface,x,z));
    const height=Math.max(...covering.map(elevation));
    const owner=covering.filter(surface => elevation(surface)===height).at(-1);
    ray.set(new THREE.Vector3((x-47.5)*3,1,(z-47.5)*3),down);
    const hits=ray.intersectObjects(meshes,false).filter(hit => Math.abs(hit.point.y-height)<1e-6);
    const names=new Set(hits.map(hit => hit.object.name));
    assert.deepEqual([...names],['campus_surface_'+owner.id],`Coincident ground faces at ${x},${z}`);
    const hit=hits[0], [left,right,north,south]=owner.footprint;
    assert.ok(Math.abs(hit.uv.x-(x-left+.39)/(right-left+.78))<1e-6,'Clipping stretched texture u');
    assert.ok(Math.abs(hit.uv.y-(south+.39-z)/(south-north+.78))<1e-6,'Clipping stretched texture v');
  }
}
check(layout.surfaces);
// Different IDs, partial overlaps on both axes, and a fully covered patch.
check([
  {id:'underlay',kind:'garden',footprint:[4,12,4,12]},
  {id:'hidden',kind:'garden',footprint:[6,8,6,8]},
  {id:'crossing',kind:'paving',footprint:[5,10,3,13]},
  {id:'corner',kind:'garden',footprint:[9,14,8,14]},
]);
'''
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                capture_output=True, text=True, timeout=20, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_running_track_texture_uses_shared_navigation_centerline(self):
        source = (ROOT / "uav_llm_demo/static/semantic_campus.js").read_text(encoding="utf-8")
        declaration = re.search(r"^  function sportTexture\(surface\) \{.*?^  \}", source, re.M | re.S)
        self.assertIsNotNone(declaration)
        surface = next(item for item in CAMPUS_LAYOUT["surfaces"] if item["kind"] == "running_track")
        script = r'''
const assert = require('node:assert/strict');
let currentPath = [], strokes = [], transforms = [];
const ctx = new Proxy({
  beginPath() { currentPath = []; },
  moveTo(x,y) { currentPath.push([x,y]); },
  lineTo(x,y) { currentPath.push([x,y]); },
  closePath() {},
  stroke() { strokes.push({points:currentPath.map(point => [...point]),width:this.lineWidth,color:this.strokeStyle}); },
  scale(x,y) { transforms.push(['scale',x,y]); },
  translate(x,y) { transforms.push(['translate',x,y]); },
}, {get(object,key) { return key in object ? object[key] : () => {}; }});
const canvas = (width,height) => [{width,height},ctx];
const textureFrom = image => image;
''' + declaration.group() + "\nconst surface = " + json.dumps(surface) + r''';
const image = sportTexture(surface);
assert.deepEqual(image,{width:512,height:768});
const band = strokes.find(stroke => stroke.color === '#ac6552');
assert.ok(band);
assert.deepEqual(band.points,surface.geometry.centerline_grid.slice(0,-1));
assert.equal(band.width,surface.geometry.lane_width_m / 3);
assert.equal(strokes.filter(stroke => stroke.color === '#e5c9b2').length,7);
const [left,right,north,south] = surface.footprint;
assert.deepEqual(transforms[0],['scale',512/(right-left+0.78),768/(south-north+0.78)]);
assert.deepEqual(transforms[1],['translate',-left+0.39,-north+0.39]);
// Geometry changes must reach the drawing without changing frontend constants.
const shifted = structuredClone(surface);
shifted.geometry.centerline_grid = shifted.geometry.centerline_grid.map(([x,y]) => [x+0.2,y-0.3]);
strokes=[];
sportTexture(shifted);
assert.deepEqual(strokes.find(stroke => stroke.color === '#ac6552').points,shifted.geometry.centerline_grid.slice(0,-1));
'''
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
