"""Campus plates identify models while retaining distinct navigation points."""
from pathlib import Path
import json
import shutil
import subprocess
import unittest

from uav_llm_demo.campus_layout import CAMPUS_LAYOUT


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for campus label checks")
class FrontendCampusLabelTests(unittest.TestCase):
    def run_labels(self, checks):
        script = r'''
import assert from 'node:assert/strict';
const THREE = await import(''' + repr((ROOT / "uav_llm_demo/static/vendor/three.module.js").as_uri()) + r''');
const {createCampusPlaceLabels} = await import(''' + repr((ROOT / "uav_llm_demo/static/semantic_campus.js").as_uri()) + r''');
globalThis.document = {createElement: () => ({width:0,height:0,getContext: () => new Proxy({},
  {get: (object,key) => object[key] || (() => {})})})};
let textureCount = 0;
const rendering = {...THREE, CanvasTexture:class extends THREE.CanvasTexture {
  constructor(...args) {super(...args); textureCount++;}
}};
const site = (id,code,label,x,y,z=3) => ({id,code,label,
  position:{x:x+3,y:y+3,z:2},label_position:{x,y,z}});
const places = [site('c1','C1','C1 科研楼',48,48,4),site('c2','C2','C2 科研楼',48.1,48.1,4),
  site('d1','D1','D1 教学楼',48.2,48.2,4),site('stadium','S1','田径场',25,78,0.3),
  site('home','南门起降点','南门起降点',48,92,0.3)];
const group = new THREE.Group();
const controller = createCampusPlaceLabels(rendering,group,{grid_size:96,campus_layout:{places}});
const camera = new THREE.PerspectiveCamera(54,800/600,.1,1000);
camera.position.set(0,200,240); camera.lookAt(0,0,0);
const pointer = new THREE.Vector2(2,2);
const update = active => controller.update(camera,pointer,active,800,600);
update(null);
''' + checks
        result = subprocess.run([shutil.which("node"), "--input-type=module", "-e", script],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_plates_stay_at_model_and_crowded_labels_are_hidden(self):
        self.run_labels(r'''
assert.equal(controller.entries.length,places.length);
for (const [index,entry] of controller.entries.entries()) {
  const place=places[index];
  assert.equal(entry.code.userData.labelText,place.code);
  assert.equal(entry.code.userData.placeLabel,place.label);
  assert.equal(entry.name.visible,false);
  assert.equal(entry.ring.visible,false);
  assert.equal(entry.code.material.depthTest,true);
  assert.equal(entry.code.material.sizeAttenuation,true);
  assert.deepEqual(entry.anchor.toArray(),[(place.label_position.x-47.5)*3,
    place.label_position.z*5,(place.label_position.y-47.5)*3]);
  assert.deepEqual(entry.code.position.toArray(),entry.anchor.toArray());
  assert.notEqual(entry.anchor.x,entry.destination.x);
  assert.equal(entry.ring.position.x,entry.destination.x);
}
assert.equal(group.children.some(object=>object.isLine),false);
assert.equal(controller.entries.slice(0,3).filter(entry=>entry.code.visible).length,1);
''')

    def test_only_hover_or_active_task_expands_name(self):
        self.run_labels(r'''
update('stadium');
const stadium=controller.entries.find(entry=>entry.id==='stadium');
assert.equal(stadium.name.visible,true);
assert.equal(stadium.code.visible,false);
assert.equal(stadium.name.userData.labelText,'S1 · 田径场');
assert.equal(stadium.name.renderOrder,8);
assert.equal(stadium.ring.visible,true);
assert.deepEqual(stadium.name.position.toArray(),stadium.anchor.toArray());
update(null);
const hovered=controller.entries.find(entry=>entry.code.visible);
pointer.set(hovered.x/800*2-1,1-hovered.y/600*2);
update(null); update(null);
assert.equal(controller.hoveredId,hovered.id);
assert.equal(hovered.name.visible,true);
assert.equal(hovered.ring.visible,false);
pointer.set(2,2); update(null);
assert.equal(controller.hoveredId,null);
assert.equal(hovered.name.visible,false);
// Nearness alone must not fill the follow view with full names.
camera.position.copy(stadium.anchor).add(new THREE.Vector3(0,20,30));
camera.lookAt(stadium.anchor); update(null);
assert.equal(stadium.name.visible,false);
assert.equal(stadium.code.visible,true);
''')

    def test_moving_camera_does_not_move_anchors_or_allocate_resources(self):
        self.run_labels(r'''
const entry=controller.entries[0];
const textureIds=controller.entries.flatMap(entry=>[entry.code.material.map.uuid,entry.name.material.map.uuid]);
const anchors=controller.entries.map(entry=>entry.anchor.toArray());
const initialTextures=textureCount;
camera.position.copy(entry.anchor).add(new THREE.Vector3(0,20,30));
camera.lookAt(entry.anchor);
for(let tick=0;tick<240;tick++) {
  camera.position.x+=.01; camera.lookAt(entry.anchor); update(tick%2?'c1':null);
}
assert.equal(textureCount,initialTextures);
assert.deepEqual(controller.entries.flatMap(entry=>[entry.code.material.map.uuid,entry.name.material.map.uuid]),textureIds);
for(const [index,item] of controller.entries.entries()) for(const object of [item.code,item.name]) {
  assert.deepEqual(object.position.toArray(),anchors[index]);
  assert.ok(object.scale.toArray().every(Number.isFinite));
}
''')

    def test_roof_attachment_and_hidden_plate_cannot_be_hovered(self):
        self.run_labels(r'''
const model=new THREE.Group();
const roof=new THREE.Mesh(new THREE.BoxGeometry(20,10,20),new THREE.MeshBasicMaterial());
roof.userData.campusBuilding='C1';roof.name='campus_building_C1';roof.position.set(0,5,0);roof.userData.cameraOccluder=true;model.add(roof);
const building={...site('roof','C1','科研楼 C1',47.5,47.5,8),building_ids:['C1']};
const attached=createCampusPlaceLabels(rendering,model,{grid_size:96,campus_layout:{places:[building]}});
const entry=attached.entries[0];
assert.ok(Math.abs(entry.anchor.y-10.18)<.00001,'Must attach to rendered roof, not navigation ceiling');
// A slatted canopy must not leave the plate underneath its ribs.
const rib=new THREE.Mesh(new THREE.BoxGeometry(20,.2,.3),new THREE.MeshBasicMaterial());
rib.position.set(0,14,-.45);rib.userData.cameraOccluder=true;model.add(rib);
const raised=createCampusPlaceLabels(rendering,model,{grid_size:96,campus_layout:{places:[building]}});
assert.ok(raised.entries[0].anchor.y>14.2);
assert.ok(Math.abs(raised.entries[0].anchor.z+.45)<.00001);
// A close wall occludes the plate. Hover must not reveal a name through it.
const wall=new THREE.Mesh(new THREE.BoxGeometry(30,40,1),new THREE.MeshBasicMaterial());
wall.position.set(0,15,30);wall.userData.cameraOccluder=true;model.add(wall);
const hidden=createCampusPlaceLabels(rendering,model,{grid_size:96,campus_layout:{places:[building]}});
camera.position.set(0,15,60);camera.lookAt(entry.anchor);
hidden.update(camera,pointer,null,800,600);
const blocked=hidden.entries[0];
pointer.set(blocked.x/800*2-1,1-blocked.y/600*2);
hidden.update(camera,pointer,null,800,600);
assert.equal(hidden.hoveredId,null);
assert.equal(blocked.name.visible,false);
''')

    def test_catalog_keeps_all_places_without_overview_label_relocation(self):
        self.run_labels("const layout = " + json.dumps(CAMPUS_LAYOUT) + r''';
for (const [width,height] of [[1000,750],[700,750],[1400,850]]) {
  const all=createCampusPlaceLabels(rendering,new THREE.Group(),{grid_size:96,campus_layout:layout});
  const overview=new THREE.PerspectiveCamera(54,width/height,.1,1200);
  const size=288,fit=Math.max(1,1.1/overview.aspect);
  overview.position.set(size*.12*fit,size*.73*fit,size*.88*fit);
  overview.lookAt(0,0,size*.08);
  all.update(overview,pointer,null,width,height);
  assert.equal(all.entries.length,layout.places.length);
  const visible=all.entries.filter(entry=>entry.code.visible);
  assert.ok(visible.length>5 && visible.length<layout.places.length);
  for(const entry of all.entries) {
    assert.equal(entry.name.visible,false);
    assert.deepEqual(entry.code.position.toArray(),entry.anchor.toArray());
  }
  for(let i=0;i<visible.length;i++) for(let j=0;j<i;j++) {
    const a=visible[i],b=visible[j];
    assert.ok(Math.abs(a.x-b.x)>=(a.width+b.width)/2 || Math.abs(a.y-b.y)>=(a.height+b.height)/2);
  }
}
''')


if __name__ == "__main__":
    unittest.main()
