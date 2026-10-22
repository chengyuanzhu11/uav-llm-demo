"""The rendered running track follows the navigation geometry, not a second oval."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from uav_llm_demo.campus_layout import CAMPUS_LAYOUT


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which("node"), "Node.js is required for campus texture checks")
class FrontendCampusGeometryTests(unittest.TestCase):
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
