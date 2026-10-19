// Read-only browser regression: generated fixtures never reach the live mission.
// NODE_PATH may point to an existing Playwright installation. Optional inputs:
// UAV_DEMO_URL, UAV_PYTHON, UAV_CAPTURE_DIR, PLAYWRIGHT_CHROMIUM_EXECUTABLE,
// UAV_TEST_FONT (a local TTF for Chinese labels in headless Linux),
// UAV_CAMPUS_BASELINE (optional previous semantic_campus.js for a pixel comparison).
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { execFileSync } = require("node:child_process");
const { chromium } = require("playwright");

const repo = path.resolve(__dirname, "..");
const baseUrl = new URL(process.env.UAV_DEMO_URL || "http://127.0.0.1:5000");
const captureDir = path.resolve(repo, process.env.UAV_CAPTURE_DIR || "outputs/campus-scene");
const groupName = "beihang_hangzhou_campus";
const expected = { buildings: 50, roofs: 4, pois: 11 };

function sourceFixtures() {
  const pythonCode = String.raw`
import json, os
# Empty api_key alone would fall back to the environment in DeepSeekAgent.
for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
    os.environ.pop(name, None)
from uav_llm_demo import create_app
from uav_llm_demo.agents import DeepSeekAgent, MockLLMAgent
from uav_llm_demo.mission_service import MissionService
from uav_llm_demo.scenarios import list_scenarios

provider_attempts = 0
def forbid_provider(*args, **kwargs):
    global provider_attempts
    provider_attempts += 1
    raise AssertionError("The visual regression must not call a model provider")
DeepSeekAgent._request_json_content = forbid_provider

result = {"scenarios": {"scenarios": list_scenarios()}}
for identifier, agent in (("semantic_campus", DeepSeekAgent(api_key="")),
                          ("urban_crossing", MockLLMAgent())):
    service = MissionService(scenario_id=identifier, agent=agent,
                             target_seed=0, traffic_seed=0)
    try:
        app = create_app(service=service)
        assert service._thread is None, "Fixture generation started a control loop"
        snapshot = service.state()
        assert not snapshot["deepseek_key_configured"]
        assert not snapshot["flight"]["running"]
        assert snapshot["flight"]["planner_calls"] == 0
        result[identifier] = snapshot
        if identifier == "semantic_campus":
            response = app.test_client().get("/")
            assert response.status_code == 200
            result["html"] = response.get_data(as_text=True)
    finally:
        service.close()
assert provider_attempts == 0
result["provider_calls"] = provider_attempts
print(json.dumps(result, ensure_ascii=False))
`;
  const output = execFileSync(process.env.UAV_PYTHON || "python3", ["-c", pythonCode], {
    cwd: repo, encoding: "utf8", maxBuffer: 24 * 1024 * 1024,
    env: { ...process.env, DEEPSEEK_API_KEY: "" }, timeout: 60000,
  });
  return JSON.parse(output);
}

async function renderFrame(page, mode = "overview") {
  return page.evaluate(async (cameraMode) => {
    // Let ResizeObserver finish before sizing the camera and taking a frame.
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    const test = window.__campusTest;
    test.setCameraMode(cameraMode);
    if (cameraMode === "overview") test.controls.update();
    test.renderer.render(test.scene, test.camera);
  }, mode);
}

async function inspectCampus(page) {
  return page.evaluate(async ({ groupName, expected }) => {
    const THREE = await import("/static/vendor/three.module.js");
    const { scene, renderer, camera, state } = window.__campusTest;
    const campus = scene.getObjectByName(groupName);
    if (!campus) throw new Error("Campus group is missing");
    const failures = [], count = { buildings: 0, roofs: 0, pois: 0, labels: 0, meshes: 0, instances: 0 };
    const layout = state.scenario.campus_layout;
    const half = (state.grid_size - 1) / 2;
    campus.updateMatrixWorld(true);
    campus.traverse(object => {
      if (object.userData.campusBuilding) count.buildings++;
      if (object.userData.campusRoof) count.roofs++;
      if (object.name.startsWith("poi_marker_")) count.pois++;
      if (object.name.startsWith("poi_label_")) count.labels++;
      if (object.isMesh) count.meshes++;
      if (object.isInstancedMesh) count.instances += object.count;
      const attributes = [...Object.values(object.geometry?.attributes || {}), object.instanceMatrix].filter(Boolean);
      if (attributes.some(attribute => attribute.array.some(value => !Number.isFinite(value)))) {
        failures.push(`${object.name}: non-finite geometry`);
      }
    });
    for (const [kind, entries] of [["building", layout.buildings], ["roof", layout.roofs]]) {
      for (const item of entries) {
        const mesh = campus.getObjectByName(`campus_${kind}_${item.id}`);
        if (!mesh) { failures.push(`${kind} ${item.id}: missing`); continue; }
        const bounds = new THREE.Box3().setFromObject(mesh);
        const [left, right, north, south] = item.footprint;
        const epsilon = 0.0001;
        if (bounds.min.x < (left - 0.39 - half) * 3 - epsilon ||
            bounds.max.x > (right + 0.39 - half) * 3 + epsilon ||
            bounds.min.z < (north - 0.39 - half) * 3 - epsilon ||
            bounds.max.z > (south + 0.39 - half) * 3 + epsilon ||
            bounds.max.y > item.height * 5 + epsilon) {
          failures.push(`${kind} ${item.id}: exceeds its collision envelope`);
        }
      }
    }
    const joinedWings = ["R1", "R2", "R3"].map(id => {
      const spine = new THREE.Box3().setFromObject(campus.getObjectByName(`campus_building_${id}-spine`));
      const middle = new THREE.Box3().setFromObject(campus.getObjectByName(`campus_building_${id}-middle`));
      const overlap = spine.clone().intersect(middle).getSize(new THREE.Vector3());
      const volume = overlap.x * overlap.y * overlap.z;
      if (volume > 0.0001) failures.push(`${id}: middle/spine cores overlap by ${volume} cubic metres`);
      return { id, overlap: overlap.toArray(), volume };
    });
    for (const [key, value] of Object.entries(expected)) {
      if (count[key] !== value) failures.push(`${key}: expected ${value}, got ${count[key]}`);
    }
    if (count.labels !== expected.pois || state.scenario.points_of_interest.length !== expected.pois) {
      failures.push("POI labels or catalog count do not match");
    }
    const bounds = new THREE.Box3().setFromObject(campus);
    const corners = [];
    for (const x of [bounds.min.x, bounds.max.x]) for (const y of [bounds.min.y, bounds.max.y]) {
      for (const z of [bounds.min.z, bounds.max.z]) corners.push(new THREE.Vector3(x, y, z));
    }
    const maximumCameraDistance = Math.max(...corners.map(point => point.distanceTo(camera.position)));
    if (camera.far <= maximumCameraDistance) failures.push("Camera far plane clips the campus");
    const sunlight = scene.children.find(object => object.isDirectionalLight);
    const skylight = scene.children.find(object => object.isHemisphereLight);
    if (!sunlight || !skylight || sunlight.intensity <= 0 || skylight.intensity <= 0) {
      failures.push("Campus daylight is missing");
    } else if (sunlight.shadow.camera.far <= Math.max(...corners.map(point => point.distanceTo(sunlight.position)))) {
      failures.push("Directional-light far plane cannot cover the campus");
    }
    const selected = document.querySelector("#scenarioSelect option:checked");
    if (selected?.value !== "semantic_campus" || selected.textContent !== state.scenario.name) {
      failures.push("Scenario selector does not show the current campus");
    }
    if (state.deepseek_key_configured || state.flight.planner_calls !== 0) failures.push("Fixture unexpectedly contains API activity");
    if (failures.length) throw new Error(failures.join("; "));
    return { ...count, calls: renderer.info.render.calls, triangles: renderer.info.render.triangles,
      cameraFar: camera.far, maximumCameraDistance: Math.round(maximumCameraDistance),
      selectedScenario: selected.textContent, joinedWings, memory: { ...renderer.info.memory } };
  }, { groupName, expected });
}

async function inspectFacadePixels(page, hasBaseline) {
  return page.evaluate(async ({ groupName, hasBaseline }) => {
    const THREE = await import("/static/vendor/three.module.js");
    const { scene, renderer, camera, controls, state } = window.__campusTest;
    const fixed = scene.getObjectByName(groupName);
    let baseline = null;
    if (hasBaseline) {
      const { buildSemanticCampus } = await import("/__campus_baseline.js");
      baseline = buildSemanticCampus(THREE, new THREE.Group(), state.scenario);
      baseline.name = "campus-flicker-baseline";
      baseline.visible = false;
      scene.add(baseline);
    }
    const original = { camera: camera.position.clone(), quaternion: camera.quaternion.clone(),
      aspect: camera.aspect, size: renderer.getSize(new THREE.Vector2()),
      shadow: renderer.shadowMap.enabled, shadowAuto: renderer.shadowMap.autoUpdate,
      controls: controls.enabled };
    // A fixed framebuffer removes viewport resizing and reduces software GPU cost.
    const width = 1280, height = 960, frames = 32;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    controls.enabled = false;
    const gl = renderer.getContext();
    const rgba = new Uint8Array(width * height * 4);
    const half = (state.grid_size - 1) / 2;
    const X = value => (value - half) * 3;
    const Z = value => (value - half) * 3;
    const building = state.scenario.campus_layout.buildings.find(item => item.id === "R2-north");
    const [left, right, north, south] = building.footprint;
    const east = X(right + 0.39), southEdge = Z(south + 0.39);
    const probes = [3.3, 4.3, 5.1, 6.9, 7.7, 9.1, 10.1].map((y, index) => ({
      id: `south-corner-${index}`, point: new THREE.Vector3(east - 0.14, y, southEdge),
    }));
    for (const y of [2.8, 5.6, 8.4, 11.2]) {
      probes.push({ id: `east-band-${y}`, point: new THREE.Vector3(east, y, Z((north + south) / 2)) });
      probes.push({ id: `south-band-${y}`, point: new THREE.Vector3(X((left + right) / 2), y, southEdge) });
    }
    function pixel(x, y) {
      const channels = [[], [], []];
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const index = ((y + dy) * width + x + dx) * 4;
        for (let c = 0; c < 3; c++) channels[c].push(rgba[index + c]);
      }
      return { rgb: channels.map(values => values.reduce((sum, n) => sum + n, 0) / values.length),
        spread: Math.max(...channels.map(values => Math.max(...values) - Math.min(...values))) };
    }
    function read() { gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, rgba); }
    function invalidateMaterials() {
      const materials = new Set();
      scene.traverse(object => {
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          if (material) materials.add(material);
        }
      });
      // renderer.shadowMap.enabled changes USE_SHADOWMAP. An already compiled
      // material must receive a new version before Three updates that define.
      for (const material of materials) material.needsUpdate = true;
    }
    const results = [], images = [];
    try {
      for (const [name, group] of [["fixed", fixed], ...(baseline ? [["baseline", baseline]] : [])]) {
        fixed.visible = group === fixed;
        if (baseline) baseline.visible = group === baseline;
        const body = group.getObjectByName("campus_building_R2-north");
        group.updateMatrixWorld(true);
        let unshadowedFirstFrame = null;
        for (const shadows of [false, true]) {
          renderer.shadowMap.enabled = shadows;
          renderer.shadowMap.needsUpdate = true;
          invalidateMaterials();
          const record = { model: name, shadows, frames, validSamples: 0, lostSamples: 0,
            temporalLossTransitions: 0, maximumChannelError: 0, perProbe: {}, firstFrameProbes: [] };
          const previousLoss = new Map();
          for (let frame = 0; frame < frames; frame++) {
            // Video position: (36, 35.24, 2.5). Slowly traverse nearby viewpoints.
            const t = frame / (frames - 1);
            const target = new THREE.Vector3(X(36 + (t - 0.5) * 0.9), 12.5,
              Z(35.24 + Math.sin(t * Math.PI * 2) * 0.25));
            camera.position.copy(target).add(new THREE.Vector3(9, 7, 12));
            camera.lookAt(target); camera.updateMatrixWorld(true);
            // Only the view camera moves in this fixture. Its movement does not
            // alter the directional light's static shadow projection.
            renderer.shadowMap.autoUpdate = frame === 0;
            renderer.shadowMap.needsUpdate = frame === 0;
            renderer.render(scene, camera);
            read();
            if (frame === 0 && !shadows) unshadowedFirstFrame = rgba.slice();
            else if (frame === 0) {
              let changedPixels = 0, totalError = 0, maximumError = 0;
              for (let index = 0; index < rgba.length; index += 4) {
                let pixelChanged = false;
                for (let c = 0; c < 3; c++) {
                  const error = Math.abs(rgba[index + c] - unshadowedFirstFrame[index + c]);
                  totalError += error; maximumError = Math.max(maximumError, error);
                  if (error > 1) pixelChanged = true;
                }
                if (pixelChanged) changedPixels++;
              }
              record.shadowContrast = { changedPixels, maximumError,
                meanChannelError: totalError / (width * height * 3) };
            }
            const projected = probes.map(probe => {
              const p = probe.point.clone().project(camera);
              const x = Math.floor((p.x + 1) * width / 2), y = Math.floor((p.y + 1) * height / 2);
              if (p.z < -1 || p.z > 1 || x < 2 || x >= width - 2 || y < 2 || y >= height - 2) return null;
              return { id: probe.id, x, y, actual: pixel(x, y) };
            }).filter(Boolean);
            if (frame === 0 || frame === Math.floor(frames / 2)) {
              images.push({ name: `${name}-${shadows ? "shadow" : "unshadowed"}-${frame}.png`,
                data: renderer.domElement.toDataURL("image/png") });
            }
            // Keep the same shadow map, camera and lighting. Hide only the main
            // wall behind the trim to obtain its unobstructed pixel reference.
            renderer.shadowMap.autoUpdate = false;
            renderer.shadowMap.needsUpdate = false;
            body.visible = false;
            renderer.render(scene, camera);
            read();
            if (frame === 0) {
              images.push({ name: `${name}-${shadows ? "shadow" : "unshadowed"}-trim-reference.png`,
                data: renderer.domElement.toDataURL("image/png") });
            }
            body.visible = true;
            for (const probe of projected) {
              const reference = pixel(probe.x, probe.y);
              if (frame === 0) record.firstFrameProbes.push({ id: probe.id, x: probe.x, y: probe.y,
                actual: probe.actual, reference });
              // Exclude silhouettes/anti-aliasing boundaries rather than counting
              // their changing raster coverage as a facade visibility failure.
              if (reference.spread > 6) continue;
              const error = Math.max(...reference.rgb.map((n, c) => Math.abs(n - probe.actual.rgb[c])));
              const lost = error > 12;
              const perProbe = record.perProbe[probe.id] ||= { valid: 0, lost: 0, transitions: 0, maxError: 0 };
              record.validSamples++; perProbe.valid++;
              record.maximumChannelError = Math.max(record.maximumChannelError, error);
              perProbe.maxError = Math.max(perProbe.maxError, error);
              if (lost) { record.lostSamples++; perProbe.lost++; }
              if (previousLoss.has(probe.id) && previousLoss.get(probe.id) !== lost) {
                record.temporalLossTransitions++; perProbe.transitions++;
              }
              previousLoss.set(probe.id, lost);
            }
          }
          record.lossRate = record.validSamples ? record.lostSamples / record.validSamples : null;
          record.cornerSamples = Object.entries(record.perProbe).filter(([id]) => id.includes("corner"))
            .reduce((sum, [, probe]) => sum + probe.valid, 0);
          record.bandSamples = Object.entries(record.perProbe).filter(([id]) => id.includes("band"))
            .reduce((sum, [, probe]) => sum + probe.valid, 0);
          console.info("campus-pixel-progress " + JSON.stringify({ model: name, shadows,
            validSamples: record.validSamples, lostSamples: record.lostSamples,
            cornerSamples: record.cornerSamples, bandSamples: record.bandSamples,
            transitions: record.temporalLossTransitions, shadowContrast: record.shadowContrast }));
          results.push(record);
        }
      }
      return { framebuffer: { width, height }, frames, worldPosition: { x: 36, y: 35.24, z: 2.5 },
        reference: "same camera and frozen shadow map; main R2 north wall hidden behind trim",
        results, images };
    } finally {
      fixed.visible = true;
      if (baseline) {
        scene.remove(baseline);
        const geometries = new Set(), materials = new Set(), textures = new Set();
        baseline.traverse(object => {
          if (object.isInstancedMesh) object.dispose();
          if (object.geometry) geometries.add(object.geometry);
          for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
            if (material) materials.add(material);
            if (material?.map) textures.add(material.map);
          }
        });
        for (const resource of [...geometries, ...materials, ...textures]) resource.dispose();
      }
      renderer.shadowMap.enabled = original.shadow;
      renderer.shadowMap.autoUpdate = original.shadowAuto;
      invalidateMaterials();
      renderer.setSize(original.size.x, original.size.y, false);
      camera.aspect = original.aspect;
      camera.position.copy(original.camera); camera.quaternion.copy(original.quaternion);
      camera.updateProjectionMatrix();
      controls.enabled = original.controls;
    }
  }, { groupName, hasBaseline });
}

async function checkViewport(page, width, height) {
  await page.setViewportSize({ width, height });
  await renderFrame(page);
  const layout = await page.evaluate(() => {
    const box = selector => {
      const rect = document.querySelector(selector).getBoundingClientRect();
      return { x: rect.x, y: rect.y, width: rect.width, height: rect.height,
        right: rect.right, bottom: rect.bottom };
    };
    return { documentWidth: document.documentElement.scrollWidth,
      scene: box("#sceneContainer"), inspector: box("#decisionInspector"),
      canvas: box("#sceneContainer canvas") };
  });
  assert.ok(layout.documentWidth <= width + 1, `${width}x${height}: horizontal page overflow`);
  assert.ok(layout.scene.width > 150 && layout.scene.height > 150, `${width}x${height}: scene collapsed`);
  assert.ok(layout.scene.x >= -1 && layout.scene.right <= width + 1, `${width}x${height}: scene exceeds viewport`);
  assert.ok(layout.inspector.x >= -1 && layout.inspector.right <= width + 1, `${width}x${height}: inspector exceeds viewport`);
  assert.ok(Math.abs(layout.canvas.width - layout.scene.width) <= 1 &&
    Math.abs(layout.canvas.height - layout.scene.height) <= 1, `${width}x${height}: canvas did not resize`);
  if (width <= 700) assert.ok(layout.inspector.y >= layout.scene.bottom - 1, "Mobile inspector overlaps scene");
  else assert.ok(layout.inspector.x >= layout.scene.right - 1, "Desktop inspector overlaps scene");
  return { width, height, ...layout };
}

async function main() {
  const fixtures = sourceFixtures();
  fs.mkdirSync(captureDir, { recursive: true });
  let activeState = fixtures.semantic_campus;
  const writes = [], unexpectedRequests = [], errors = [], screenshots = [], viewports = [];
  let appSource = fs.readFileSync(path.join(repo, "uav_llm_demo/static/app.js"), "utf8");
  assert.ok(appSource.includes("const sceneContainer ="), "Frontend test insertion point changed");
  appSource = appSource.replace("const sceneContainer =", "await window.__campusTestFontReady;\nconst sceneContainer =");
  appSource += "\nwindow.__campusTest = {scene, renderer, camera, controls, acceptState, setCameraMode, stopPolling() {pollActive = false;}, get state() {return state;}};\n";
  const font = process.env.UAV_TEST_FONT ? fs.readFileSync(process.env.UAV_TEST_FONT) : null;
  const baselineSource = process.env.UAV_CAMPUS_BASELINE
    ? fs.readFileSync(process.env.UAV_CAMPUS_BASELINE, "utf8") : null;
  const browser = await chromium.launch({
    headless: true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}),
    args: ["--no-sandbox", "--disable-dev-shm-usage", "--use-gl=angle", "--use-angle=swiftshader",
      "--enable-webgl", "--enable-unsafe-swiftshader"],
  });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1, serviceWorkers: "block" });
    page.setDefaultTimeout(30000);
    page.on("pageerror", error => errors.push(error.message));
    page.on("console", message => {
      if (message.text().startsWith("campus-pixel-progress ")) process.stderr.write(message.text() + "\n");
    });
    await page.route("**/*", async route => {
      const request = route.request(), url = new URL(request.url());
      if (request.method() !== "GET") {
        writes.push({ method: request.method(), path: url.pathname });
        return route.abort();
      }
      if (url.origin !== baseUrl.origin) {
        unexpectedRequests.push({ origin: url.origin, path: url.pathname });
        return route.abort();
      }
      if (url.pathname === "/") return route.fulfill({ contentType: "text/html", body: fixtures.html });
      if (url.pathname === "/api/state") return route.fulfill({ contentType: "application/json", body: JSON.stringify(activeState) });
      if (url.pathname === "/api/scenarios") return route.fulfill({ contentType: "application/json", body: JSON.stringify(fixtures.scenarios) });
      if (url.pathname === "/static/app.js") return route.fulfill({ contentType: "text/javascript", body: appSource });
      if (url.pathname === "/__campus_test_font.ttf" && font) return route.fulfill({ contentType: "font/ttf", body: font });
      if (url.pathname === "/__campus_baseline.js" && baselineSource) {
        return route.fulfill({ contentType: "text/javascript", body: baselineSource });
      }
      if (url.pathname.startsWith("/static/")) {
        const staticRoot = path.join(repo, "uav_llm_demo/static");
        const filename = path.resolve(staticRoot, decodeURIComponent(url.pathname.slice("/static/".length)));
        if (!filename.startsWith(staticRoot + path.sep) || !fs.existsSync(filename)) {
          unexpectedRequests.push({ path: url.pathname });
          return route.abort();
        }
        const extension = path.extname(filename);
        const contentType = ({ ".js": "text/javascript", ".css": "text/css", ".png": "image/png",
          ".svg": "image/svg+xml", ".woff2": "font/woff2" })[extension] || "application/octet-stream";
        return route.fulfill({ contentType, body: fs.readFileSync(filename) });
      }
      if (url.pathname === "/favicon.ico") return route.fulfill({ status: 204, body: "" });
      unexpectedRequests.push({ path: url.pathname });
      return route.abort();
    });
    await page.addInitScript(hasFont => {
      window.__campusTestFontReady = Promise.resolve();
      if (hasFont) {
        const face = new FontFace("Microsoft YaHei", "url(/__campus_test_font.ttf)");
        document.fonts.add(face);
        window.__campusTestFontReady = face.load();
      }
    }, Boolean(font));
    await page.goto(new URL("/", baseUrl).href, { waitUntil: "domcontentloaded" });
    await page.waitForFunction(name => window.__campusTest?.scene.getObjectByName(name) &&
      window.__campusTest.renderer.info.render.frame > 0, groupName);
    // One software WebGL context is enough; manually render all subsequent checks.
    await page.evaluate(() => {
      window.__campusTest.renderer.setAnimationLoop(null);
      window.__campusTest.stopPolling();
    });
    await page.waitForFunction(() => document.querySelector("#scenarioSelect option:checked")?.value === "semantic_campus");
    viewports.push(await checkViewport(page, 1440, 900));
    const initial = await inspectCampus(page);
    async function capture(name) {
      const destination = path.join(captureDir, name);
      await page.screenshot({ path: destination, animations: "disabled", timeout: 60000 });
      screenshots.push(destination);
    }
    await capture("campus-overview-1440.png");
    await renderFrame(page, "follow");
    const follow = await page.evaluate(() => {
      const { camera, controls } = window.__campusTest;
      return { enabledOrbit: controls.enabled, camera: camera.position.toArray(),
        selected: document.querySelector("#followBtn").classList.contains("active") };
    });
    assert.equal(follow.enabledOrbit, false, "Follow mode did not disable orbit");
    assert.equal(follow.selected, true, "Follow mode is not selected");
    assert.ok(follow.camera.every(Number.isFinite), "Follow camera has invalid coordinates");
    await capture("campus-follow-1440.png");
    await renderFrame(page);
    const orbit = await page.evaluate(() => {
      const { renderer, scene, camera, controls } = window.__campusTest;
      const before = camera.position.clone();
      camera.position.add(new camera.position.constructor(4, 0, -4));
      controls.update(); renderer.render(scene, camera);
      return camera.position.distanceTo(before);
    });
    assert.ok(orbit > 1, "Overview camera did not respond to orbit adjustment");

    const facadePixels = await inspectFacadePixels(page, Boolean(baselineSource));
    for (const image of facadePixels.images) {
      const destination = path.join(captureDir, image.name);
      fs.writeFileSync(destination, Buffer.from(image.data.split(",")[1], "base64"));
      screenshots.push(destination);
    }
    delete facadePixels.images;
    fs.writeFileSync(path.join(captureDir, "facade-pixels.json"), JSON.stringify(facadePixels, null, 2));
    for (const result of facadePixels.results.filter(result => result.shadows)) {
      assert.ok(result.shadowContrast.changedPixels > 100,
        `${result.model}: toggling shadow shaders did not change rendered pixels`);
    }
    for (const result of facadePixels.results.filter(result => result.model === "fixed")) {
      assert.ok(result.validSamples >= 100, `Insufficient facade interior pixels with shadows=${result.shadows}`);
      assert.ok(result.cornerSamples >= 100, `Insufficient corner-column interior pixels with shadows=${result.shadows}`);
      assert.ok(result.bandSamples >= 50, `Insufficient floor-band interior pixels with shadows=${result.shadows}`);
      assert.equal(result.lostSamples, 0, `Fixed facade loses ${result.lostSamples}/${result.validSamples} trim samples with shadows=${result.shadows}`);
    }
    if (baselineSource) {
      for (const shadows of [false, true]) {
        const before = facadePixels.results.find(result => result.model === "baseline" && result.shadows === shadows);
        assert.ok(before.lostSamples > 0, `Baseline did not reproduce trim depth conflict with shadows=${shadows}`);
        assert.ok(before.temporalLossTransitions > 0, `Baseline did not reproduce temporal flicker with shadows=${shadows}`);
      }
    }

    // Observe the original objects themselves; object counts alone miss GPU leaks.
    await page.evaluate(name => {
      const { scene } = window.__campusTest;
      const group = scene.getObjectByName(name);
      const probe = { group, instances: new Set(), geometries: new Set(), materials: new Set(), textures: new Set(),
        disposedInstances: new Set(), disposedGeometries: new Set(), disposedMaterials: new Set(), disposedTextures: new Set() };
      scene.traverse(object => { if (object.isInstancedMesh) probe.instances.add(object); });
      group.traverse(object => {
        if (object.geometry) probe.geometries.add(object.geometry);
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) {
          if (material) probe.materials.add(material);
          if (material?.map) probe.textures.add(material.map);
        }
      });
      for (const [source, disposed] of [["instances", "disposedInstances"], ["geometries", "disposedGeometries"],
        ["materials", "disposedMaterials"], ["textures", "disposedTextures"]]) {
        for (const resource of probe[source]) resource.addEventListener("dispose", () => probe[disposed].add(resource));
      }
      window.__campusTest.disposalProbe = probe;
    }, groupName);
    activeState = { ...fixtures.urban_crossing, sequence: activeState.sequence + 100 };
    await page.evaluate(snapshot => window.__campusTest.acceptState(snapshot), activeState);
    await renderFrame(page);
    const disposed = await page.evaluate(name => {
      const { scene, disposalProbe: probe } = window.__campusTest;
      if (scene.getObjectByName(name) || probe.group.parent) throw new Error("Previous campus group remains attached");
      const counts = {};
      for (const [source, released] of [["instances", "disposedInstances"], ["geometries", "disposedGeometries"],
        ["materials", "disposedMaterials"], ["textures", "disposedTextures"]]) {
        counts[source] = probe[source].size;
        if (probe[source].size === 0 || probe[source].size !== probe[released].size) {
          throw new Error(`${source}: released ${probe[released].size}/${probe[source].size}`);
        }
      }
      delete window.__campusTest.disposalProbe;
      return counts;
    }, groupName);
    activeState = { ...fixtures.semantic_campus, sequence: activeState.sequence + 100 };
    await page.evaluate(snapshot => window.__campusTest.acceptState(snapshot), activeState);
    await renderFrame(page);
    const restored = await inspectCampus(page);
    for (const field of ["meshes", "instances", "buildings", "roofs", "pois", "labels"]) {
      assert.equal(restored[field], initial[field], `${field} accumulated after scenario round trip`);
    }
    assert.equal(await page.evaluate(name => window.__campusTest.scene.children.filter(child => child.name === name).length,
      groupName), 1, "More than one campus group exists");
    viewports.push(await checkViewport(page, 980, 720));
    await capture("campus-overview-980.png");
    viewports.push(await checkViewport(page, 390, 844));
    assert.deepEqual(writes, [], "The frontend attempted state-changing requests");
    assert.deepEqual(unexpectedRequests, [], "Unexpected requests escaped the fixture allowlist");
    assert.deepEqual(errors, [], "Browser errors occurred");
    const report = { ok: true, source: "isolated current-source fixtures", apiWrites: writes.length,
      providerCalls: fixtures.provider_calls, initial, restored, disposed, follow, facadePixels, viewports, screenshots, errors };
    fs.writeFileSync(path.join(captureDir, "browser-check.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error(error.stack || String(error)); process.exitCode = 1; });
