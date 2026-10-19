/** Deterministic browser replay of captured Mock control telemetry; zero network writes. */
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const fixturePath = process.env.UAV_FOLLOW_FIXTURE
  || path.join(root, 'outputs/follow-camera/flight-fixture.json.gz');
const output = path.resolve(process.env.UAV_CAPTURE_DIR || path.join(root, 'outputs/follow-camera'));
if (!fs.existsSync(fixturePath)) {
  throw new Error(`Missing flight fixture: ${fixturePath}\nGenerate it with: python scripts/capture_follow_fixture.py --output "${fixturePath}"`);
}
const raw = fs.readFileSync(fixturePath);
const fixture = JSON.parse(fixturePath.endsWith('.gz') ? zlib.gunzipSync(raw) : raw);
assert.equal(fixture.final.deepseek_key_configured, false, 'Replay fixture must contain no configured key');
assert.equal(fixture.final.diagnostics.usage.requests, 0, 'Fixture must be produced without provider calls');
assert.ok(fixture.final.milestones.task_completed, 'Fixture must complete the actual outbound and return flight');
fs.mkdirSync(output, { recursive: true });

const probeSource = `
const replayClock = {now: 0, delta: 1 / 60};
const replayIngest = motionTimeline.ingest.bind(motionTimeline);
const replaySample = motionTimeline.sample.bind(motionTimeline);
motionTimeline.ingest = samples => replayIngest(samples, replayClock.now);
motionTimeline.sample = () => {
  const frame = replaySample(replayClock.now);
  if (window.followProbe) window.followProbe.frame = frame;
  return frame;
};
timer.update = () => {};
timer.getDelta = () => replayClock.delta;
timer.getElapsed = () => replayClock.now;
window.followProbe = {
  THREE, scene, camera, renderer, timer, motionTimeline, followRig, acceptState, animate, setCameraMode,
  get uavGroup() { return uavGroup; }, get flightTrail() { return flightTrail; },
  get trailHead() { return trailHead; }, get obstacles() { return cameraObstacles; },
  get state() { return state; }, get rotors() { return rotorMeshes; },
  get birds() { return birdMeshes; }, get goalStar() { return goalStar; },
  clock: replayClock, frame: null, stopPolling() { pollActive = false; },
};
`;
const source = fs.readFileSync(path.join(root, 'uav_llm_demo/static/app.js'), 'utf8');
assert.ok(source.includes('renderer.setAnimationLoop(animate);'), 'Production animation entrypoint changed');
const appSource = source.replace('renderer.setAnimationLoop(animate);', 'renderer.setAnimationLoop(null);') + probeSource;
const mime = { '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.ttf': 'font/ttf' };

(async () => {
  const browser = await chromium.launch({
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE } : {}),
    headless: true,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl', '--enable-unsafe-swiftshader', '--no-sandbox'],
  });
  const errors = [];
  const writes = [];
  const unknownRequests = [];
  let requests = 0;
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      requests += 1;
      const request = route.request();
      const url = new URL(request.url());
      if (request.method() !== 'GET') {
        writes.push({ method: request.method(), path: url.pathname });
        await route.abort();
        return;
      }
      if (url.origin !== 'http://follow-replay.invalid') {
        unknownRequests.push(url.origin + url.pathname);
        await route.abort();
        return;
      }
      if (url.pathname === '/') {
        let html = fixture.html;
        if (process.env.UAV_TEST_FONT) {
          html = html.replace('</head>', '<style>@font-face{font-family:ReplayChinese;src:url(/replay-font.ttf)}body,button,input,select,textarea{font-family:ReplayChinese,sans-serif!important}</style></head>');
          html = html.replace('<script type="module" src="/static/app.js"></script>',
            '<script type="module">await document.fonts.load("16px ReplayChinese"); await import("/static/app.js");</script>');
        }
        await route.fulfill({ contentType: 'text/html', body: html });
      } else if (url.pathname === '/api/state') {
        await route.fulfill({ json: fixture.initial });
      } else if (url.pathname === '/api/scenarios') {
        await route.fulfill({ json: fixture.catalog });
      } else if (url.pathname === '/static/app.js') {
        await route.fulfill({ contentType: 'text/javascript', body: appSource });
      } else if (url.pathname === '/replay-font.ttf' && process.env.UAV_TEST_FONT) {
        await route.fulfill({ contentType: 'font/ttf', body: fs.readFileSync(process.env.UAV_TEST_FONT) });
      } else if (url.pathname.startsWith('/static/')) {
        const staticRoot = path.join(root, 'uav_llm_demo/static');
        const localPath = path.resolve(staticRoot, decodeURIComponent(url.pathname.slice('/static/'.length)));
        assert.ok(localPath.startsWith(staticRoot + path.sep), 'Static path must remain within repository assets');
        await route.fulfill({ contentType: mime[path.extname(localPath)] || 'application/octet-stream', body: fs.readFileSync(localPath) });
      } else {
        unknownRequests.push(url.pathname);
        await route.abort();
      }
    });
    await page.goto('http://follow-replay.invalid/', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.followProbe?.uavGroup && window.followProbe.state, null, { timeout: 30000 });
    await page.evaluate(({ initial, final, samples }) => {
      const p = window.followProbe;
      p.stopPolling();
      const ensure = (test, message) => { if (!test) throw new Error(message); };
      const clone = value => structuredClone(value);
      let sequence = 10000;
      const runId = final.diagnostics.run_id;
      const finalPath = final.continuous_path;
      const duration = samples.at(-1).t;
      const targetAt = final.milestones.target_reached_at;
      const returnedAt = final.milestones.base_returned_at;
      const realRender = p.renderer.render.bind(p.renderer);
      p.renderer.render = () => {};
      p.setCameraMode('follow');
      let step = 0;
      let nextPoll = 0;
      let pollIndex = 0;
      let sampleIndex = 0;
      let lastCamera = null;
      let previousFullChunk = null;
      let firstFullPositions = null;
      let firstFullDisposed = 0;
      let lastEmergency = 0;
      let lastSweep = 0;
      let collectedAt = null;
      const ray = new p.THREE.Ray();
      const hit = new p.THREE.Vector3();
      const times = [];
      const polls = [];
      const failures = [];
      const clampEvents = [];
      const totals = {frames: 0, maxCameraStep: 0, maxCameraSpeed: 0, maxHeadError: 0,
        maxVisibleTriangles: 0, maxTrafficError: 0, maxCpuMs: 0, frameResets: 0, timelineStalls: 0};
      let lastDisplayTime = null;
      function packet(at) {
        while (sampleIndex + 1 < samples.length && samples[sampleIndex + 1].t <= at + 1e-8) sampleIndex++;
        const sample = samples[sampleIndex];
        const payload = clone(final);
        payload.sequence = sequence++;
        payload.diagnostics.run_id = runId;
        payload.simulation_time = sample.t;
        payload.uav = {x: sample.position[0], y: sample.position[1], z: sample.position[2]};
        payload.motion_samples = samples.slice(Math.max(0, sampleIndex - 31), sampleIndex + 1);
        payload.continuous_path = finalPath.slice(0, sample.path_index + 1);
        payload.path = payload.continuous_path;
        payload.path_start_index = 0;
        payload.status = sample.t >= returnedAt ? 'completed' : 'active';
        payload.phase = sample.t >= returnedAt ? 'completed' : sample.t >= targetAt ? 'returning_home' : 'fly_to_target';
        payload.flight.running = !sample.landed && sample.t < returnedAt;
        payload.flight.stage = sample.landed ? 'landed' : sample.t < 6 ? 'takeoff'
          : sample.t > returnedAt - 6 ? 'landing' : 'cruise';
        payload.flight.mode = sample.t >= returnedAt ? 'completed' : sample.landed ? 'idle' : 'tracking';
        payload.flight.speed_mps = Math.hypot(...sample.velocity);
        payload.milestones.target_reached = sample.t >= targetAt;
        payload.milestones.base_returned = sample.t >= returnedAt;
        payload.milestones.task_completed = sample.t >= returnedAt;
        payload.milestones.stationary = sample.landed;
        payload.milestones.target_reached_at = sample.t >= targetAt ? targetAt : null;
        payload.milestones.base_returned_at = sample.t >= returnedAt ? returnedAt : null;
        payload.dynamic_objects.forEach(item => {
          const bird = sample.traffic[item.id];
          if (bird) item.position = {x: bird[0], y: bird[1], z: bird[2]};
        });
        return payload;
      }
      const world = position => new p.THREE.Vector3((position[0] - (final.grid_size - 1) / 2) * 3,
        position[2] * 5, (position[1] - (final.grid_size - 1) / 2) * 3);
      function violation(kind, extra = {}) {
        if (failures.length < 20) failures.push({kind, time: p.clock.now, displayTime: p.frame?.time,
          uav: p.uavGroup.position.toArray(), camera: p.camera.position.toArray(), ...extra});
      }
      p.replay = {
        duration,
        advance(end, fps = 60) {
          p.clock.delta = 1 / fps;
          while (step / fps <= end + 1e-8) {
            const now = step / fps;
            p.clock.now = now;
            const started = performance.now();
            if (now + 1e-8 >= nextPoll) {
              const ingestStart = performance.now();
              ensure(p.acceptState(packet(now)), 'Fresh packet rejected');
              polls.push(performance.now() - ingestStart);
              nextPoll += [0.125, 0.160, 0.210][pollIndex++ % 3];
            }
            p.animate();
            const cpu = performance.now() - started;
            times.push(cpu);
            totals.maxCpuMs = Math.max(totals.maxCpuMs, cpu);
            totals.frames++;
            const frame = p.frame;
            ensure(frame, 'No display frame');
            if (frame.reset) totals.frameResets++;
            if (lastDisplayTime !== null && frame.time === lastDisplayTime && now > 1 && now < duration - 1) totals.timelineStalls++;
            lastDisplayTime = frame.time;
            if (lastCamera) {
              const movement = p.camera.position.distanceTo(lastCamera);
              totals.maxCameraStep = Math.max(totals.maxCameraStep, movement);
              totals.maxCameraSpeed = Math.max(totals.maxCameraSpeed, movement * fps);
            }
            const stats = p.flightTrail.stats;
            totals.maxVisibleTriangles = Math.max(totals.maxVisibleTriangles, stats.visibleTriangles);
            if (stats.visibleTriangles > frame.path_index * 12) violation('trail exceeds displayed sample', {stats, pathIndex: frame.path_index});
            const attribute = p.trailHead.geometry.attributes.position;
            if (p.trailHead.visible) {
              const endPoint = new p.THREE.Vector3().fromBufferAttribute(attribute, 1);
              const error = endPoint.distanceTo(p.uavGroup.position);
              totals.maxHeadError = Math.max(totals.maxHeadError, error);
              if (error > 0.00002) violation('trail head detached', {error});
            }
            const full = p.flightTrail._chunks[0];
            if (full?.count === p.flightTrail.chunkCapacity && !previousFullChunk) {
              previousFullChunk = full;
              firstFullPositions = full.positions.slice();
              full.geometry.addEventListener('dispose', () => firstFullDisposed++);
            }
            if (previousFullChunk && p.flightTrail._chunks[0] !== previousFullChunk) violation('completed chunk replaced');
            for (const [id, bird] of p.birds) {
              const sample = frame.traffic[id];
              if (!sample || !bird.visible) continue;
              const error = bird.position.distanceTo(world(sample.slice(0, 3)));
              totals.maxTrafficError = Math.max(totals.maxTrafficError, error);
              if (error > 1e-7) violation('bird uses different display clock', {id, error});
            }
            const sight = p.camera.position.clone().sub(p.uavGroup.position);
            const distance = sight.length();
            for (let index = 0; index < p.obstacles.length; index++) {
              const obstacle = p.obstacles[index];
              if (obstacle.containsPoint(p.camera.position)) violation('camera inside building', {obstacle: index});
              ray.set(p.uavGroup.position, sight.clone().normalize());
              if (ray.intersectBox(obstacle, hit) && hit.distanceTo(p.uavGroup.position) < distance - 1e-5) violation('building blocks sightline', {obstacle: index});
              if (lastCamera) {
                const sweep = p.camera.position.clone().sub(lastCamera);
                const travel = sweep.length();
                if (travel > 1e-8) {
                  ray.set(lastCamera, sweep.divideScalar(travel));
                  if (ray.intersectBox(obstacle, hit) && hit.distanceTo(lastCamera) < travel - 1e-5) violation('camera crosses building', {obstacle: index});
                }
              }
            }
            const d = p.followRig.diagnostics;
            if (d.emergencyClamps > lastEmergency || d.sweepClamps > lastSweep) clampEvents.push({time: now,
              displayTime: frame.time, uav: p.uavGroup.position.toArray(), camera: p.camera.position.toArray(),
              emergencyClamps: d.emergencyClamps, sweepClamps: d.sweepClamps});
            lastEmergency = d.emergencyClamps;
            lastSweep = d.sweepClamps;
            if (p.goalStar?.userData.reward.collected && collectedAt === null) collectedAt = frame.time;
            if (collectedAt !== null && collectedAt < targetAt - 1e-7) violation('star collected before displayed arrival');
            lastCamera = p.camera.position.clone();
            step++;
          }
          return { ...totals, diagnostics: p.followRig.diagnostics, failures, clampEvents: clampEvents.slice(0, 30) };
        },
        capture() { realRender(p.scene, p.camera); },
        result() {
          ensure(previousFullChunk, 'Fixture never fills a trail chunk');
          ensure(firstFullDisposed === 0, 'Completed chunk disposed during append');
          ensure(firstFullPositions.every((value, index) => value === previousFullChunk.positions[index]), 'Completed chunk buffer rewritten');
          const percentile = (array, fraction) => [...array].sort((a, b) => a - b)[Math.min(array.length - 1, Math.floor(array.length * fraction))];
          return { ...totals, diagnostics: p.followRig.diagnostics, failures, clampEvents,
            cpu: {median: percentile(times, 0.5), p95: percentile(times, 0.95), p99: percentile(times, 0.99),
              over16ms: times.filter(value => value > 16.7).length, over50ms: times.filter(value => value > 50).length,
              note: 'CPU replay with rasterization disabled; not an interactive browser FPS measurement'},
            packetCpu: {median: percentile(polls, 0.5), p95: percentile(polls, 0.95)},
            trail: p.flightTrail.stats, completedChunkPreserved: true, finalLanded: p.frame.landed,
            finalDisplayTime: p.frame.time, starCollectedAt: collectedAt, targetReachedAt: targetAt };
        },
        lifecycle() {
          const trail = p.flightTrail;
          const chunks = [...trail._chunks];
          let geometryDisposed = 0, materialDisposed = 0;
          chunks.forEach(chunk => chunk.geometry.addEventListener('dispose', () => geometryDisposed++));
          trail.material.addEventListener('dispose', () => materialDisposed++);
          const reset = clone(initial);
          reset.sequence = sequence++;
          reset.diagnostics.run_id = runId + '-reset';
          ensure(p.acceptState(reset), 'Reset rejected');
          p.animate();
          ensure(p.flightTrail !== trail && p.flightTrail.stats.points <= 1, 'Reset retained old history');
          ensure(geometryDisposed === chunks.length && materialDisposed === 1, 'Trail does not own exactly one disposal per resource');
          const htmlTrail = p.flightTrail;
          const airsim = clone(reset);
          airsim.sequence = sequence++;
          airsim.runtime_backend = 'airsim';
          ensure(p.acceptState(airsim), 'AirSim switch rejected');
          const actors = [p.uavGroup, ...p.birds.values()];
          const positions = actors.map(actor => actor.position.toArray());
          const rotations = p.rotors.map(rotor => rotor.rotation.y);
          for (let i = 0; i < 30; i++) { p.clock.now += 1 / 60; p.animate(); }
          ensure(p.flightTrail === null && !p.trailHead, 'HTML trail survived AirSim switch');
          ensure(actors.every((actor, index) => !actor.visible && actor.position.toArray().every((v, j) => v === positions[index][j])), 'HTML actor animated under AirSim');
          ensure(p.rotors.every((rotor, i) => rotor.rotation.y === rotations[i]), 'HTML rotors animated under AirSim');
          ensure(htmlTrail._disposed, 'Old HTML trail was not disposed');
          reset.sequence = sequence++;
          ensure(p.acceptState(reset), 'HTML restore rejected');
          p.animate();
          ensure(p.flightTrail && p.flightTrail.stats.points <= 1 && p.uavGroup.visible, 'HTML restore failed');
          return { reset: true, htmlToAirSimToHtml: true, disposedGeometries: geometryDisposed,
            disposedMaterials: materialDisposed, airSimHtmlActorsFrozen: true };
        },
      };
    }, { initial: fixture.initial, final: fixture.final, samples: fixture.samples });

    const screenshots = [];
    for (const [name, at] of [['takeoff', 5], ['crossing', 28], ['returned', fixture.samples.at(-1).t + 1]]) {
      const partial = await page.evaluate(end => window.followProbe.replay.advance(end), at);
      if (partial.clampEvents.length || partial.failures.length) {
        console.log(JSON.stringify({ stage: name, diagnostics: partial.diagnostics, failures: partial.failures, clampEvents: partial.clampEvents }));
      }
      await page.evaluate(() => window.followProbe.replay.capture());
      const filename = path.join(output, `follow-replay-${name}.png`);
      await page.screenshot({ path: filename });
      screenshots.push(filename);
    }
    const replay = await page.evaluate(() => window.followProbe.replay.result());
    const lifecycle = await page.evaluate(() => window.followProbe.replay.lifecycle());
    let campus = { checked: false, reason: 'Set UAV_CAMPUS_FIXTURE to a key-free offline campus state to include map switching' };
    if (process.env.UAV_CAMPUS_FIXTURE) {
      const campusState = JSON.parse(fs.readFileSync(process.env.UAV_CAMPUS_FIXTURE, 'utf8'));
      assert.equal(campusState.deepseek_key_configured, false);
      campus = await page.evaluate(payload => {
        const p = window.followProbe;
        payload.sequence = p.state.sequence + 1;
        payload.diagnostics.run_id += '-camera-map-check';
        p.acceptState(payload);
        p.setCameraMode('overview');
        p.animate();
        p.replay.capture();
        const group = p.scene.getObjectByName('beihang_hangzhou_campus');
        if (!group) throw new Error('Campus geometry was not built');
        let buildings = 0;
        group.traverse(object => { if (object.name.startsWith('campus_building_')) buildings++; });
        return {checked: true, buildings, scenarioId: p.state.scenario_id};
      }, campusState);
      assert.equal(campus.scenarioId, 'semantic_campus');
      assert.equal(campus.buildings, 50);
      const filename = path.join(output, 'follow-replay-campus-overview.png');
      await page.screenshot({ path: filename });
      screenshots.push(filename);
      campus.restoredUrban = await page.evaluate(payload => {
        const p = window.followProbe;
        const campusGroup = p.scene.getObjectByName('beihang_hangzhou_campus');
        payload.sequence = p.state.sequence + 1;
        payload.diagnostics.run_id += '-restore-urban';
        p.acceptState(payload);
        p.setCameraMode('follow');
        p.animate();
        return campusGroup.parent === null && !p.scene.getObjectByName('beihang_hangzhou_campus')
          && p.state.scenario_id === 'urban_crossing' && Boolean(p.flightTrail);
      }, fixture.initial);
      assert.equal(campus.restoredUrban, true);
    }
    const pageBounds = [];
    for (const width of [1440, 980, 390]) {
      await page.setViewportSize({ width, height: 900 });
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const bounds = await page.evaluate(() => {
        const canvas = window.followProbe.renderer.domElement.getBoundingClientRect();
        return {width: innerWidth, scrollWidth: document.documentElement.scrollWidth,
          canvasWidth: canvas.width, canvasHeight: canvas.height};
      });
      assert.ok(bounds.scrollWidth <= width + 1, `Horizontal page overflow at ${width}px`);
      assert.ok(bounds.canvasWidth > 100 && bounds.canvasHeight > 100, `Collapsed scene at ${width}px`);
      pageBounds.push(bounds);
    }
    const result = { replay, lifecycle, campus, pageBounds, pageErrors: errors, blockedWrites: writes, unknownRequests, requests, screenshots };
    fs.writeFileSync(path.join(output, 'browser-replay.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
    assert.deepEqual(errors, [], 'Browser page errors');
    assert.deepEqual(writes, [], 'Replay attempted an external mutation');
    assert.deepEqual(unknownRequests, [], 'Replay attempted an unexpected resource');
    assert.deepEqual(replay.failures, [], 'Actual Urban route violates visual/safety invariants');
    assert.equal(replay.diagnostics.emergencyClamps, 0, 'Actual route requires emergency camera retraction');
    assert.equal(replay.diagnostics.sweepClamps, 0, 'Actual route requires camera sweep clamp');
    assert.equal(replay.finalLanded, true);
    assert.ok(replay.starCollectedAt >= replay.targetReachedAt);
    assert.ok(replay.timelineStalls === 0, 'Normal packet jitter should not stall displayed flight');
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
