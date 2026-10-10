const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const output = path.resolve(__dirname, '../artifacts/low-flight');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const waitState = async predicate => {
      const deadline = Date.now() + 15000;
      while (Date.now() < deadline) {
        const current = await page.request.get('http://127.0.0.1:5000/api/state').then(response => response.json());
        if (predicate(current)) return current;
        await page.waitForTimeout(100);
      }
      throw new Error('Timed out waiting for mission state');
    };
    const waitMode = async mode => {
      for (let attempt = 0; attempt < 150; attempt++) {
        if (await page.locator('#telemetryPhaseValue').textContent() === mode) return;
        await page.waitForTimeout(100);
      }
      throw new Error(`UI did not reach ${mode}`);
    };
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.route('**/static/app.js', async route => {
      const response = await route.fetch();
      await route.fulfill({ response, body: await response.text() + '\nwindow.__flightQA = {scene,camera,controls,birdMeshes,renderer,THREE};' });
    });
    await page.goto('http://127.0.0.1:5000');
    await page.waitForFunction(() => window.__flightQA?.birdMeshes.size > 0);
    const initial = await page.request.get('http://127.0.0.1:5000/api/state').then(response => response.json());
    assert.equal(initial.agent_backend, 'mock', 'Only exercise offline Mock');
    await page.locator('#resetBtn').click();
    await waitMode('idle');
    for (const [width, height] of [[1440, 900], [1280, 720], [980, 720]]) {
      await page.setViewportSize({ width, height });
      await page.waitForTimeout(400);
      const bounds = await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        const gl = canvas.getContext('webgl2');
        const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
        gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const colors = new Set();
        for (let index = 0; index < pixels.length; index += 400) colors.add(`${pixels[index]},${pixels[index+1]},${pixels[index+2]}`);
        const controls = [...document.querySelectorAll('.toolbar button, .toolbar select')].filter(element => element.getClientRects().length);
        return { overflow: document.documentElement.scrollWidth > innerWidth, colors: colors.size,
          clipped: controls.filter(element => { const bounds = element.getBoundingClientRect(); return bounds.right > innerWidth || bounds.left < 0; }).length };
      });
      assert.equal(bounds.overflow, false);
      assert.equal(bounds.clipped, 0);
      assert.ok(bounds.colors > 20, '3D canvas must contain varied rendered pixels');
      await page.screenshot({ path: path.join(output, `overview-${width}.png`) });
    }
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.locator('#planBtn').click();
    await waitState(current => current.flight.plan_id >= 1);
    await waitMode('paused');
    await page.locator('#startBtn').click();
    await waitState(current => current.flight.running && current.flight.speed_mps > 0.2);
    await page.waitForTimeout(1500);
    await page.locator('#startBtn').click();
    await waitState(current => !current.flight.running && current.flight.speed_mps < 0.02);
    await waitMode('paused');
    const before = await page.evaluate(() => [...window.__flightQA.birdMeshes.values()].map(bird => bird.position.toArray()));
    await page.waitForTimeout(1100);
    const after = await page.evaluate(() => [...window.__flightQA.birdMeshes.values()].map(bird => bird.position.toArray()));
    assert.notDeepEqual(before, after, 'Birds continue during UAV pause');
    const paused = await page.request.get('http://127.0.0.1:5000/api/state').then(response => response.json());
    assert.ok(paused.uav.z >= 1.99 && paused.uav.z <= 2.41);
    assert.equal(paused.flight.running, false);
    await page.evaluate(() => {
      const qa = window.__flightQA;
      const bird = [...qa.birdMeshes.values()][0];
      qa.camera.position.copy(bird.position).add(new qa.THREE.Vector3(7, 5, 8));
      qa.controls.minDistance = 2;
      qa.controls.target.copy(bird.position);
      qa.controls.update();
    });
    await page.waitForTimeout(200);
    const pointer = await page.evaluate(() => {
      const qa = window.__flightQA;
      const bird = [...qa.birdMeshes.values()][0];
      const point = bird.position.clone().project(qa.camera);
      const bounds = qa.renderer.domElement.getBoundingClientRect();
      return { x: bounds.left + (point.x + 1) * bounds.width / 2, y: bounds.top + (1 - point.y) * bounds.height / 2 };
    });
    await page.mouse.move(pointer.x, pointer.y);
    await page.screenshot({ path: path.join(output, 'bird-detail.png') });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ screenshots: output, consoleErrors: errors, pausedHeightM: paused.uav.z * 5,
      plannerCalls: paused.flight.planner_calls, movingWhilePaused: true }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
