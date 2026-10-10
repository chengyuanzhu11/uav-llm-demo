const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const base = process.env.UAV_DEMO_URL;
  assert.match(base || '', /^http:\/\/127\.0\.0\.1:\d+$/);
  assert.ok(!base.endsWith(':5000'), 'Do not modify the live user session');
  const output = path.resolve(__dirname, '../artifacts/avoidance');
  fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.route('**/static/app.js', async route => {
      const response = await route.fetch();
      await route.fulfill({ response, body: await response.text()
        + '\nwindow.__avoidanceQA = () => ({route:avoidanceRoute,renderer,scene,camera,state});' });
    });
    await page.goto(base);
    await page.waitForFunction(() => Boolean(window.__avoidanceQA?.().state));
    let state = await (await page.request.post(base + '/__test__/advance')).json();
    assert.equal(state.agent_backend, 'mock');
    assert.equal(state.flight.avoidance.active, true);
    await page.waitForFunction(() => window.__avoidanceQA().route?.visible);
    assert.match(await page.locator('#guardStatus').textContent(), /LOCAL AVOIDANCE/);
    assert.match(await page.locator('#plannerProgress').textContent(), /local replans/);
    for (const [width, height] of [[1440, 900], [1280, 720], [980, 720]]) {
      await page.setViewportSize({ width, height });
      await page.waitForTimeout(300);
      const bounds = await page.evaluate(() => {
        const qa = window.__avoidanceQA();
        qa.renderer.render(qa.scene, qa.camera);
        const gl = qa.renderer.getContext();
        const pixels = new Uint8Array(gl.drawingBufferWidth * gl.drawingBufferHeight * 4);
        gl.readPixels(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        const colors = new Set();
        for (let i = 0; i < pixels.length; i += 400) colors.add(`${pixels[i]},${pixels[i + 1]},${pixels[i + 2]}`);
        const canvas = qa.renderer.domElement.getBoundingClientRect();
        const inspector = document.querySelector('#decisionInspector').getBoundingClientRect();
        return { colors: colors.size, overflow: document.documentElement.scrollWidth > innerWidth,
          overlap: Math.min(canvas.right, inspector.right) > Math.max(canvas.left, inspector.left)
            && Math.min(canvas.bottom, inspector.bottom) > Math.max(canvas.top, inspector.top) };
      });
      assert.ok(bounds.colors > 20, 'Canvas must contain a rendered scene');
      assert.equal(bounds.overflow, false);
      assert.equal(bounds.overlap, false);
      await page.screenshot({ path: path.join(output, `avoidance-${width}.png`) });
    }
    const before = state.uav;
    for (let i = 0; i < 13; i++) state = await (await page.request.post(base + '/__test__/advance')).json();
    assert.notDeepEqual(state.uav, before);
    assert.equal(state.flight.avoidance.active, false);
    assert.equal(state.flight.planner_calls, 0);
    assert.equal(state.flight.running, true);
    await page.waitForFunction(() => !window.__avoidanceQA().route.visible);
    const failed = await (await page.request.post(base + '/__test__/provider_failure')).json();
    assert.equal(failed.flight.planner_calls, 1);
    assert.match(failed.flight.error, /HTTP 401/);
    assert.equal(failed.history.at(-1).decision.provider_error, true);
    assert.ok(!JSON.stringify(failed).includes('dummy-test-credential'));
    await page.waitForFunction(() => document.querySelector('#guardStatus')?.textContent === 'API ERROR');
    assert.match(await page.locator('#feedbackText').textContent(), /HTTP 401/);
    await page.screenshot({ path: path.join(output, 'provider-error-1440.png') });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ viewports: [1440, 1280, 980], detour: true, rejoined: true,
      plannerCalls: state.flight.planner_calls, providerFailureShown: true, consoleErrors: errors, output }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
