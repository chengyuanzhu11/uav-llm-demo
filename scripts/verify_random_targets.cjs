const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');

(async () => {
  const base = process.env.UAV_DEMO_URL;
  assert.ok(base, 'Use verify_random_targets.py to create an isolated test server');
  const browser = await chromium.launch({ headless: true, channel: 'msedge',
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-webgl'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
    await page.route('**/static/app.js', async route => {
      const response = await route.fetch();
      await route.fulfill({ response, body: await response.text()
        + '\nwindow.__targetQA = {scene,renderer,toWorld,getState:()=>state};' });
    });
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.__targetQA?.getState()?.target);
    const targets = [];
    for (let attempt = 0; attempt < 3; attempt++) {
      if (attempt) await page.locator('#resetBtn').click();
      await page.waitForFunction(previous => {
        const qa = window.__targetQA;
        const state = qa?.getState();
        if (!state || JSON.stringify(state.target) === previous) return false;
        const pads = qa.scene.children.filter(object => object.userData.environment
          && object.material?.color?.getHex() === 0xf2b84b);
        const pad = state.scene_layout.find(item => item.name === 'target_pad');
        return pads.length === 1 && pads[0].position.distanceTo(qa.toWorld(pad.position)) < 1e-6;
      }, attempt ? JSON.stringify(targets.at(-1)) : null);
      const target = await page.evaluate(() => window.__targetQA.getState().target);
      targets.push(target);
      const current = await (await page.request.get(base + '/api/state')).json();
      assert.equal(current.agent_backend, 'mock');
      assert.equal(current.flight.planner_calls, 0);
      assert.deepEqual(current.target, target);
    }
    await page.locator('#planBtn').click();
    await page.waitForFunction(() => window.__targetQA.getState().flight.plan_id === 1);
    const planned = await page.evaluate(() => window.__targetQA.getState());
    assert.deepEqual(planned.target, targets.at(-1));
    assert.deepEqual(planned.flight.waypoints.at(-1), planned.target);
    for (const [width, height] of [[1440, 900], [1280, 720], [980, 720], [390, 844]]) {
      await page.setViewportSize({ width, height });
      await page.waitForTimeout(350);
      const canvas = await page.locator('#sceneContainer canvas').screenshot();
      const sharp = require('sharp');
      const { data, info } = await sharp(canvas).raw().toBuffer({ resolveWithObject: true });
      const colors = new Set();
      for (let index = 0; index < data.length; index += info.channels * 23) {
        colors.add(`${data[index] >> 4}:${data[index + 1] >> 4}:${data[index + 2] >> 4}`);
      }
      assert.ok(colors.size > 25, '3D canvas must not be blank');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      await page.screenshot({ path: path.resolve(__dirname, `../artifacts/random-target-${width}.png`) });
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ targets, plannerCalls: planned.flight.planner_calls,
      errors, viewports: [1440, 1280, 980, 390] }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
