const { chromium } = require('playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ headless: true, channel: 'msedge' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const state = async () => (await page.request.get('http://127.0.0.1:5000/api/state')).json();
    const wait = async predicate => {
      for (let attempt = 0; attempt < 450; attempt++) {
        const current = await state();
        assert.equal(current.flight.error, null);
        if (predicate(current)) return current;
        await page.waitForTimeout(100);
      }
      throw new Error('Mission state timed out');
    };
    await page.goto('http://127.0.0.1:5000');
    assert.equal((await state()).agent_backend, 'mock');
    assert.equal((await state()).runtime_backend, 'html');
    await page.locator('#resetBtn').click();
    await wait(current => current.flight.mode === 'idle');
    await page.request.post('http://127.0.0.1:5000/api/mission/goal', { data: { target: { x: 3, y: 18, z: 2 } } });
    await page.request.post('http://127.0.0.1:5000/api/mission/start');
    const returning = await wait(current => current.phase === 'returning_home');
    assert.equal(returning.status, 'active');
    await page.request.post('http://127.0.0.1:5000/api/runtime/stop');
    await wait(current => !current.flight.running && current.flight.speed_mps < 0.02);
    await page.waitForTimeout(500);
    assert.match(await page.locator('#phaseValue').textContent(), /Returning/);
    const output = path.resolve(__dirname, '../artifacts/return-flight');
    fs.mkdirSync(output, { recursive: true });
    await page.screenshot({ path: path.join(output, 'return-paused.png') });
    await page.locator('#startBtn').click();
    const completed = await wait(current => current.status === 'completed');
    assert.ok(Math.hypot((completed.uav.x - completed.base.x) * 3,
      (completed.uav.y - completed.base.y) * 3, (completed.uav.z - completed.base.z) * 5) < 0.18);
    await page.waitForTimeout(500);
    assert.match(await page.locator('#phaseValue').textContent(), /Completed/);
    await page.screenshot({ path: path.join(output, 'completed.png') });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ phase: completed.phase, plannerCalls: completed.flight.planner_calls,
      position: completed.uav, errors, screenshots: output }));
    await page.request.post('http://127.0.0.1:5000/api/scenario', { data: { scenario_id: 'urban_crossing' } });
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
