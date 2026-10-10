const { chromium } = require("playwright");
const fs = require("node:fs");
const path = require("node:path");
const assert = require("node:assert/strict");

(async () => {
  const browser = await chromium.launch({ channel: "msedge", headless: true,
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"] });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on("pageerror", e => errors.push(e.message));
    const source = fs.readFileSync(path.join(__dirname, "../uav_llm_demo/static/app.js"), "utf8");
    await page.route("**/static/app.js*", route => route.fulfill({ contentType: "text/javascript",
      body: source + `\nwindow.cameraProbe = { run() {
        const saved = cameraObstacles;
        cameraObstacles = [];
        setCameraMode('follow');
        const initial = camera.position.clone().sub(uavGroup.position);
        let drift = 0;
        for (let i=0;i<300;i++) {
          uavGroup.position.x += i % 6 === 0 ? 0.08 : 0;
          updateFollowCamera(1/60);
          drift = Math.max(drift, camera.position.clone().sub(uavGroup.position).distanceTo(initial));
        }
        const p = uavGroup.position;
        cameraObstacles = [new THREE.Box3(p.clone().add(new THREE.Vector3(-100,1,-100)),
          p.clone().add(new THREE.Vector3(100,2,100)))];
        followOffset.set(9,7,12);
        const short = chooseFollowOffset(p).clone();
        let changes = 0;
        for(let i=0;i<300;i++) if(chooseFollowOffset(p).distanceTo(short)>1e-8) changes++;
        cameraObstacles = saved;
        uavGroup.position.copy(toWorld(state.uav));
        setCameraMode('follow');
        return {drift, changes};
      }};` }));
    // Read-only state; no mission commands or paid provider requests.
    await page.goto("http://127.0.0.1:5000", { waitUntil: "domcontentloaded" });
    await page.waitForFunction(() => window.cameraProbe && !document.querySelector('#planBtn').disabled);
    const result = await page.evaluate(() => window.cameraProbe.run());
    assert.ok(result.drift < 1e-8, JSON.stringify(result));
    assert.equal(result.changes, 0);
    for (const width of [1440, 980]) {
      await page.setViewportSize({ width, height: 900 });
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(__dirname, `../artifacts/follow-stable-${width}.png`) });
    }
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ ...result, pageErrors: errors }));
  } finally { await browser.close(); }
})().catch(e => { console.error(e); process.exitCode = 1; });
