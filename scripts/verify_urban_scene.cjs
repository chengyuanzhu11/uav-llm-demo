const { chromium } = require("playwright");
const sharp = require("sharp");
const path = require("node:path");


async function main() {
  const baseUrl = process.env.UAV_DEMO_URL || "http://127.0.0.1:5000";
  const browser = await chromium.launch({
    channel: "msedge",
    headless: true,
    args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-webgl"],
  });
  try {
    for (const [width, height] of [[1440, 900], [980, 720], [390, 844]]) {
      const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
      const errors = [];
      page.on("pageerror", (error) => errors.push(error.message));
      await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
      await page.locator("#sceneContainer canvas").waitFor({ timeout: 15000 });
      await page.waitForTimeout(2500);
      const rect = await page.locator("#sceneContainer").boundingBox();
      const inspector = await page.locator("#decisionInspector").boundingBox();
      const documentWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      if (documentWidth > width + 1 ||
          (width <= 700 && inspector.y < rect.y + rect.height - 1) ||
          (width > 700 && inspector.x < rect.x + rect.width - 1)) {
        throw new Error(`${width}x${height}: scene and inspector overlap or page overflows`);
      }
      const output = path.join(__dirname, "..", "artifacts", `urban-${width}.png`);
      const png = await page.screenshot({ path: output });
      const image = sharp(png);
      const left = Math.max(0, Math.round(rect.x + rect.width * 0.16));
      const top = Math.max(0, Math.round(rect.y + rect.height * 0.2));
      const cropWidth = Math.max(1, Math.round(rect.width * 0.68));
      const cropHeight = Math.max(1, Math.round(rect.height * 0.6));
      const { data, info } = await image.extract({ left, top, width: cropWidth, height: cropHeight })
        .raw().toBuffer({ resolveWithObject: true });
      const colors = new Set();
      for (let i = 0; i < data.length; i += info.channels * 19) {
        colors.add(`${data[i] >> 4}:${data[i + 1] >> 4}:${data[i + 2] >> 4}`);
      }
      if (colors.size < 25 || errors.length) {
        throw new Error(`${width}x${height}: ${colors.size} color bins; page errors: ${errors.join(" | ")}`);
      }
      const startX = rect.x + rect.width * 0.72;
      const startY = rect.y + rect.height * 0.68;
      await page.mouse.move(startX, startY);
      await page.mouse.down();
      await page.mouse.move(startX - 45, startY - 18, { steps: 8 });
      await page.mouse.up();
      await page.waitForTimeout(300);
      const { data: before, info: screenshotInfo } = await sharp(png).raw().toBuffer({ resolveWithObject: true });
      const after = await sharp(await page.screenshot()).raw().toBuffer();
      let changed = 0;
      for (let i = 0; i < before.length; i += screenshotInfo.channels * 13) {
        if (Math.abs(before[i] - after[i]) + Math.abs(before[i + 1] - after[i + 1]) +
            Math.abs(before[i + 2] - after[i + 2]) > 45) changed += 1;
      }
      if (changed < 250) throw new Error(`${width}x${height}: scene did not respond to orbit drag`);
      if (width === 1440) {
        const initial = await page.request.get(`${baseUrl}/api/state`).then((response) => response.json());
        if (initial.agent_backend !== "mock") {
          throw new Error("Visual flight check requires Mock agent; refusing to call a paid API");
        }
        const roofClearance = await page.evaluate(async () => {
          const THREE = await import("three");
          const { buildUrbanCity } = await import("/static/urban_city.js");
          const state = await fetch("/api/state").then((response) => response.json());
          const half = (state.grid_size - 1) / 2;
          const toWorld = (point) => new THREE.Vector3((point.x - half) * 3, point.z * 5,
            (point.y - half) * 3);
          const roads = state.scene_layout.filter(item => item.name.startsWith("road_"));
          const city = buildUrbanCity(state.scenario, toWorld, 3, 5, roads);
          let minimum = Infinity;
          for (let i = 0; i < city.children.length - 1; i += 1) {
            if (!city.children[i].userData.cameraOccluder) continue;
            const mass = new THREE.Box3().setFromObject(city.children[i]);
            const cap = new THREE.Box3().setFromObject(city.children[i + 1]);
            minimum = Math.min(minimum, cap.min.y - mass.max.y);
          }
          city.traverse((object) => {
            object.geometry?.dispose();
            const materials = Array.isArray(object.material) ? object.material : [object.material];
            materials.forEach((material) => { material?.map?.dispose(); material?.dispose(); });
          });
          return minimum;
        });
        if (roofClearance < 0.005) throw new Error(`Coplanar roof cap: ${roofClearance}`);
        await page.locator("#planBtn").click();
        await page.waitForFunction(async () => {
          const state = await fetch("/api/state").then((response) => response.json());
          return state.flight.plan_id > 0 || state.flight.error;
        }, null, { timeout: 15000 });
        const planned = await page.request.get(`${baseUrl}/api/state`).then((response) => response.json());
        if (planned.flight.error) throw new Error(`Mock planning failed: ${planned.flight.error}`);
        await page.locator("#followBtn").click();
        await page.locator("#startBtn").click();
        await page.waitForFunction(async () => {
          const state = await fetch("/api/state").then((response) => response.json());
          return Math.hypot(state.uav.x - 1, state.uav.y - 18, state.uav.z - 2) > 0.005;
        }, null, { timeout: 15000 });
        await page.screenshot({ path: path.join(__dirname, "..", "artifacts", "urban-follow.png") });
        await page.request.post(`${baseUrl}/api/reset`);
        const probeState = await page.request.get(`${baseUrl}/api/state`).then((response) => response.json());
        probeState.uav = { ...probeState.uav, x: 16, y: 3.33, z: 2 };
        await page.route("**/api/state", async (route) => {
          await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(probeState) });
        });
        await page.reload({ waitUntil: "domcontentloaded" });
        await page.locator("#sceneContainer canvas").waitFor();
        await page.locator("#followBtn").click();
        await page.waitForTimeout(500);
        await page.screenshot({ path: path.join(__dirname, "..", "artifacts", "urban-follow-near-building.png") });
        for (const y of [4.2, 5.2, 6.2, 7.2]) {
          probeState.uav.y = y;
          probeState.sequence += 1;
          await page.waitForTimeout(220);
        }
        await page.waitForTimeout(600);
        await page.screenshot({ path: path.join(__dirname, "..", "artifacts", "urban-follow-moving.png") });
        await page.unroute("**/api/state");
        if (errors.length) throw new Error(`Follow view page errors: ${errors.join(" | ")}`);
        console.log(`roof cap clearance ${roofClearance.toFixed(3)} m; follow view screenshot captured`);
      }
      console.log(`${width}x${height}: canvas ${Math.round(rect.width)}x${Math.round(rect.height)}, ${colors.size} color bins, ${changed} changed samples, no overlap/page errors; ${output}`);
      await page.close();
    }
  } finally {
    await browser.close();
  }
}


main().catch((error) => { console.error(error); process.exitCode = 1; });
