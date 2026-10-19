/** Read-only browser acceptance against verify_traffic_upgrade.py's local fixture. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');
const base = process.env.UAV_TRAFFIC_TEST_URL;
assert.match(base || '', /^http:\/\/127\.0\.0\.1:\d+$/);
assert.notEqual(new URL(base).port, '5000', 'Never use the live user service');
const output = path.resolve(process.env.UAV_CAPTURE_DIR || path.join(root, 'outputs/traffic-upgrade'));
fs.mkdirSync(output, { recursive: true });
const source = fs.readFileSync(path.join(root, 'uav_llm_demo/static/app.js'), 'utf8');
assert.ok(source.includes('renderer.setAnimationLoop(animate);'));
const probe = `
const qaClock = {now:0, delta:1/30};
const qaIngest = motionTimeline.ingest.bind(motionTimeline);
const qaSample = motionTimeline.sample.bind(motionTimeline);
motionTimeline.ingest = samples => qaIngest(samples, qaClock.now);
motionTimeline.sample = () => qaSample(qaClock.now);
timer.update = () => {};
timer.getDelta = () => qaClock.delta;
timer.getElapsed = () => qaClock.now;
window.trafficQA = {THREE, renderer, scene, camera, controls, acceptState, animate,
  setCameraMode, motionTimeline, clock:qaClock, updateBirdHover,
  get state(){return state;}, get birds(){return birdMeshes;}, get uav(){return uavGroup;},
  get mode(){return cameraMode;}, get pointer(){return birdPointer;},
  get tooltip(){return birdTooltip;}, stopPolling(){pollActive=false;}};
`;

(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {}),
    args:['--use-gl=angle','--use-angle=swiftshader','--enable-webgl','--enable-unsafe-swiftshader','--no-sandbox']});
  const errors = [], consoleErrors = [], denied = [];
  try {
    const page = await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    await page.route('**/*', async route => {
      const request = route.request(), url = new URL(request.url());
      if (url.origin !== base || request.method() !== 'GET') {
        denied.push({origin:url.origin,path:url.pathname,method:request.method()});
        return route.abort();
      }
      if (url.pathname === '/static/app.js') {
        return route.fulfill({contentType:'text/javascript',body:
          source.replace('renderer.setAnimationLoop(animate);','renderer.setAnimationLoop(null);')+probe});
      }
      if (url.pathname === '/' && process.env.UAV_TEST_FONT) {
        const response = await route.fetch();
        const html = (await response.text()).replace('</head>',
          '<style>@font-face{font-family:TrafficChinese;src:url(/__font.ttf)}body,button,input,select,textarea{font-family:TrafficChinese,sans-serif!important}</style></head>');
        return route.fulfill({response,body:html});
      }
      return route.continue();
    });
    await page.goto(base, {waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => window.trafficQA?.state && window.trafficQA.uav, null, {timeout:30000});
    await page.evaluate(async () => {
      const p=window.trafficQA;
      p.stopPolling();
      const fixture=await (await fetch('/__fixture')).json();
      if (fixture.provider_calls !== 0) throw new Error('Fixture contacted a provider');
      window.trafficReplay={states:fixture.states,index:0,frames:0,renderCalls:0,positions:[],cameras:[],packets:0};
      const replay=window.trafficReplay;
      const originalRender=p.renderer.render.bind(p.renderer);
      p.renderer.render=(...args)=>{replay.renderCalls++;return originalRender(...args);};
      p.setCameraMode('follow');
      p.clock.now=3.8;
    });
    for (let batch=0; batch<6; batch++) {
      await page.evaluate(async () => {
        const p=window.trafficQA,r=window.trafficReplay;
        for(let frame=0;frame<30;frame++) {
          p.clock.now += 1/30;
          while(r.index+1<r.states.length && r.states[r.index+1].simulation_time<=p.clock.now+1e-9) {
            p.acceptState(structuredClone(r.states[++r.index])); r.packets++;
          }
          p.animate(); r.frames++;
          r.positions.push(p.uav.position.toArray()); r.cameras.push(p.camera.position.toArray());
          if(![...p.uav.position.toArray(),...p.camera.position.toArray()].every(Number.isFinite)) throw new Error('Non-finite follow pose');
          await new Promise(resolve=>requestAnimationFrame(resolve));
        }
      });
      if(batch===0) await page.screenshot({path:path.join(output,'follow-start.png')});
    }
    const result=await page.evaluate(() => {
      const p=window.trafficQA,r=window.trafficReplay;
      const check=(condition,message)=>{if(!condition)throw new Error(message);};
      check(p.mode==='follow','Follow mode is not active');
      check(r.frames===180 && r.renderCalls>=180,'Every replay frame must reach real WebGL rendering');
      const movement=Math.hypot(...r.positions.at(-1).map((v,i)=>v-r.positions[0][i]));
      const cameraMovement=Math.hypot(...r.cameras.at(-1).map((v,i)=>v-r.cameras[0][i]));
      check(movement>3 && cameraMovement>3,'The UAV and following camera must both move');
      const envelopes=p.state.dynamic_objects.map(item=>{
        const bird=p.birds.get(item.id);check(Boolean(bird),'Missing bird');
        const d=bird.userData;
        check(Math.abs(d.contactBubble.scale.x-item.contact_distance_m)<1e-8,'Contact radius differs from API');
        check(Math.abs(d.safetyBubble.scale.x-item.separation_distance_m)<1e-8,'Safety radius differs from API');
        check(Math.abs(d.anatomy.scale.x*1.65-item.radius_m)<1e-8,'Anatomy radius is not metric');
        return {id:item.id,radius_m:item.radius_m,contact_distance_m:d.contactBubble.scale.x,
          separation_distance_m:d.safetyBubble.scale.x};
      });
      const gl=p.renderer.getContext();p.renderer.render(p.scene,p.camera);
      const pixels=new Uint8Array(gl.drawingBufferWidth*gl.drawingBufferHeight*4);
      gl.readPixels(0,0,gl.drawingBufferWidth,gl.drawingBufferHeight,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
      const colors=new Set();for(let i=0;i<pixels.length;i+=400)colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
      check(colors.size>30,'WebGL frame is blank');check(gl.getError()===gl.NO_ERROR,'WebGL reported an error');
      check(!p.state.deepseek_key_configured && p.state.diagnostics.usage.requests===0,'Provider calls/key must remain absent');
      const maxStep=values=>Math.max(...values.slice(1).map((v,i)=>Math.hypot(...v.map((x,a)=>x-values[i][a]))));
      return {ok:true,scenario:p.state.scenario_id,agent:p.state.agent_backend,frames:r.frames,
        render_calls:r.renderCalls,packets:r.packets,display_seconds:6,movement_m:movement,
        camera_movement_m:cameraMovement,max_uav_step_m:maxStep(r.positions),max_camera_step_m:maxStep(r.cameras),
        pixel_colors:colors.size,webgl_version:gl.getParameter(gl.VERSION),
        gl_renderer:gl.getParameter(gl.RENDERER),envelopes,provider_calls:0,key_configured:false,
        note:'Deterministic 30 Hz replay rendered through actual WebGL; this is not a desktop FPS benchmark.'};
    });
    await page.screenshot({path:path.join(output,'follow-end.png')});
    result.envelope_closeup=await page.evaluate(() => {
      const p=window.trafficQA;
      const birds=[...p.birds.values()].filter(item=>item.visible);
      if(!birds.length)throw new Error('No bird available for envelope close-up');
      p.setCameraMode('overview');
      // Fixed traffic can put the old close-up camera inside a building. Find
      // a genuinely visible view without hiding geometry or bypassing hover.
      const offsets=[[7,4,8],[0,2,4],[0,2,-4],[4,2,0],[-4,2,0],[0,6,0]];
      const attempts=[];
      let bird=null;
      for(const candidate of birds) {
        for(const offset of offsets) {
          p.camera.position.copy(candidate.position).add(new p.THREE.Vector3(...offset));
          p.camera.lookAt(candidate.position);p.camera.updateMatrixWorld(true);
          p.pointer.set(0,0);p.scene.updateMatrixWorld(true);p.updateBirdHover(true);
          const visible=!p.tooltip.hidden && p.tooltip.textContent.startsWith(`${candidate.userData.id} ·`);
          attempts.push({bird:candidate.userData.id,offset,tooltip_visible:visible});
          if(visible){bird=candidate;break;}
        }
        if(bird)break;
      }
      if(!bird)throw new Error('Actual raycast did not expose bird tooltip from any close-up view');
      bird.userData.safetyBubble.visible=true;bird.userData.contactBubble.visible=true;
      p.renderer.render(p.scene,p.camera);
      if(!p.tooltip.textContent.includes('内圈接触') || !p.tooltip.textContent.includes('外圈安全'))throw new Error('Envelope legend missing');
      return {bird:bird.userData.id,tooltip:p.tooltip.textContent,view_attempts:attempts,
        contact_radius_m:bird.userData.contactBubble.scale.x,safety_radius_m:bird.userData.safetyBubble.scale.x};
    });
    await page.screenshot({path:path.join(output,'envelope-closeup.png')});
    assert.deepEqual(errors,[]);assert.deepEqual(consoleErrors,[]);assert.deepEqual(denied,[]);
    Object.assign(result,{page_errors:errors,console_errors:consoleErrors,denied_requests:denied,
      isolated_port:Number(new URL(base).port),screenshots:['follow-start.png','follow-end.png','envelope-closeup.png']});
    fs.writeFileSync(path.join(output,'browser-check.json'),JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify(result));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});
