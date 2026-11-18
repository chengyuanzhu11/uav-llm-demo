/** Real WebGL RGB/depth, isolated service, measured grounding oracle. */
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const url = process.env.UAV_VISUAL_URL, output = process.env.UAV_VISUAL_OUTPUT;

(async () => {
  const browser = await chromium.launch({headless:true,
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE}:{}),
    args:['--use-gl=angle','--use-angle=swiftshader','--enable-webgl','--enable-unsafe-swiftshader','--no-sandbox']});
  const page = await browser.newPage({viewport:{width:1280,height:900}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/static/app.js',async route=>route.fulfill({response:await page.request.get(url+'/__verify__/app.js')}));
  let final;
  try {
    await page.goto(url);
    await page.waitForFunction(()=>window.visualProbe?.state?.scenario?.mission_mode==='semantic');
    const fixture=await (await page.request.get(url+'/__verify__/fixture')).json();
    const geometry=await page.evaluate(async ({variant,scene})=>{
      const p=window.visualProbe,T=p.THREE;
      const {OnboardCamera}=await import('/static/onboard_camera.js');
      // Independent calibration scene: a plane is 10m from the optical centre.
      const calibration=new T.Scene();calibration.background=new T.Color('#ffffff');
      const plane=new T.Mesh(new T.PlaneGeometry(20,20),new T.MeshBasicMaterial({color:'#ff0000',side:T.DoubleSide}));
      plane.rotation.y=Math.PI/2;plane.position.x=10;plane.userData.environment=true;calibration.add(plane);
      const sensor=new OnboardCamera(T,p.renderer,calibration);
      const packet=sensor.capture({frame_id:'calibration',width:320,height:240,vertical_fov_deg:75,near_m:.1,far_m:250,
        yaw_deg:0,pitch_deg:0,position_grid:[0,0,0]},1);
      const depth=packet.depth_m[120*320+160];
      if(Math.abs(depth-10)>.01)throw new Error('Depth calibration failed: '+depth);
      plane.rotation.set(-Math.PI/2,0,0);plane.position.set(0,0,0);
      const down=sensor.capture({frame_id:'down',width:320,height:240,vertical_fov_deg:75,near_m:.1,far_m:250,
        yaw_deg:135,pitch_deg:-90,position_grid:[0,0,2]},1);
      const downDepth=down.depth_m[120*320+160];
      if(Math.abs(downDepth-10)>.01)throw new Error('Vertical camera calibration failed: '+downDepth);
      if (scene!=='markers') return {depth,downDepth};
      // Markers are ground paint, not collidable obstacles. They exist only in
      // this private visual fixture; neither planner receives their coordinates.
      const centre=variant ? [26.5,79.5] : [22.5,79.5], distractor=variant ? [22.5,79.5] : [26.5,79.5];
      const half=(p.state.grid_size-1)/2;
      function marker(pos,shape,color){
        const geometry=new T.ShapeGeometry(shape);
        const mesh=new T.Mesh(geometry,new T.MeshBasicMaterial({color,side:T.DoubleSide}));
        mesh.rotation.x=-Math.PI/2;mesh.position.set((pos[0]-half)*3,.12,(pos[1]-half)*3);
        mesh.userData.environment=true;p.scene.add(mesh);return mesh;
      }
      const triangle=new T.Shape();triangle.moveTo(-3,-3);triangle.lineTo(3,-3);triangle.lineTo(0,3);triangle.closePath();
      const circle=new T.Shape();circle.absarc(0,0,3,0,Math.PI*2,false);
      marker(centre,triangle,'#ff1010');marker(distractor,circle,'#1050ff');
      if (variant) {
        // An opaque screen hides the target from the initial north-facing view.
        // Its 7.5m top is below the enforced 10m transit floor, including clearance.
        const screen=new T.Mesh(new T.BoxGeometry(10,7.5,.3),new T.MeshStandardMaterial({color:'#858880'}));
        screen.position.set((centre[0]-half)*3,3.75,(81.5-half)*3);
        screen.userData.environment=true;p.scene.add(screen);
      }
      return {centre,distractor,depth,downDepth};
    },fixture);
    const started=Date.now();
    await page.request.post(url+'/__verify__/start');
    let last='',decodedId=null,initialSaved=false,savedFrame=null,lastReport=0;
    const frames={};
    const tracked=new Map();
    let tracking=false;
    const deadline=started+fixture.wall_budget*1000;
    while(Date.now()<deadline){
      const state=await (await page.request.post(url+'/__verify__/advance')).json();
      await page.evaluate(async s=>{window.visualProbe.acceptState(s);await window.visualProbe.captureOnboardObservation();},state);
      const vision=state.onboard, mission=state.semantic_mission;
      const progress=mission.motion_progress;
      if(progress?.phase==='tracking')tracking=true;
      if(tracking){
        const end=mission.evidence.find(e=>e.action==='motion')?.simulation_time ?? Infinity;
        for(const sample of state.motion_samples || []){
          if(sample.t>=state.simulation_time-1.05 && sample.t<=end)tracked.set(sample.t,sample);
        }
        if(Number.isFinite(end))tracking=false;
      }
      const status=[mission.status,mission.current_step_id,vision.calls,vision.feedback?.tool,vision.feedback?.code].join(':');
      if(status!==last || Date.now()-lastReport>20000){
        console.log(JSON.stringify({status,time:state.simulation_time,position:state.observation?.position,feedback:vision.feedback}));
        last=status;lastReport=Date.now();
      }
      if(!fixture.live){
        const payload=await (await page.request.get(url+'/__verify__/decode')).json();
        if(payload && payload.frame.frame_id!==decodedId){
          decodedId=payload.frame.frame_id;
          const pixel=await page.evaluate(async rgb=>{
            const image=new Image();image.src=rgb;await image.decode();
            const canvas=document.createElement('canvas');canvas.width=image.width;canvas.height=image.height;
            const ctx=canvas.getContext('2d');ctx.drawImage(image,0,0);const data=ctx.getImageData(0,0,canvas.width,canvas.height).data;
            let sx=0,sy=0,count=0;
            for(let y=0;y<canvas.height;y++)for(let x=0;x<canvas.width;x++){
              const i=(y*canvas.width+x)*4;
              if(data[i]>150 && data[i]>data[i+1]*1.8 && data[i]>data[i+2]*1.8){sx+=x;sy+=y;count++;}
            }
            return count>12 ? [(sx/count+.5)/canvas.width,(sy/count+.5)/canvas.height]:null;
          },payload.frame.rgb);
          await page.request.post(url+'/__verify__/decode',{data:{pixel}});
        }
      }
      if(vision.frame && vision.frame.frame_id!==savedFrame){
        frames[vision.frame.frame_id]=vision.frame;
        const image=await page.evaluate(()=>document.querySelector('img[alt="当前机载传感器视图"]')?.src);
        if(image?.startsWith('data:')){
          fs.mkdirSync(path.dirname(output),{recursive:true});
          const data=Buffer.from(image.split(',')[1],'base64');
          if(!initialSaved){fs.writeFileSync(output+'.initial.png',data);initialSaved=true;}
          fs.writeFileSync(output+'.latest.png',data);savedFrame=vision.frame.frame_id;
        }
      }
      if(['completed','safety_wait','generation_failed','needs_clarification','failed','error'].includes(mission.status)){
        final=state;break;
      }
      await new Promise(resolve=>setTimeout(resolve,vision.status==='reasoning'?50:5));
    }
    if(!final)final=await (await page.request.get(url+'/api/state')).json();
    const evidence=final.semantic_mission.evidence.find(e=>e.action==='visual');
    const views=evidence?.result.views || [];
    fs.mkdirSync(path.dirname(output),{recursive:true});
    // Preserve completion and frame provenance even if independent scoring fails.
    fs.writeFileSync(output,JSON.stringify({live:fixture.live,scene:fixture.scene,oracle:null,
      semantic:final.semantic_mission,vision:final.onboard,frames},null,2));
    const distances=geometry.centre ? views.map(v=>Math.hypot((v.point_grid[0]-geometry.centre[0])*3,(v.point_grid[1]-geometry.centre[1])*3)) : [];
    const inside=fixture.scene==='markers' ? views.map(v=>{
      const x=(v.point_grid[0]-geometry.centre[0])*3,y=(v.point_grid[1]-geometry.centre[1])*3;
      // The paint is at 0.12m, above the 0.10m field. Depth calibration supports
      // a 1cm surface tolerance, so a nearby background pixel cannot score true.
      return y>=-3.15 && y<=3.15 && Math.abs(x)<=(y+3)/2+.15 && Math.abs(v.point_grid[2]*5-.12)<.01;
    }) : await page.evaluate(({views,frames,kind,target})=>{
      const p=window.visualProbe,T=p.THREE,half=(p.state.grid_size-1)/2;
      p.scene.updateMatrixWorld(true);
      const surfaces=[];
      for(const root of p.scene.children.filter(o=>o.userData.environment)) {
        root.traverseVisible(o=>{if(o.isMesh)surfaces.push(o);});
      }
      return views.map(view=>{
        const f=frames[view.frame_id]; if(!f)return false;
        const origin=new T.Vector3((f.position_grid[0]-half)*3,f.position_grid[2]*5,(f.position_grid[1]-half)*3);
        const measured=new T.Vector3((view.point_grid[0]-half)*3,view.point_grid[2]*5,(view.point_grid[1]-half)*3);
        const ray=new T.Raycaster(origin,measured.clone().sub(origin).normalize());
        // Filter before raycasting: sprites require a display camera and are
        // intentionally absent from the physical sensor's RGB/depth pass.
        const hits=ray.intersectObjects(surfaces,false);
        const hit=hits[0]; if(!hit || hit.point.distanceTo(measured)>.03)return false;
        if(kind==='tree')return hit.object.name.startsWith('campus_tree_');
        let object=hit.object;
        while(object && !object.userData.landmarkId)object=object.parent;
        return object?.userData.landmarkId===(target || (kind==='bin_orbit'?'F14':'F01'));
      });
    },{views,frames,kind:fixture.scene,target:fixture.target_landmark});
    const orbit=fixture.scene==='bin_orbit' ? await page.evaluate(samples=>{
      const p=window.visualProbe,T=p.THREE,layout=p.state.scenario.campus_layout;
      const target=layout.landmarks.find(o=>o.id==='F14').position;
      const region=layout.spatial_objects.west_garden.center_grid;
      const points=samples.map(s=>[(s.position[0]-target[0])*3,(s.position[1]-target[1])*3]);
      const winding=center=>{
        const angles=samples.map(s=>Math.atan2(s.position[1]-center[1],s.position[0]-center[0]));
        return angles.slice(1).reduce((sum,a,i)=>sum+Math.atan2(Math.sin(a-angles[i]),Math.cos(a-angles[i])),0);
      };
      let xx=0,xy=0,x=0,yy=0,y=0,bx=0,by=0,b=0;
      for(const [px,py] of points){const q=px*px+py*py;xx+=px*px;xy+=px*py;x+=px;yy+=py*py;y+=py;bx+=px*q;by+=py*q;b+=q;}
      const matrix=new T.Matrix3().set(xx,xy,x,xy,yy,y,x,y,points.length);
      const fit=Math.abs(matrix.determinant())>1e-6 ? new T.Vector3(bx,by,b).applyMatrix3(matrix.invert()):null;
      const targetWinding=winding(target),centerError=fit?Math.hypot(fit.x/2,fit.y/2):Infinity;
      return {samples:points.length,target_winding_rad:targetWinding,region_winding_rad:winding(region),
        fitted_center_error_m:centerError,passed:points.length>30 && Math.abs(targetWinding)>5.8
          && Math.abs(targetWinding)<6.7 && centerError<2};
    },[...tracked.values()].sort((a,b)=>a.t-b.t)) : null;
    const motion=final.semantic_mission.evidence.find(e=>e.action==='motion');
    const oracle={correct_target:views.length===2 && inside.every(Boolean),inside_target:inside,target_errors_m:distances,
      ...(orbit?{orbit,correct_motion_binding:motion?.result.target_binding?.evidence_id===evidence?.evidence_id}:{}),
      completed:final.semantic_mission.status==='completed',returned:final.semantic_mission.evidence.some(e=>e.action==='return_home'&&e.result.landed),
      safe_success:final.milestones?.safe_success,calibration_depth_m:geometry.depth,
      vertical_calibration_depth_m:geometry.downDepth,browser_errors:errors};
    fs.mkdirSync(path.dirname(output),{recursive:true});
    fs.writeFileSync(output,JSON.stringify({live:fixture.live,variant:fixture.variant,scene:fixture.scene,wall_seconds:(Date.now()-started)/1000,oracle,semantic:final.semantic_mission,
      vision:final.onboard,diagnostics:final.diagnostics,tracking_samples:[...tracked.values()]},null,2));
    console.log(JSON.stringify(oracle));
    assert.equal(errors.length,0);assert.ok(oracle.correct_target && oracle.completed && oracle.returned && oracle.safe_success,'Independent visual task oracle failed');
    if(orbit)assert.ok(orbit.passed && oracle.correct_motion_binding,'Trajectory did not circle the visually observed object');
  }finally{await browser.close();}
})().catch(e=>{console.error(e.stack);process.exitCode=1;});
