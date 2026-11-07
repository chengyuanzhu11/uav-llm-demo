// A calibrated sensor camera, never driven by OrbitControls or FollowCameraRig.
export class OnboardCamera {
  constructor(THREE, renderer, scene) {
    this.THREE = THREE; this.renderer = renderer; this.scene = scene;
    this.camera = new THREE.PerspectiveCamera();
    this.target = new THREE.WebGLRenderTarget(320, 240, {depthBuffer: true});
    this.target.texture.colorSpace = THREE.SRGBColorSpace;
    this.target.depthTexture = new THREE.DepthTexture(320, 240);
    this.depthTarget = new THREE.WebGLRenderTarget(320, 240, {depthBuffer: true});
    // Read the RGB pass's actual depth attachment. A separate override-material
    // scene pass would disagree on sidedness/alpha cutouts and thin surfaces.
    this.depthScene = new THREE.Scene();
    this.depthCamera = new THREE.Camera();
    this.depthScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), new THREE.ShaderMaterial({
      uniforms: {tDepth: {value: this.target.depthTexture}}, depthTest: false, depthWrite: false,
      vertexShader: 'varying vec2 vUv; void main(){vUv=uv;gl_Position=vec4(position.xy,0.0,1.0);}',
      fragmentShader: '#include <packing>\nuniform sampler2D tDepth; varying vec2 vUv; void main(){gl_FragColor=packDepthToRGBA(texture2D(tDepth,vUv).x);}',
    })));
    this.canvas = document.createElement("canvas");
    this.canvas.width = 320; this.canvas.height = 240;
  }

  capture(request, gridSize) {
    const T = this.THREE, r = this.renderer, s = this.scene, c = this.camera;
    const {width: w, height: h, near_m: near, far_m: far} = request;
    if (w !== 320 || h !== 240) throw new Error("Unsupported sensor resolution");
    c.fov = request.vertical_fov_deg; c.aspect = w / h; c.near = near; c.far = far;
    c.updateProjectionMatrix();
    const half = (gridSize - 1) / 2, p = request.position_grid;
    c.position.set((p[0] - half) * 3, p[2] * 5, (p[1] - half) * 3);
    const yaw = request.yaw_deg * Math.PI / 180, pitch = request.pitch_deg * Math.PI / 180;
    c.up.set(-Math.sin(pitch) * Math.cos(yaw), Math.cos(pitch), -Math.sin(pitch) * Math.sin(yaw));
    c.lookAt(c.position.clone().add(new T.Vector3(Math.cos(pitch) * Math.cos(yaw),
      Math.sin(pitch), Math.cos(pitch) * Math.sin(yaw))));
    c.updateMatrixWorld(true);
    const hidden = [];
    // Annotations, paths, UAV display interpolation and display-only beacons are
    // absent from BOTH RGB and depth. Architectural texture signs remain visible.
    for (const child of s.children) {
      if (!child.isLight && !child.userData.environment && child.visible) {
        hidden.push(child); child.visible = false;
      }
    }
    s.traverse(o => { if (o.isSprite && o.visible) { hidden.push(o); o.visible = false; } });
    const old = {target: r.getRenderTarget(), background: s.background, fog: s.fog,
      material: s.overrideMaterial, tone: r.toneMapping, autoClear: r.autoClear,
      clear: r.getClearColor(new T.Color()), alpha: r.getClearAlpha(), shadow: r.shadowMap.autoUpdate};
    const rgba = new Uint8Array(w * h * 4), packed = new Uint8Array(w * h * 4);
    try {
      r.autoClear = true;
      s.overrideMaterial = null;
      r.setRenderTarget(this.target); r.render(s, c);
      r.readRenderTargetPixels(this.target, 0, 0, w, h, rgba);
      r.toneMapping = T.NoToneMapping; r.shadowMap.autoUpdate = false;
      r.setClearColor(0xffffff, 1); r.setRenderTarget(this.depthTarget); r.render(this.depthScene, this.depthCamera);
      r.readRenderTargetPixels(this.depthTarget, 0, 0, w, h, packed);
    } finally {
      s.overrideMaterial = old.material; s.background = old.background; s.fog = old.fog;
      r.toneMapping = old.tone; r.autoClear = old.autoClear; r.shadowMap.autoUpdate = old.shadow;
      r.setClearColor(old.clear, old.alpha); r.setRenderTarget(old.target);
      for (const o of hidden) o.visible = true;
    }
    const pixels = new Uint8ClampedArray(rgba.length), depth = new Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dst = y * w + x, src = (h - 1 - y) * w + x, i = src * 4;
      pixels.set(rgba.subarray(i, i + 4), dst * 4);
      // Three.js packing.glsl UnpackFactors4 (checked against the vendored build).
      const z = packed[i] / 256 + packed[i + 1] / 65536 + packed[i + 2] / 16777216
        + packed[i + 3] / (255 * 16777216);
      depth[dst] = z >= 1 - 1e-8 ? 0 : Math.min(far, near * far / (far - z * (far - near)));
    }
    this.canvas.getContext("2d").putImageData(new ImageData(pixels, w, h), 0, 0);
    return {frame_id: request.frame_id, rgb: this.canvas.toDataURL("image/png"), depth_m: depth};
  }
}
