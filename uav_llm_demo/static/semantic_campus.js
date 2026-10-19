// A compact architectural reconstruction of Beihang's Hangzhou campus.
// Geometry and navigation share campus_layout; x/y = 3 m, elevation = 5 m.
export function buildSemanticCampus(THREE, group, scenario) {
  const layout = scenario.campus_layout;
  if (!layout) return group;
  const half = (scenario.grid_size - 1) / 2;
  const X = value => (value - half) * 3;
  const Z = value => (value - half) * 3;
  const rect = ([left, right, north, south]) => ({
    left: X(left - 0.39), right: X(right + 0.39),
    north: Z(north - 0.39), south: Z(south + 0.39),
    x: X((left + right) / 2), z: Z((north + south) / 2),
    width: (right - left + 0.78) * 3, depth: (south - north + 0.78) * 3,
  });
  group.name = "beihang_hangzhou_campus";
  group.userData.environment = true;
  group.userData.layoutVersion = layout.version || 1;
  const batches = new Map();
  const unitBox = new THREE.BoxGeometry(1, 1, 1);
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const v = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const colors = {
    stone: "#dedfd8", trim: "#e8e9e1", silver: "#cdd1cc", glass: "#4b7175",
    bronze: "#a8957a", roof: "#969d99", water: "#648f8c", path: "#cdcbb9",
    grass: "#929f70", grassDark: "#7e8f61", bark: "#756e55", asphalt: "#7c8380",
  };
  function material(color, options = {}) {
    return new THREE.MeshStandardMaterial({color, roughness: 0.78, ...options});
  }
  const stone = material(colors.stone);
  const trim = material(colors.trim);
  const roofConcrete = material(colors.roof);
  const darkMetal = material("#718180", {metalness: 0.35, roughness: 0.55});
  const paleMetal = material("#d9dfda", {metalness: 0.38, roughness: 0.4});
  const waterMaterial = material(colors.water, {metalness: 0.16, roughness: 0.3});
  const stonePaving = material(colors.path);
  const lineMaterial = new THREE.LineBasicMaterial({color: "#e8e8d8", transparent: true, opacity: 0.85});

  function textureFrom(canvas, repeat = false) {
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = 4;
    if (repeat) texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
    return texture;
  }
  function canvas(width, height) {
    const image = document.createElement("canvas");
    image.width = width; image.height = height;
    return [image, image.getContext("2d")];
  }
  // One tile represents one structural bay and one floor. All buildings reuse it.
  function facadeTexture(kind) {
    const [image, ctx] = canvas(128, 160);
    const bronze = kind === "bronze";
    const curtain = kind === "glass";
    const ribbon = kind === "ribbon";
    ctx.fillStyle = bronze ? "#b3a18a" : "#d7dbd6";
    ctx.fillRect(0, 0, 128, 160);
    const gradient = ctx.createLinearGradient(0, 0, 128, 160);
    gradient.addColorStop(0, curtain ? "#6e9699" : "#718b8b");
    gradient.addColorStop(0.6, curtain ? "#3c656c" : "#48656a");
    gradient.addColorStop(1, "#809791");
    ctx.fillStyle = gradient;
    ctx.fillRect(curtain ? 5 : 9, curtain ? 6 : 26, curtain ? 118 : 110, curtain ? 148 : 105);
    ctx.fillStyle = "rgba(235,241,230,0.23)";
    ctx.fillRect(8, 9, 18, 139);
    ctx.fillStyle = curtain ? "#d3dcd4" : "#c4ceca";
    ctx.fillRect(61, curtain ? 4 : 24, 4, curtain ? 152 : 109);
    ctx.fillRect(6, 80, 116, 3);
    ctx.fillStyle = "rgba(25,52,54,0.26)";
    ctx.fillRect(5, 150, 118, 6);
    if (ribbon) {
      ctx.fillStyle = "#e3e5dc";
      ctx.fillRect(0, 0, 128, 27); ctx.fillRect(0, 130, 128, 30);
      ctx.fillStyle = "#aebeb7"; ctx.fillRect(0, 130, 128, 3);
    }
    if (bronze) {
      ctx.fillStyle = "#89735d";
      for (let x = 14; x < 120; x += 12) ctx.fillRect(x, 24, 3, 111);
      ctx.fillStyle = "#cfbea1";
      ctx.fillRect(0, 133, 128, 7);
    }
    return textureFrom(image, true);
  }
  const facades = Object.fromEntries(["stone", "glass", "bronze", "ribbon"].map(kind => [kind,
    material("#ffffff", {map: facadeTexture(kind), metalness: kind === "glass" ? 0.24 : 0.08,
      roughness: kind === "glass" ? 0.38 : 0.66})]));
  const [roofCanvas, roofContext] = canvas(128, 128);
  roofContext.fillStyle = "#d1d5d0"; roofContext.fillRect(0, 0, 128, 128);
  roofContext.fillStyle = "#b5bfbc"; roofContext.fillRect(0, 0, 2, 128);
  roofContext.fillStyle = "rgba(240,243,235,0.7)"; roofContext.fillRect(3, 0, 1, 128);
  roofContext.fillStyle = "#c1c8c2"; roofContext.fillRect(0, 0, 128, 1);
  const roofSheet = material("#ffffff", {map: textureFrom(roofCanvas, true), metalness: 0.45,
    roughness: 0.48, side: THREE.DoubleSide});

  function box(name, x, y, z, width, height, depth, mat, solid = false) {
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth), mat);
    mesh.name = name; mesh.position.set(x, y, z);
    mesh.castShadow = solid; mesh.receiveShadow = true;
    mesh.userData.cameraOccluder = solid;
    group.add(mesh);
    return mesh;
  }
  function instance(name, geometry, mat, x, y, z, width, height, depth, angle = 0) {
    const key = `${name}:${mat.uuid}`;
    if (!batches.has(key)) batches.set(key, {name, geometry, mat, transforms: []});
    quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), angle);
    matrix.compose(v.set(x, y, z), quaternion, scale.set(width, height, depth));
    batches.get(key).transforms.push(matrix.clone());
  }
  function detail(x, y, z, width, height, depth, mat = trim, angle = 0) {
    instance("campus_architectural_details", unitBox, mat, x, y, z, width, height, depth, angle);
  }
  function strip(points, width, mat, height, name, curb = false) {
    for (let i = 1; i < points.length; i++) {
      const ax = X(points[i - 1][0]), az = Z(points[i - 1][1]);
      const bx = X(points[i][0]), bz = Z(points[i][1]);
      const length = Math.hypot(bx - ax, bz - az);
      if (length < 0.01) continue;
      const angle = Math.atan2(bx - ax, bz - az);
      instance(name, unitBox, mat, (ax + bx) / 2, height, (az + bz) / 2,
        width * 3, 0.12, length + 0.05, angle);
      if (curb) {
        const dx = Math.cos(angle) * (width * 1.5 + 0.14);
        const dz = -Math.sin(angle) * (width * 1.5 + 0.14);
        for (const side of [-1, 1]) detail((ax + bx) / 2 + dx * side, height + 0.08,
          (az + bz) / 2 + dz * side, 0.25, 0.18, length, stonePaving, angle);
      }
    }
  }

  // The river and roads are deliberately ground surfaces, never hidden aerial obstacles.
  for (const water of layout.waterways || []) {
    strip(water.points, water.width, waterMaterial, 0.13, `campus_water_${water.id}`, true);
  }
  const roadMaterial = material(colors.asphalt);
  for (const road of layout.roads || []) {
    const bridge = road.kind === "bridge" || /bridge|entry/.test(road.id);
    strip(road.points, road.width, bridge ? stonePaving : roadMaterial,
      bridge ? 0.35 : 0.22, `campus_road_${road.id}`, true);
    if (!bridge && road.width >= 1.1) {
      const segments = [];
      for (let i = 1; i < road.points.length; i++) {
        const [ax, az] = road.points[i - 1], [bx, bz] = road.points[i];
        const length = Math.hypot(bx - ax, bz - az);
        for (let d = 0.4; d < length - 0.4; d += 1.6) {
          const a = d / length, b = Math.min(d + 0.65, length - 0.2) / length;
          segments.push(new THREE.Vector3(X(ax + (bx - ax) * a), 0.285, Z(az + (bz - az) * a)),
            new THREE.Vector3(X(ax + (bx - ax) * b), 0.285, Z(az + (bz - az) * b)));
        }
      }
      const marks = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(segments), lineMaterial);
      marks.name = `campus_road_marks_${road.id}`; group.add(marks);
    }
  }

  function sportTexture(kind) {
    const [image, ctx] = canvas(512, 768);
    const w = image.width, h = image.height;
    ctx.fillStyle = "#b4b7a3"; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#eeeede"; ctx.lineWidth = 3;
    const rounded = (x, y, width, height, radius) => {
      ctx.beginPath(); ctx.roundRect(x, y, width, height, radius);
    };
    function football(left, top, width, height) {
      for (let i = 0; i < 10; i++) {
        ctx.fillStyle = i % 2 ? "#7f9c59" : "#789451";
        ctx.fillRect(left, top + height * i / 10, width, height / 10);
      }
      ctx.strokeStyle = "#f1eee0"; ctx.lineWidth = 2.8;
      ctx.strokeRect(left + 6, top + 6, width - 12, height - 12);
      ctx.beginPath(); ctx.moveTo(left + 6, top + height / 2); ctx.lineTo(left + width - 6, top + height / 2); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(left + width / 2, top + height / 2, width * 0.16, width * 0.16, 0, 0, Math.PI * 2); ctx.stroke();
      for (const end of [0, 1]) {
        const edge = top + (end ? height - 6 : 6), direction = end ? -1 : 1;
        ctx.strokeRect(left + width * 0.23, edge, width * 0.54, direction * height * 0.15);
        ctx.strokeRect(left + width * 0.37, edge, width * 0.26, direction * height * 0.055);
      }
    }
    if (kind === "running_track" || kind === "stadium") {
      ctx.fillStyle = "#929e72"; ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#ac6552"; rounded(14, 12, w - 28, h - 24, 240); ctx.fill();
      for (let lane = 0; lane < 7; lane++) {
        const inset = 22 + lane * 10;
        ctx.lineWidth = lane === 0 ? 3 : 1.7;
        ctx.strokeStyle = "#e5c9b2";
        rounded(inset, 20 + lane * 10, w - inset * 2, h - 40 - lane * 20, 233 - lane * 10); ctx.stroke();
      }
      ctx.fillStyle = "#809652"; rounded(98, 96, w - 196, h - 192, 147); ctx.fill();
      football(110, 193, w - 220, h - 386);
      ctx.strokeStyle = "#f2e5d5"; ctx.lineWidth = 2;
      for (let lane = 0; lane < 6; lane++) {
        ctx.beginPath(); ctx.moveTo(28 + lane * 10, 462 + lane * 5); ctx.lineTo(36 + lane * 10, 462 + lane * 5); ctx.stroke();
      }
      ctx.fillStyle = "#a2b9bf"; ctx.fillRect(219, 136, 74, 24); ctx.fillRect(219, 611, 74, 24);
    } else if (kind === "football") {
      football(18, 24, w - 36, h - 48);
    } else {
      ctx.fillStyle = kind === "basketball" ? "#819eaa" : "#799b8d"; ctx.fillRect(0, 0, w, h);
      const cols = kind === "basketball" ? 3 : 2, rows = kind === "basketball" ? 3 : 2;
      for (let col = 0; col < cols; col++) for (let row = 0; row < rows; row++) {
        const cw = w / cols, ch = h / rows;
        const x = col * cw + 13, y = row * ch + 16, iw = cw - 26, ih = ch - 32;
        ctx.fillStyle = kind === "tennis" ? "#6c9a82" : kind === "volleyball" ? "#c39679" : "#547e96";
        ctx.fillRect(x, y, iw, ih); ctx.strokeStyle = "#e5ece2"; ctx.lineWidth = 2;
        ctx.strokeRect(x, y, iw, ih);
        ctx.beginPath(); ctx.moveTo(x, y + ih / 2); ctx.lineTo(x + iw, y + ih / 2); ctx.stroke();
        if (kind === "basketball") {
          ctx.beginPath(); ctx.arc(x + iw / 2, y + ih / 2, iw * 0.2, 0, Math.PI * 2); ctx.stroke();
          for (const end of [0, 1]) {
            ctx.strokeRect(x + iw * 0.33, y + (end ? ih * 0.75 : 0), iw * 0.34, ih * 0.25);
            ctx.beginPath(); ctx.arc(x + iw / 2, y + (end ? ih * 0.75 : ih * 0.25), iw * 0.27,
              end ? Math.PI : 0, end ? Math.PI * 2 : Math.PI); ctx.stroke();
          }
        } else {
          ctx.strokeRect(x + iw * 0.14, y, iw * 0.72, ih);
          ctx.strokeRect(x + iw * 0.14, y + ih * 0.22, iw * 0.72, ih * 0.56);
        }
      }
    }
    return textureFrom(image);
  }
  function plazaTexture() {
    const [image, ctx] = canvas(768, 768);
    ctx.fillStyle = "#c8c7b6"; ctx.fillRect(0, 0, 768, 768);
    ctx.strokeStyle = "rgba(104,114,108,0.2)"; ctx.lineWidth = 1;
    for (let i = 0; i <= 768; i += 24) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 768); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(768, i); ctx.stroke();
    }
    ctx.fillStyle = "#839b68";
    for (const left of [26, 428]) {
      ctx.beginPath(); ctx.moveTo(left, 24); ctx.lineTo(left + 308, 24);
      ctx.lineTo(left + 308, 493); ctx.lineTo(left, 660); ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = "#e4e1ce"; ctx.lineWidth = 13;
    for (const left of [26, 428]) {
      ctx.beginPath(); ctx.moveTo(left, 595); ctx.lineTo(left + 308, 302); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(left, 387); ctx.lineTo(left + 308, 654); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(left + 154, 180, 102, 136, 0, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = "#e1deca"; ctx.fillRect(346, 0, 76, 768);
    ctx.fillStyle = "#a4ad8d"; ctx.fillRect(371, 30, 25, 622);
    ctx.fillStyle = "#7c9990"; ctx.fillRect(365, 644, 35, 30);
    return textureFrom(image);
  }
  function landscapeTexture(kind, id) {
    const [image, ctx] = canvas(512, 512);
    const garden = kind === "garden";
    ctx.fillStyle = garden ? "#91a27b" : "#c5c8b9"; ctx.fillRect(0, 0, 512, 512);
    ctx.strokeStyle = garden ? "rgba(105,130,91,0.13)" : "rgba(104,118,105,0.15)";
    ctx.lineWidth = 1;
    for (let n = 0; n <= 512; n += 22) {
      ctx.beginPath(); ctx.moveTo(n, 0); ctx.lineTo(n, 512); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, n); ctx.lineTo(512, n); ctx.stroke();
    }
    if (garden) {
      ctx.strokeStyle = "#d9d7c2"; ctx.lineWidth = 14;
      ctx.beginPath(); ctx.moveTo(10, 417); ctx.bezierCurveTo(60, 185, 394, 440, 500, 46); ctx.stroke();
      ctx.beginPath(); ctx.ellipse(170, 237, 106, 159, 0.55, 0, Math.PI * 2); ctx.stroke();
      ctx.lineWidth = 8;
      ctx.beginPath(); ctx.moveTo(26, 19); ctx.lineTo(485, 460); ctx.stroke();
    } else if (/research/.test(id)) {
      ctx.fillStyle = "#849a74";
      ctx.fillRect(161, 110, 187, 101); ctx.fillRect(161, 351, 187, 51);
      ctx.fillStyle = "#6e958d"; ctx.fillRect(362, 105, 25, 106);
      ctx.strokeStyle = "#e2dfcc"; ctx.lineWidth = 8;
      ctx.strokeRect(153, 102, 203, 117); ctx.strokeRect(153, 343, 203, 67);
    }
    return textureFrom(image);
  }
  for (const surface of layout.surfaces || []) {
    const r = rect(surface.footprint);
    const sport = ["running_track", "stadium", "football", "basketball", "tennis", "volleyball"].includes(surface.kind);
    const plaza = surface.kind === "plaza" || surface.kind === "south_square";
    const landscaped = ["garden", "paving"].includes(surface.kind);
    const mat = sport || plaza || landscaped ? material("#ffffff", {map: sport ? sportTexture(surface.kind)
      : plaza ? plazaTexture() : landscapeTexture(surface.kind, surface.id), roughness: 1})
      : material(surface.color || (surface.kind === "lawn" ? colors.grass : colors.path));
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(r.width, r.depth), mat);
    mesh.rotation.x = -Math.PI / 2;
    mesh.position.set(r.x, surface.kind === "launch" ? 0.007 : sport ? 0.1 : 0.07, r.z);
    mesh.receiveShadow = true; mesh.name = `campus_surface_${surface.id}`;
    group.add(mesh);
    if (surface.kind === "running_track") {
      // The west spectator terrace stays below 2.7 m, inside the sports parcel.
      for (let step = 0; step < 4; step++) detail(r.left + 0.22 + step * 0.4,
        0.25 + (3 - step) * 0.19, r.z, 0.4, 0.22 + (3 - step) * 0.38, r.depth * 0.57, stone);
      const bleachers = new THREE.Mesh(new THREE.PlaneGeometry(2.3, r.depth * 0.61), paleMetal);
      bleachers.rotation.x = -Math.PI / 2; bleachers.position.set(r.left + 1.15, 2.55, r.z);
      bleachers.name = "campus_stadium_stand_canopy"; group.add(bleachers);
    }
  }

  // Keep facade glass behind the flush structural trim. Coincident opaque
  // faces otherwise alternate in the depth buffer as the camera moves.
  const facadeRecess = 0.03;
  const buildingCores = new Map((layout.buildings || []).map(building => {
    const r = rect(building.footprint);
    return [building.id, {left: r.left + facadeRecess, right: r.right - facadeRecess,
      north: r.north + facadeRecess, south: r.south - facadeRecess}];
  }));
  for (const building of layout.buildings || []) {
    const r = rect(building.footprint), ceiling = building.height * 5;
    const height = Math.max(1, ceiling - 0.58);
    const kind = building.kind === "research" ? "ribbon"
      : Object.hasOwn(facades, building.facade) ? building.facade
      : /dorm|residen|apartment/.test(building.kind) ? "bronze" : "stone";
    const core = {...buildingCores.get(building.id)};
    // The research middle wing joins its spine. Render that shared volume once;
    // the navigation footprint still contains the full original building cells.
    if (building.kind === "research" && building.id.endsWith("-middle")) {
      const spine = buildingCores.get(building.id.replace(/-middle$/, "-spine"));
      if (spine && core.north >= spine.north && core.south <= spine.south) {
        if (core.left === spine.left && core.right > spine.right) core.left = spine.right;
        else if (core.right === spine.right && core.left < spine.left) core.right = spine.left;
      }
    }
    const coreWidth = core.right - core.left, coreDepth = core.south - core.north;
    const geometry = new THREE.BoxGeometry(coreWidth, height, coreDepth);
    const uvs = geometry.attributes.uv;
    // BoxGeometry creates four vertices for each face: +/- X, +/- Y, +/- Z.
    for (let face = 0; face < 6; face++) {
      const span = face < 2 ? coreDepth : coreWidth;
      for (let corner = 0; corner < 4; corner++) {
        const index = face * 4 + corner;
        uvs.setXY(index, uvs.getX(index) * span / 2.7,
          uvs.getY(index) * (face === 2 || face === 3 ? coreDepth : height) / 2.8);
      }
    }
    const faceIndices = [...geometry.index.array];
    geometry.setIndex([0, 1, 4, 5, 2, 3].flatMap(face => faceIndices.slice(face * 6, face * 6 + 6)));
    geometry.clearGroups(); geometry.addGroup(0, 24, 0); geometry.addGroup(24, 12, 1);
    const mesh = new THREE.Mesh(geometry, [facades[kind], roofConcrete]);
    mesh.position.set((core.left + core.right) / 2, height / 2 + 0.03,
      (core.north + core.south) / 2);
    mesh.name = `campus_building_${building.id}`;
    mesh.userData.cameraOccluder = true;
    mesh.userData.campusBuilding = building.id;
    mesh.userData.footprint = building.footprint;
    mesh.castShadow = mesh.receiveShadow = true; group.add(mesh);
    // Flush slab edges and slender parapets preserve a continuous facade silhouette.
    const thickness = 0.24;
    for (const z of [r.north + thickness / 2, r.south - thickness / 2]) {
      detail(r.x, ceiling - 0.25, z, r.width, 0.5, thickness);
    }
    for (const x of [r.left + thickness / 2, r.right - thickness / 2]) {
      detail(x, ceiling - 0.25, r.z, thickness, 0.5, r.depth);
    }
    for (const x of [r.left + 0.14, r.right - 0.14]) {
      for (const z of [r.north + 0.14, r.south - 0.14]) detail(x, height / 2, z, 0.28, height, 0.28);
    }
    // The plinth and corner columns share one finish at their flush junction.
    detail(r.x, 0.28, r.z, r.width, 0.5, r.depth, trim);
    if (["research", "residential", "faculty"].includes(building.kind)) {
      for (let level = 2.8; level < height - 0.4; level += 2.8) {
        for (const z of [r.north + 0.065, r.south - 0.065]) detail(r.x, level,
          z, r.width, 0.16, 0.13, trim);
        for (const x of [r.left + 0.065, r.right - 0.065]) detail(x, level,
          r.z, 0.13, 0.16, r.depth, trim);
      }
    }
    if (r.width > 7 && r.depth > 5) {
      detail(r.x + r.width * 0.19, ceiling - 0.37, r.z, Math.min(3.3, r.width * 0.18), 0.37,
        Math.min(2.3, r.depth * 0.22), darkMetal);
      detail(r.x - r.width * 0.13, ceiling - 0.39, r.z + r.depth * 0.16, 1.8, 0.33, 1.2, paleMetal);
    }
    if (r.width >= 10 && kind !== "bronze") {
      // A light entrance frame, contained within the authoritative building volume.
      const entryWidth = Math.min(5, r.width * 0.28);
      detail(r.x, 3.1, r.south - 0.16, entryWidth, 0.24, 0.32);
      for (const offset of [-1, 1]) detail(r.x + offset * entryWidth / 2, 1.6,
        r.south - 0.14, 0.22, 3.2, 0.28);
    }
  }

  const curvedRoofs = (layout.roofs || []).filter(item => item.type === "curved");
  const mainRoof = curvedRoofs[0];
  const longNorth = Math.min(...curvedRoofs.map(item => rect(item.footprint).north));
  const longSouth = Math.max(...curvedRoofs.map(item => rect(item.footprint).south));
  for (const roof of layout.roofs || []) {
    const r = rect(roof.footprint), holes = (roof.holes || []).map(rect);
    const skylights = (roof.skylights || []).map(rect);
    const openings = [...holes, ...skylights];
    const wave = roof.type === "wave", ribbed = roof.type === "ribbed";
    const heightAt = (x, z) => {
      if (wave) {
        const t = (x - r.left) / r.width;
        return (roof.height - roof.sag * (0.5 + 0.5 * Math.sin(t * Math.PI * 5 - Math.PI / 2))) * 5;
      }
      // Both northern and southern wings lie on one long, gently sagging roof.
      const u = (z - (longNorth + longSouth) / 2) / Math.max(1, (longSouth - longNorth) / 2);
      const profile = ribbed ? mainRoof : roof;
      return Math.min(roof.height * 5, (profile.height - profile.sag * (1 - u * u)) * 5);
    };
    function cuts(start, end, values) {
      const out = [start, end, ...values.filter(value => value > start && value < end)];
      const count = Math.ceil((end - start) / (wave ? 0.6 : 1.8));
      for (let i = 1; i < count; i++) out.push(start + (end - start) * i / count);
      return [...new Set(out)].sort((a, b) => a - b);
    }
    const xs = cuts(r.left, r.right, openings.flatMap(h => [h.left, h.right]));
    const ribEdges = [];
    if (ribbed) for (let z = r.north; z < r.south; z += 1.2) ribEdges.push(z, Math.min(r.south, z + 0.43));
    const zs = cuts(r.north, r.south, [...openings.flatMap(h => [h.north, h.south]), ...ribEdges]);
    const positions = [], uv = [], normals = [];
    const thickness = 0.32;
    const point = (x, z, underside = false) => [x, heightAt(x, z) - (underside ? thickness : 0), z];
    function face(a, b, c, d) {
      const faceNormal = new THREE.Vector3().crossVectors(
        new THREE.Vector3(b[0] - a[0], b[1] - a[1], b[2] - a[2]),
        new THREE.Vector3(d[0] - a[0], d[1] - a[1], d[2] - a[2])).normalize();
      for (const p of [a, b, d, b, c, d]) {
        positions.push(...p); uv.push(p[0] / 1.7, p[2] / 5.5);
        if (Math.abs(faceNormal.y) > 0.2) {
          const delta = 0.03;
          const dx = (heightAt(p[0] + delta, p[2]) - heightAt(p[0] - delta, p[2])) / (2 * delta);
          const dz = (heightAt(p[0], p[2] + delta) - heightAt(p[0], p[2] - delta)) / (2 * delta);
          const smooth = new THREE.Vector3(-dx, 1, -dz).normalize().multiplyScalar(Math.sign(faceNormal.y));
          normals.push(smooth.x, smooth.y, smooth.z);
        } else normals.push(faceNormal.x, faceNormal.y, faceNormal.z);
      }
    }
    for (let i = 1; i < xs.length; i++) for (let j = 1; j < zs.length; j++) {
      const x0 = xs[i - 1], x1 = xs[i], z0 = zs[j - 1], z1 = zs[j];
      if (ribbed && (((z0 + z1) / 2 - r.north) % 1.2) > 0.43) continue;
      if (openings.some(h => (x0 + x1) / 2 > h.left && (x0 + x1) / 2 < h.right
        && (z0 + z1) / 2 > h.north && (z0 + z1) / 2 < h.south)) continue;
      face(point(x0, z0), point(x0, z1), point(x1, z1), point(x1, z0));
      face(point(x1, z0, true), point(x1, z1, true), point(x0, z1, true), point(x0, z0, true));
    }
    const boundaries = [r, ...openings];
    for (const boundary of boundaries) {
      for (const z of [boundary.north, boundary.south]) {
        const cutsX = cuts(boundary.left, boundary.right, []);
        for (let i = 1; i < cutsX.length; i++) face(point(cutsX[i - 1], z), point(cutsX[i], z),
          point(cutsX[i], z, true), point(cutsX[i - 1], z, true));
      }
      for (const x of [boundary.left, boundary.right]) {
        const cutsZ = cuts(boundary.north, boundary.south, []);
        for (let i = 1; i < cutsZ.length; i++) face(point(x, cutsZ[i - 1]), point(x, cutsZ[i]),
          point(x, cutsZ[i], true), point(x, cutsZ[i - 1], true));
      }
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geometry.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
    const mesh = new THREE.Mesh(geometry, roofSheet);
    mesh.name = `campus_roof_${roof.id}`; mesh.castShadow = mesh.receiveShadow = true;
    mesh.userData.cameraOccluder = true; mesh.userData.campusRoof = roof.id;
    mesh.userData.holes = roof.holes || []; group.add(mesh);
    for (const [index, skylight] of skylights.entries()) {
      const glassPositions = [];
      const glassXs = cuts(skylight.left, skylight.right, []);
      const glassZs = cuts(skylight.north, skylight.south, []);
      for (let i = 1; i < glassXs.length; i++) for (let j = 1; j < glassZs.length; j++) {
        for (const [x, z] of [[glassXs[i - 1], glassZs[j - 1]], [glassXs[i - 1], glassZs[j]],
          [glassXs[i], glassZs[j - 1]], [glassXs[i - 1], glassZs[j]],
          [glassXs[i], glassZs[j]], [glassXs[i], glassZs[j - 1]]]) glassPositions.push(x, heightAt(x, z) - 0.16, z);
      }
      const paneGeometry = new THREE.BufferGeometry();
      paneGeometry.setAttribute("position", new THREE.Float32BufferAttribute(glassPositions, 3));
      paneGeometry.computeVertexNormals();
      const pane = new THREE.Mesh(paneGeometry, material("#6c8c89", {roughness: 0.25, metalness: 0.38,
        side: THREE.DoubleSide}));
      pane.name = `campus_skylight_${roof.id}_${index}`; pane.receiveShadow = true; group.add(pane);
      const frameMat = new THREE.LineBasicMaterial({color: "#aebcba"});
      const framePoints = [];
      for (let x = skylight.left; x <= skylight.right; x += 1.4) {
        for (let j = 1; j < glassZs.length; j++) {
          framePoints.push(new THREE.Vector3(x, heightAt(x, glassZs[j - 1]) - 0.12, glassZs[j - 1]),
            new THREE.Vector3(x, heightAt(x, glassZs[j]) - 0.12, glassZs[j]));
        }
      }
      for (let z = skylight.north; z <= skylight.south; z += 1.4) framePoints.push(
        new THREE.Vector3(skylight.left, heightAt(skylight.left, z) - 0.12, z),
        new THREE.Vector3(skylight.right, heightAt(skylight.right, z) - 0.12, z));
      const frame = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(framePoints), frameMat);
      frame.name = `campus_skylight_frames_${roof.id}_${index}`; group.add(frame);
    }
    // Close the wave-roof end walls within the same reserved sports hall volume.
    if (wave) {
      const sportsBuilding = (layout.buildings || []).find(item => item.kind === "sports_hall");
      const wallBase = (sportsBuilding?.height || roof.height - roof.sag) * 5 - 0.58;
      for (const z of [r.north + 0.04, r.south - 0.04]) {
        const wallPositions = [], wallUV = [];
        for (let i = 1; i < xs.length; i++) {
          const a = xs[i - 1], b = xs[i];
          for (const [x, y] of [[a, wallBase], [b, wallBase], [a, heightAt(a, z) - thickness],
            [b, wallBase], [b, heightAt(b, z) - thickness], [a, heightAt(a, z) - thickness]]) {
            wallPositions.push(x, y, z); wallUV.push((x - r.left) / 2.7, y / 2.8);
          }
        }
        const wallGeometry = new THREE.BufferGeometry();
        wallGeometry.setAttribute("position", new THREE.Float32BufferAttribute(wallPositions, 3));
        wallGeometry.setAttribute("uv", new THREE.Float32BufferAttribute(wallUV, 2));
        wallGeometry.computeVertexNormals();
        const wallMaterial = facades.glass.clone(); wallMaterial.side = THREE.DoubleSide;
        const wall = new THREE.Mesh(wallGeometry, wallMaterial);
        wall.name = "campus_sports_wave_endwall"; wall.castShadow = wall.receiveShadow = true; group.add(wall);
      }
    }
    // Thin steel columns are inside the same reserved roof footprint.
    if (!wave) {
      for (const x of [r.left + 0.75, r.right - 0.75]) {
        for (let z = r.north + 1.2; z <= r.south - 1.2; z += 8.4) {
          if (holes.some(h => x > h.left && x < h.right && z > h.north && z < h.south)) continue;
          const columnHeight = heightAt(x, z) - thickness;
          detail(x, columnHeight / 2, z, 0.25, columnHeight, 0.25, paleMetal);
        }
      }
    }
  }

  const canopyGeometry = new THREE.IcosahedronGeometry(1, 1);
  const trunkGeometry = new THREE.CylinderGeometry(0.11, 0.15, 1, 5);
  const treeMaterials = ["#648565", "#759371", "#839a70"].map(color => material(color, {roughness: 1}));
  const trunkMaterial = material(colors.bark);
  const home = scenario.points_of_interest?.find(item => item.id === "home")?.position;
  for (const [index, tree] of (layout.trees || []).entries()) {
    if (home && Math.hypot(tree.x - home.x, tree.y - home.y) * 3 < 5.2) continue;
    const height = Math.min(2.7, tree.height * 5);
    const radius = Math.min(1.65, tree.radius * 3);
    const x = X(tree.x), z = Z(tree.y);
    instance("campus_tree_trunks", trunkGeometry, trunkMaterial, x, height * 0.22, z,
      1, height * 0.44, 1);
    instance("campus_tree_canopies", canopyGeometry, treeMaterials[index % 3],
      x, height * 0.64, z, radius, height * 0.36, radius * 0.91);
  }

  // The low paired entrance piers and bridge rails echo the south gate plaza.
  const southEntry = (layout.roads || []).find(road => road.id === "south-entry");
  if (southEntry) {
    for (const side of [-1, 1]) {
      detail(X(48) + side * 3.3, 1.24, Z(91.5), 0.62, 2.4, 0.72, trim);
      detail(X(48) + side * 3.9, 0.64, Z(92), 0.16, 0.75, 10.2, paleMetal);
    }
    // A low monument sign is east of the entry bridge, clear of the launch pad.
    const [signCanvas, signContext] = canvas(1024, 256);
    signContext.fillStyle = "#e5e7df"; signContext.fillRect(0, 0, 1024, 256);
    signContext.fillStyle = "#385b76"; signContext.fillRect(28, 32, 8, 187);
    signContext.font = '600 84px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
    signContext.textAlign = "center"; signContext.textBaseline = "middle";
    signContext.fillText("北京航空航天大学", 535, 89, 900);
    signContext.fillStyle = "#677c83";
    signContext.font = '500 49px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
    signContext.fillText("杭州国际校园", 535, 179, 780);
    const sign = box("campus_south_gate_sign", X(54), 1.25, Z(90.7), 9.6, 2.35, 0.4, stone, true);
    const signFace = new THREE.Mesh(new THREE.PlaneGeometry(9.6, 2.35),
      material("#ffffff", {map: textureFrom(signCanvas), roughness: 0.8}));
    signFace.position.set(sign.position.x, sign.position.y, sign.position.z + 0.205);
    signFace.name = "campus_south_gate_name"; group.add(signFace);
  }

  function label(text, x, y, z, color, width = 0.12) {
    const [image, ctx] = canvas(512, 100);
    ctx.fillStyle = "rgba(247,247,238,0.9)"; ctx.beginPath(); ctx.roundRect(2, 2, 508, 96, 12); ctx.fill();
    ctx.fillStyle = color; ctx.fillRect(13, 18, 5, 64);
    ctx.font = '500 45px "Microsoft YaHei", "Noto Sans CJK SC", sans-serif';
    ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = "#304548";
    ctx.fillText(text, 266, 53, 461);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: textureFrom(image),
      transparent: true, depthWrite: false, depthTest: false, sizeAttenuation: false,
      toneMapped: false}));
    sprite.position.set(x, y, z); sprite.scale.set(width, width * 100 / 512, 1);
    sprite.renderOrder = 4; return sprite;
  }
  for (const site of scenario.points_of_interest || []) {
    const position = site.position;
    const color = site.id === "home" ? "#3b986f" : "#466f82";
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.75, 0.88, 48),
      new THREE.MeshBasicMaterial({color, side: THREE.DoubleSide, transparent: true, opacity: 0.8, depthWrite: false}));
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(X(position.x), 0.26, Z(position.y));
    ring.name = `poi_marker_${site.id}`; group.add(ring);
    const offset = site.id === "conference" ? -4 : site.id === "library" ? 4 : 0;
    const annotationHeight = position.z * 5 + 2.4;
    const annotation = label(site.label, X(position.x) + offset, annotationHeight, Z(position.y), color,
      site.label.length > 9 ? 0.15 : 0.12);
    annotation.name = `poi_label_${site.id}`; group.add(annotation);
    const leader = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(X(position.x), 0.3, Z(position.y)),
      new THREE.Vector3(X(position.x), annotationHeight - 0.5, Z(position.y)),
      new THREE.Vector3(X(position.x) + offset, annotationHeight - 0.5, Z(position.y)),
    ]), new THREE.LineBasicMaterial({color, transparent: true, opacity: 0.35}));
    leader.name = `poi_leader_${site.id}`; group.add(leader);
  }

  for (const batch of batches.values()) {
    if (!batch.transforms.length) continue;
    const mesh = new THREE.InstancedMesh(batch.geometry, batch.mat, batch.transforms.length);
    batch.transforms.forEach((transform, index) => mesh.setMatrixAt(index, transform));
    mesh.name = batch.name; mesh.instanceMatrix.needsUpdate = true;
    mesh.castShadow = !batch.name.startsWith("campus_water") && !batch.name.startsWith("campus_road");
    mesh.receiveShadow = true; mesh.computeBoundingSphere(); group.add(mesh);
  }
  return group;
}
