import * as THREE from "three";

const PALETTES = {
  glass: { wall: "#68818f", frame: "#a8b7bc", glazing: "#2b4654", roof: "#666e72" },
  brick: { wall: "#796d5d", frame: "#a3967d", glazing: "#3f4849", roof: "#66655f" },
  bronze: { wall: "#80735c", frame: "#b6a580", glazing: "#4a5150", roof: "#6b695f" },
  stone: { wall: "#b0aea4", frame: "#c5c2b9", glazing: "#4b5b64", roof: "#747876" },
  terracotta: { wall: "#8d7764", frame: "#b1a089", glazing: "#424d50", roof: "#716960" },
  silver: { wall: "#98a1a5", frame: "#bec5c7", glazing: "#3b5260", roof: "#707879" },
};

function noise(x, y, seed = 0) {
  const value = Math.sin(x * 127.1 + y * 311.7 + seed * 74.7) * 43758.5453;
  return value - Math.floor(value);
}

function canvasTexture(canvas) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export function urbanSurfaceTexture(kind) {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = kind === "asphalt" ? "#34393b" : "#868780";
  ctx.fillRect(0, 0, 256, 256);
  for (let y = 0; y < 256; y += 2) {
    for (let x = 0; x < 256; x += 2) {
      const grain = Math.floor(noise(x, y) * 60);
      ctx.fillStyle = `rgba(${grain + 90},${grain + 90},${grain + 85},0.16)`;
      ctx.fillRect(x, y, 2, 2);
    }
  }
  if (kind !== "asphalt") {
    ctx.strokeStyle = "rgba(45,49,47,0.25)";
    ctx.lineWidth = 1;
    for (let cell = 0; cell <= 256; cell += 64) {
      ctx.beginPath(); ctx.moveTo(cell, 0); ctx.lineTo(cell, 256); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, cell); ctx.lineTo(256, cell); ctx.stroke();
    }
  }
  const texture = canvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function facadeMap(width, height, palette, use, profile, groundFloor) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 512;
  const ctx = canvas.getContext("2d");
  const curtain = ["office", "mixed"].includes(use) && palette !== PALETTES.stone;
  const floors = Math.max(1, Math.round(height / 3.1));
  const bays = Math.max(2, Math.round(width / (curtain ? 1.1 : 1.15)));
  const floorHeight = canvas.height / floors;
  const bayWidth = canvas.width / bays;
  const pier = Math.max(2, canvas.width * (curtain ? 0.055 : 0.12) / width);
  const sill = Math.max(3, canvas.height * (curtain ? 0.1 : 0.26) / height);
  ctx.fillStyle = palette.wall;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let floor = 0; floor < floors; floor += 1) {
    const y = canvas.height - (floor + 1) * floorHeight;
    const storefront = groundFloor && floor === 0;
    for (let bay = 0; bay < bays; bay += 1) {
      const x = bay * bayWidth + pier;
      const top = storefront ? sill * 0.6 : sill;
      const windowHeight = floorHeight - top - sill;
      ctx.fillStyle = palette.glazing;
      ctx.fillRect(x, y + top, bayWidth - pier * 2, windowHeight);
      const reflected = noise(bay, floor, width) * 0.16;
      ctx.fillStyle = `rgba(191,205,204,${0.04 + reflected})`;
      ctx.fillRect(x, y + top, bayWidth - pier * 2, windowHeight * 0.55);
      if (noise(bay, floor, height) > 0.96) {
        ctx.fillStyle = "rgba(214,177,108,0.26)";
        ctx.fillRect(x, y + top, bayWidth - pier * 2, windowHeight);
      }
      ctx.fillStyle = "rgba(18,24,26,0.42)";
      ctx.fillRect(x, y + top + windowHeight - 2, bayWidth - pier * 2, 2);
    }
    ctx.fillStyle = palette.frame;
    ctx.fillRect(0, y, canvas.width, Math.max(2, sill * 0.18));
    if (floor % 5 === 0) {
      ctx.fillStyle = "rgba(32,35,35,0.15)";
      ctx.fillRect(0, y, canvas.width, Math.max(3, sill * 0.3));
    }
  }
  if (profile === "braced") {
    ctx.strokeStyle = "#605e52";
    ctx.lineWidth = Math.max(6, canvas.width * 0.13 / width);
    for (let section = 0; section < 3; section += 1) {
      const top = section * canvas.height / 3;
      const bottom = (section + 1) * canvas.height / 3;
      ctx.beginPath(); ctx.moveTo(0, top); ctx.lineTo(canvas.width, bottom);
      ctx.moveTo(canvas.width, top); ctx.lineTo(0, bottom); ctx.stroke();
    }
  }
  return canvasTexture(canvas);
}

function addMass(group, bounds, bottom, top, palette, use, profile, toWorld, scale,
                 altitudeScale, groundFloor, textures, parcelId) {
  const [left, right, north, south] = bounds;
  const width = (right - left + 0.78) * scale;
  const depth = (south - north + 0.78) * scale;
  const fullHeight = (top - bottom) * altitudeScale;
  // Roof details occupy the declared obstacle volume, never extra airspace.
  const roofSpace = Math.min(0.8, fullHeight * 0.08);
  const height = fullHeight - roofSpace;
  function material(faceWidth) {
    const key = JSON.stringify([faceWidth, height, palette, use, profile, groundFloor]);
    if (!textures.has(key)) textures.set(key, facadeMap(faceWidth, height, palette, use, profile, groundFloor));
    const glass = [PALETTES.glass, PALETTES.silver, PALETTES.bronze].includes(palette);
    return new THREE.MeshStandardMaterial({ map: textures.get(key),
      roughness: glass ? 0.38 : 0.85, metalness: glass ? 0.18 : 0.02 });
  }
  const roofMaterial = new THREE.MeshStandardMaterial({
    color: palette.roof, map: textures.get("roof"), roughness: 0.94,
  });
  let mesh;
  if (profile === "round") {
    const geometry = new THREE.CylinderGeometry(0.5, 0.5, height, 32);
    geometry.scale(width, 1, depth);
    mesh = new THREE.Mesh(geometry, [material(Math.PI * (width + depth) / 2), roofMaterial, roofMaterial]);
  } else {
    const sideX = material(depth);
    const sideZ = material(width);
    mesh = new THREE.Mesh(new THREE.BoxGeometry(width, height, depth),
      [sideX, sideX, roofMaterial, roofMaterial, sideZ, sideZ]);
  }
  const center = toWorld({ x: (left + right) / 2, y: (north + south) / 2, z: bottom });
  mesh.position.set(center.x, bottom * altitudeScale + height / 2, center.z);
  mesh.castShadow = mesh.receiveShadow = true;
  mesh.userData.cameraOccluder = true;
  mesh.userData.parcelId = parcelId;
  group.add(mesh);
  const roofY = bottom * altitudeScale + height;
  const rim = new THREE.Mesh(profile === "round"
    ? new THREE.CylinderGeometry(0.5, 0.5, roofSpace * 0.15, 32).scale(width * 0.97, 1, depth * 0.97)
    : new THREE.BoxGeometry(width * 0.97, roofSpace * 0.15, depth * 0.97), roofMaterial);
  rim.position.set(center.x, roofY + roofSpace * 0.18, center.z);
  rim.userData.parcelId = parcelId;
  group.add(rim);
  if (width > 3 && depth > 3) {
    for (let unit = 0; unit < 2; unit += 1) {
      const equipment = new THREE.Mesh(new THREE.BoxGeometry(width * 0.16, roofSpace * 0.65, depth * 0.18),
        new THREE.MeshStandardMaterial({ color: "#8a8c81", roughness: 0.88 }));
      equipment.position.set(center.x + width * (unit ? 0.2 : -0.2),
        roofY + roofSpace * 0.45, center.z);
      equipment.castShadow = true;
      equipment.userData.parcelId = parcelId;
      group.add(equipment);
    }
  }
}

function addStreetTree(group, x, y, toWorld) {
  const position = toWorld({ x, y, z: 0 });
  const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.09, 0.14, 2.2, 6),
    new THREE.MeshStandardMaterial({ color: "#514c3e", roughness: 1 }));
  trunk.position.set(position.x, 1.1, position.z);
  trunk.castShadow = true;
  group.add(trunk);
  for (let leaf = 0; leaf < 3; leaf += 1) {
    const crown = new THREE.Mesh(new THREE.IcosahedronGeometry(0.75, 1),
      new THREE.MeshStandardMaterial({ color: ["#455346", "#4d5d46", "#536348"][leaf], roughness: 1 }));
    crown.position.set(position.x + (leaf - 1) * 0.36, 2.9 + (leaf % 2) * 0.35, position.z);
    crown.scale.set(0.95, 1.3, 0.85);
    crown.castShadow = true;
    group.add(crown);
  }
}

export function buildUrbanCity(scenario, toWorld, scale, altitudeScale, roadLayout) {
  const group = new THREE.Group();
  group.name = "urban_city";
  group.userData.environment = true;
  const textures = new Map([["roof", urbanSurfaceTexture("concrete")]]);
  for (const parcel of scenario.urban_parcels) {
    const palette = PALETTES[parcel.facade] || PALETTES.stone;
    const profile = parcel.profile || "slab";
    addMass(group, parcel.footprint, 0, parcel.podium_height, palette, parcel.use,
      parcel.tower ? "slab" : profile, toWorld, scale, altitudeScale, true, textures, parcel.id);
    if (parcel.tower) {
      const [left, right, north, south, top] = parcel.tower;
      addMass(group, [left, right, north, south], parcel.podium_height, top, palette,
        parcel.use, profile, toWorld, scale, altitudeScale, false, textures, parcel.id);
    }
  }
  // The wider skyline is decorative and remains outside the flight grid.
  const roadBounds = roadLayout.map(road => [
    road.position.x - road.size.x / (2 * scale), road.position.x + road.size.x / (2 * scale),
    road.position.y - road.size.y / (2 * scale), road.position.y + road.size.y / (2 * scale),
  ]);
  const edge = scenario.grid_size - 1;
  for (let block = 0; block < 4; block += 1) {
    const along = block * 12 - 3;
    for (let side = 0; side < 4; side += 1) {
      const bounds = side === 0 ? [-10, -5, along, along + 3]
        : side === 1 ? [edge + 5, edge + 10, along, along + 3]
        : side === 2 ? [along, along + 3, -10, -5]
        : [along, along + 3, edge + 5, edge + 10];
      if (roadBounds.some(road => bounds[0] - 0.39 < road[1] && bounds[1] + 0.39 > road[0]
          && bounds[2] - 0.39 < road[3] && bounds[3] + 0.39 > road[2])) continue;
      const palette = Object.values(PALETTES)[(block + side) % 6];
      const variant = block + side * 3;
      const profile = variant % 11 === 0 ? "round" : "slab";
      addMass(group, bounds, 0, 3 + ((block * 3 + side * 2) % 7), palette, "office",
        profile, toWorld, scale, altitudeScale,
        true, textures, null);
    }
  }
  const centers = scenario.urban_street_centers ?? [(scenario.grid_size - 1) / 2];
  for (const center of centers) {
    for (let coordinate = 2.5; coordinate < edge; coordinate += 5) {
      if (centers.some((crossing) => Math.abs(coordinate - crossing) < 3)) continue;
      addStreetTree(group, center - 1.8, coordinate, toWorld);
      addStreetTree(group, center + 1.8, coordinate, toWorld);
      addStreetTree(group, coordinate, center - 1.8, toWorld);
      addStreetTree(group, coordinate, center + 1.8, toWorld);
    }
  }
  return group;
}
