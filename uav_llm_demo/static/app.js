import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";


const sceneContainer = document.getElementById("sceneContainer");
const stepBtn = document.getElementById("stepBtn");
const autoBtn = document.getElementById("autoBtn");
const resetBtn = document.getElementById("resetBtn");
const overviewBtn = document.getElementById("overviewBtn");
const followBtn = document.getElementById("followBtn");
const runtimeButtons = [...document.querySelectorAll("[data-runtime]")];
const airsimPanel = document.getElementById("airsimPanel");
const vehicleName = document.getElementById("vehicleName");
const airsimHost = document.getElementById("airsimHost");
const airsimPort = document.getElementById("airsimPort");
const reconnectAirSimBtn = document.getElementById("reconnectAirSimBtn");
const windNorth = document.getElementById("windNorth");
const windEast = document.getElementById("windEast");
const windDown = document.getElementById("windDown");
const applyWindBtn = document.getElementById("applyWindBtn");
const scenarioSelect = document.getElementById("scenarioSelect");

const missionBadge = document.getElementById("missionBadge");
const positionValue = document.getElementById("positionValue");
const altitudeValue = document.getElementById("altitudeValue");
const batteryValue = document.getElementById("batteryValue");
const phaseValue = document.getElementById("phaseValue");
const threatValue = document.getElementById("threatValue");
const backendValue = document.getElementById("backendValue");
const observationText = document.getElementById("observationText");
const decisionJson = document.getElementById("decisionJson");
const feedbackText = document.getElementById("feedbackText");
const historyList = document.getElementById("historyList");
const stepCount = document.getElementById("stepCount");
const runtimeStatus = document.getElementById("runtimeStatus");
const kinematicsValue = document.getElementById("kinematicsValue");
const metricsValue = document.getElementById("metricsValue");

const WORLD_SCALE = 3;
const ALTITUDE_SCALE = 2.15;
const timer = new THREE.Timer();

let state = null;
let autoRunning = false;
let stepInFlight = false;
let runtimePending = false;
let cameraMode = "overview";
let environmentReady = false;
let uavGroup = null;
let targetMarker = null;
let baseMarker = null;
let pathLine = null;
let birdMeshes = new Map();
let birdTargets = new Map();
let rotorMeshes = [];
let activeTrajectory = null;
let trajectoryStartedAt = 0;
let trajectoryDurationMs = 0;
let scenariosLoaded = false;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9ec4d5);
scene.fog = new THREE.FogExp2(0xa8c7d3, 0.009);

const camera = new THREE.PerspectiveCamera(
  54,
  sceneContainer.clientWidth / sceneContainer.clientHeight,
  0.1,
  400,
);
camera.position.set(40, 42, 48);

const renderer = new THREE.WebGLRenderer({
  antialias: true,
  powerPreference: "high-performance",
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(sceneContainer.clientWidth, sceneContainer.clientHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
sceneContainer.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 16;
controls.maxDistance = 115;
controls.target.set(0, 6, 0);

setupLighting();
setupAtmosphere();


async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}


async function loadState() {
  await loadScenarios();
  state = await requestJson("/api/state");
  syncScene(true);
  renderUi();
}


async function loadScenarios() {
  if (scenariosLoaded) {
    return;
  }
  const payload = await requestJson("/api/scenarios");
  scenarioSelect.replaceChildren();
  payload.scenarios.forEach((scenario) => {
    const option = document.createElement("option");
    option.value = scenario.id;
    option.textContent = scenario.name;
    scenarioSelect.appendChild(option);
  });
  scenariosLoaded = true;
}


async function stepSimulation() {
  if (stepInFlight) {
    return;
  }

  stepInFlight = true;
  stepBtn.disabled = true;
  try {
    state = await requestJson("/api/step", { method: "POST" });
    syncScene(false);
    renderUi();
    stopAutoWhenDone();
  } catch (error) {
    feedbackText.textContent = error.message;
    stopAuto();
  } finally {
    stepInFlight = false;
    stepBtn.disabled = false;
  }
}


async function resetSimulation() {
  stopAuto();
  state = await requestJson("/api/reset", { method: "POST" });
  clearEnvironment();
  syncScene(true);
  renderUi();
}


async function switchScenario(scenarioId) {
  if (!state || runtimePending || scenarioId === state.scenario_id) {
    return;
  }
  await stopAuto(true);
  setRuntimePending(true);
  try {
    state = await requestJson("/api/scenario", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    clearEnvironment();
    syncScene(true);
    renderUi();
  } catch (error) {
    feedbackText.textContent = error.message;
    scenarioSelect.value = state.scenario_id;
  } finally {
    setRuntimePending(false);
  }
}


function toggleAuto() {
  if (autoRunning) {
    void stopAuto(true);
    return;
  }
  autoRunning = true;
  autoBtn.textContent = "暂停";
  runAutoLoop();
}


async function stopAuto(commandStop = false) {
  autoRunning = false;
  autoBtn.textContent = "自动运行";
  if (commandStop && state) {
    try {
      state = await requestJson("/api/runtime/stop", { method: "POST" });
      renderUi();
    } catch (error) {
      feedbackText.textContent = error.message;
    }
  }
}


function stopAutoWhenDone() {
  if (state && state.status !== "active") {
    stopAuto();
  }
}


async function runAutoLoop() {
  while (autoRunning && state && state.status === "active") {
    await stepSimulation();
    if (!autoRunning || !state || state.status !== "active") {
      break;
    }
    const motionDelay = Math.max(
      120,
      Number(state.trajectory_duration || 0) * 1000 + 80,
    );
    await sleep(motionDelay);
  }
  stopAutoWhenDone();
}


function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}


async function switchRuntime(backend, force = false) {
  if (runtimePending || !state) {
    return;
  }
  if (!force && backend === state.runtime_backend) {
    return;
  }
  const confirmed = window.confirm(
    `切换到 ${backend.toUpperCase()} 会停止自动运行并从基地重置任务。是否继续？`,
  );
  if (!confirmed) {
    renderRuntimeControls();
    return;
  }

  await stopAuto(true);
  setRuntimePending(true);
  try {
    state = await requestJson("/api/runtime", {
      method: "POST",
      body: JSON.stringify({
        backend,
        config: backend === "airsim" ? airsimConfig() : {},
      }),
    });
    clearEnvironment();
    syncScene(true);
    renderUi();
  } catch (error) {
    feedbackText.textContent = error.message;
    renderRuntimeControls();
  } finally {
    setRuntimePending(false);
  }
}


async function applyWind() {
  applyWindBtn.disabled = true;
  try {
    state = await requestJson("/api/runtime/wind", {
      method: "POST",
      body: JSON.stringify({
        north: Number(windNorth.value),
        east: Number(windEast.value),
        down: Number(windDown.value),
      }),
    });
    renderUi();
  } catch (error) {
    feedbackText.textContent = error.message;
  } finally {
    applyWindBtn.disabled = !state?.runtime?.connected;
  }
}


function airsimConfig() {
  return {
    host: airsimHost.value.trim(),
    port: Number(airsimPort.value),
    vehicle_name: vehicleName.value.trim(),
    wind_ned: {
      north: Number(windNorth.value),
      east: Number(windEast.value),
      down: Number(windDown.value),
    },
  };
}


function setRuntimePending(pending) {
  runtimePending = pending;
  document.body.classList.toggle("runtime-pending", pending);
  stepBtn.disabled = pending;
  autoBtn.disabled = pending || state?.status !== "active";
  resetBtn.disabled = pending;
  reconnectAirSimBtn.disabled = pending;
  applyWindBtn.disabled = pending || !state?.runtime?.connected;
  scenarioSelect.disabled = pending;
  runtimeButtons.forEach((button) => {
    button.disabled = pending;
  });
  runtimeStatus.setAttribute("aria-busy", String(pending));
  if (state) {
    renderRuntimeControls();
  }
}


function setupLighting() {
  const skyLight = new THREE.HemisphereLight(0xdff4ff, 0x506447, 2.1);
  scene.add(skyLight);

  const sun = new THREE.DirectionalLight(0xfff1d2, 4.2);
  sun.position.set(-32, 55, 18);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -55;
  sun.shadow.camera.right = 55;
  sun.shadow.camera.top = 55;
  sun.shadow.camera.bottom = -55;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 130;
  scene.add(sun);
}


function setupAtmosphere() {
  const sunDisc = new THREE.Mesh(
    new THREE.SphereGeometry(3.2, 24, 24),
    new THREE.MeshBasicMaterial({ color: 0xffe0a0 }),
  );
  sunDisc.position.set(-58, 55, -80);
  scene.add(sunDisc);

  for (let i = 0; i < 10; i += 1) {
    const cloud = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({
      color: 0xffffff,
      roughness: 1,
      transparent: true,
      opacity: 0.7,
    });
    for (let j = 0; j < 4; j += 1) {
      const puff = new THREE.Mesh(
        new THREE.SphereGeometry(2.2 + (j % 2), 12, 8),
        material,
      );
      puff.position.set(j * 2.1, Math.sin(j) * 0.8, 0);
      cloud.add(puff);
    }
    cloud.position.set(
      -65 + ((i * 19) % 130),
      28 + (i % 3) * 6,
      -58 + ((i * 31) % 116),
    );
    cloud.scale.setScalar(0.8 + (i % 3) * 0.2);
    scene.add(cloud);
  }
}


function buildEnvironment() {
  if (environmentReady) {
    return;
  }

  const terrain = new THREE.Mesh(
    new THREE.PlaneGeometry(state.grid_size * WORLD_SCALE + 22, state.grid_size * WORLD_SCALE + 22),
    new THREE.MeshStandardMaterial({ color: 0x75945e, roughness: 0.96 }),
  );
  terrain.rotation.x = -Math.PI / 2;
  terrain.receiveShadow = true;
  terrain.userData.environment = true;
  scene.add(terrain);

  addRoads();
  addBuildings();
  addTrees();

  baseMarker = createLandingMarker(0x4fd477, "base");
  baseMarker.position.copy(toWorld(state.base));
  scene.add(baseMarker);

  targetMarker = createLandingMarker(0xf2b84b, "target");
  targetMarker.position.copy(toWorld(state.target));
  scene.add(targetMarker);

  uavGroup = createDrone();
  uavGroup.position.copy(toWorld(state.uav));
  scene.add(uavGroup);

  state.dynamic_objects.forEach((item) => {
    const bird = createBird();
    bird.position.copy(toWorld(item.position));
    scene.add(bird);
    birdMeshes.set(item.id, bird);
    birdTargets.set(item.id, toWorld(item.position));
  });

  environmentReady = true;
}


function addRoads() {
  const roadMaterial = new THREE.MeshStandardMaterial({
    color: 0x4b5150,
    roughness: 0.92,
  });
  const laneMaterial = new THREE.MeshBasicMaterial({ color: 0xe7dfb4 });
  [-18, 18].forEach((offset) => {
    const road = new THREE.Mesh(new THREE.PlaneGeometry(5.4, 78), roadMaterial);
    road.rotation.x = -Math.PI / 2;
    road.position.set(offset, 0.035, 0);
    road.receiveShadow = true;
    road.userData.environment = true;
    scene.add(road);

    for (let z = -34; z <= 34; z += 6) {
      const stripe = new THREE.Mesh(new THREE.PlaneGeometry(0.15, 2.8), laneMaterial);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(offset, 0.05, z);
      stripe.userData.environment = true;
      scene.add(stripe);
    }
  });
}


function addBuildings() {
  const colors = [0xbcc4c3, 0x9da9aa, 0xc6b8a1, 0x7e9195, 0xb2aca2];
  state.buildings.forEach((building, index) => {
    const height = building.height * ALTITUDE_SCALE;
    const geometry = new THREE.BoxGeometry(WORLD_SCALE * 0.78, height, WORLD_SCALE * 0.78);
    const material = new THREE.MeshStandardMaterial({
      color: colors[index % colors.length],
      roughness: 0.72,
      metalness: 0.05,
    });
    const mesh = new THREE.Mesh(geometry, material);
    const world = toWorld({ x: building.x, y: building.y, z: 0 });
    mesh.position.set(world.x, height / 2, world.z);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.userData.environment = true;
    scene.add(mesh);

    addWindows(mesh, height, index);
  });
}


function addWindows(building, height, index) {
  if (height < 7) {
    return;
  }
  const windowMaterial = new THREE.MeshBasicMaterial({
    color: index % 3 === 0 ? 0xb8e3e8 : 0x36535a,
  });
  const rows = Math.max(2, Math.floor(height / 3.5));
  for (let row = 0; row < rows; row += 1) {
    const windowMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(0.85, 0.62),
      windowMaterial,
    );
    windowMesh.position.set(
      building.position.x,
      1.8 + row * 3.0,
      building.position.z + WORLD_SCALE * 0.397,
    );
    windowMesh.userData.environment = true;
    scene.add(windowMesh);
  }
}


function addTrees() {
  const occupied = new Set(state.buildings.map((item) => `${item.x},${item.y}`));
  for (let i = 0; i < 42; i += 1) {
    const x = (i * 7 + 2) % state.grid_size;
    const y = (i * 11 + 3) % state.grid_size;
    if (occupied.has(`${x},${y}`) || (x < 3 && y > 15) || (x > 15 && y < 5)) {
      continue;
    }
    const tree = new THREE.Group();
    const trunk = new THREE.Mesh(
      new THREE.CylinderGeometry(0.13, 0.19, 1.2, 7),
      new THREE.MeshStandardMaterial({ color: 0x6d5438, roughness: 1 }),
    );
    trunk.position.y = 0.6;
    const crown = new THREE.Mesh(
      new THREE.ConeGeometry(0.75, 2.2, 8),
      new THREE.MeshStandardMaterial({ color: 0x3d7045, roughness: 1 }),
    );
    crown.position.y = 2.05;
    trunk.castShadow = true;
    crown.castShadow = true;
    tree.add(trunk, crown);
    const world = toWorld({ x, y, z: 0 });
    tree.position.set(world.x, 0, world.z);
    tree.userData.environment = true;
    scene.add(tree);
  }
}


function createDrone() {
  const group = new THREE.Group();
  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0x233238,
    roughness: 0.35,
    metalness: 0.75,
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: 0x45d5d2,
    emissive: 0x0d6565,
    emissiveIntensity: 1.5,
    metalness: 0.35,
    roughness: 0.3,
  });

  const body = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.48, 1.15), bodyMaterial);
  body.castShadow = true;
  group.add(body);

  const cameraPod = new THREE.Mesh(
    new THREE.SphereGeometry(0.32, 16, 12),
    new THREE.MeshStandardMaterial({ color: 0x10181b, metalness: 0.8, roughness: 0.12 }),
  );
  cameraPod.position.set(0, -0.38, -0.35);
  group.add(cameraPod);

  [[1, 1], [1, -1], [-1, 1], [-1, -1]].forEach(([sx, sz], index) => {
    const arm = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.08, 1.65, 8),
      bodyMaterial,
    );
    arm.rotation.z = Math.PI / 2;
    arm.rotation.y = sx === sz ? Math.PI / 4 : -Math.PI / 4;
    arm.position.set(sx * 0.68, 0.04, sz * 0.55);
    group.add(arm);

    const motor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.22, 0.24, 14),
      accentMaterial,
    );
    motor.position.set(sx * 1.18, 0.12, sz * 1.0);
    group.add(motor);

    const rotor = new THREE.Mesh(
      new THREE.BoxGeometry(1.35, 0.025, 0.09),
      new THREE.MeshStandardMaterial({
        color: 0x101719,
        transparent: true,
        opacity: 0.7,
      }),
    );
    rotor.position.copy(motor.position);
    rotor.position.y += 0.18;
    rotor.userData.speed = index % 2 === 0 ? 1 : -1;
    rotorMeshes.push(rotor);
    group.add(rotor);
  });

  const beacon = new THREE.PointLight(0x45d5d2, 3, 7);
  beacon.position.set(0, 0.1, 0);
  group.add(beacon);
  group.scale.setScalar(0.72);
  return group;
}


function createBird() {
  const group = new THREE.Group();
  const material = new THREE.MeshStandardMaterial({
    color: 0x2e3031,
    roughness: 0.9,
    side: THREE.DoubleSide,
  });
  const body = new THREE.Mesh(new THREE.SphereGeometry(0.34, 12, 8), material);
  body.scale.set(1.4, 0.7, 0.7);
  body.castShadow = true;
  group.add(body);

  const wingGeometry = new THREE.BufferGeometry();
  wingGeometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(
      [0, 0, 0, 1.25, 0.05, 0.25, 0.25, 0, -0.18],
      3,
    ),
  );
  wingGeometry.computeVertexNormals();

  const leftWing = new THREE.Mesh(wingGeometry, material);
  const rightWing = new THREE.Mesh(wingGeometry, material);
  rightWing.scale.x = -1;
  leftWing.userData.wing = true;
  rightWing.userData.wing = true;
  group.add(leftWing, rightWing);

  const warning = new THREE.PointLight(0xff5d4b, 1.8, 5);
  group.add(warning);

  const safetyBubble = new THREE.Mesh(
    new THREE.SphereGeometry(1.05, 14, 10),
    new THREE.MeshBasicMaterial({
      color: 0xff6b57,
      transparent: true,
      opacity: 0.16,
      wireframe: true,
    }),
  );
  group.add(safetyBubble);
  group.userData.wings = [leftWing, rightWing];
  return group;
}


function createLandingMarker(color, type) {
  const group = new THREE.Group();
  const pad = new THREE.Mesh(
    new THREE.CylinderGeometry(1.25, 1.25, 0.12, 32),
    new THREE.MeshStandardMaterial({
      color: 0x263331,
      roughness: 0.72,
      metalness: 0.22,
    }),
  );
  pad.receiveShadow = true;
  group.add(pad);

  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.84, 0.08, 10, 36),
    new THREE.MeshBasicMaterial({ color }),
  );
  ring.rotation.x = Math.PI / 2;
  ring.position.y = 0.1;
  group.add(ring);

  const beam = new THREE.Mesh(
    new THREE.CylinderGeometry(0.3, 0.75, type === "target" ? 8 : 3, 18, 1, true),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: type === "target" ? 0.16 : 0.08,
      side: THREE.DoubleSide,
    }),
  );
  beam.position.y = type === "target" ? 4 : 1.5;
  group.add(beam);
  return group;
}


function syncScene(initial) {
  buildEnvironment();

  const uavTarget = toWorld(state.uav);
  uavGroup.userData.targetPosition = uavTarget;
  if (initial) {
    uavGroup.position.copy(uavTarget);
    activeTrajectory = null;
  } else if (state.trajectory && state.trajectory.length > 1) {
    const trajectoryPoints = state.trajectory.map((point) => toWorld(point));
    activeTrajectory = new THREE.CatmullRomCurve3(
      trajectoryPoints,
      false,
      "centripetal",
      0.35,
    );
    trajectoryStartedAt = performance.now();
    trajectoryDurationMs = Math.max(
      120,
      Number(state.trajectory_duration || 0.9) * 1000,
    );
  }

  state.dynamic_objects.forEach((item) => {
    let bird = birdMeshes.get(item.id);
    if (!bird) {
      bird = createBird();
      scene.add(bird);
      birdMeshes.set(item.id, bird);
    }
    const target = toWorld(item.position);
    birdTargets.set(item.id, target);
    if (initial) {
      bird.position.copy(target);
    }
    const velocity = new THREE.Vector3(item.velocity.x, 0, item.velocity.y);
    if (velocity.lengthSq() > 0.001) {
      bird.rotation.y = Math.atan2(velocity.x, velocity.z);
    }
  });

  updatePath();
}


function updatePath() {
  if (pathLine) {
    scene.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
  }
  const points = (state.continuous_path || state.path).map((point) => toWorld(point));
  if (points.length < 2) {
    return;
  }
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({
    color: 0x45d5d2,
    transparent: true,
    opacity: 0.92,
  });
  pathLine = new THREE.Line(geometry, material);
  scene.add(pathLine);
}


function clearEnvironment() {
  [...scene.children].forEach((child) => {
    if (child.userData.environment || child === uavGroup || birdMeshesHas(child) || child === pathLine || child === targetMarker || child === baseMarker) {
      scene.remove(child);
    }
  });
  birdMeshes.clear();
  birdTargets.clear();
  rotorMeshes = [];
  uavGroup = null;
  pathLine = null;
  targetMarker = null;
  baseMarker = null;
  environmentReady = false;
}


function birdMeshesHas(object) {
  for (const bird of birdMeshes.values()) {
    if (bird === object) {
      return true;
    }
  }
  return false;
}


function renderUi() {
  missionBadge.textContent = state.status;
  missionBadge.style.color = state.status === "completed" ? "#62d57b" : state.status === "failed" ? "#ff6b57" : "#45d5d2";
  positionValue.textContent = [
    `(${formatNumber(state.uav.x)}`,
    formatNumber(state.uav.y),
    `${formatNumber(state.uav.z)})`,
  ].join(", ");
  const altitudeScale = Number(
    state.runtime?.meters_per_altitude_level
      || state.meters_per_altitude_level
      || 5,
  );
  altitudeValue.textContent = `${(state.uav.z * altitudeScale).toFixed(1)} m`;
  batteryValue.textContent = `${state.battery}%`;
  phaseValue.textContent = state.phase;
  backendValue.textContent = state.agent_model
    ? `${state.agent_backend}:${state.agent_model}`
    : state.agent_backend;
  stepCount.textContent = `${state.step_count} steps`;
  scenarioSelect.value = state.scenario_id;
  const metrics = state.metrics || {};
  metricsValue.textContent = [
    `success ${metrics.success ? "yes" : "no"}`,
    `interventions ${metrics.safety_interventions ?? 0}`,
    `path ${Number(metrics.path_length || 0).toFixed(1)}`,
    `battery -${metrics.battery_used ?? 0}%`,
  ].join(" | ");
  const runtime = state.runtime || {};
  const connected = Boolean(runtime.connected);
  renderRuntimeControls();
  applyWindBtn.disabled = runtimePending || !connected;

  if (runtime.host) {
    setInputUnlessFocused(airsimHost, runtime.host);
  }
  if (runtime.port) {
    setInputUnlessFocused(airsimPort, runtime.port);
  }
  if (runtime.vehicle_name) {
    setInputUnlessFocused(vehicleName, runtime.vehicle_name);
  }
  const wind = runtime.wind_ned || {};
  setInputUnlessFocused(windNorth, wind.x ?? 0);
  setInputUnlessFocused(windEast, wind.y ?? 0);
  setInputUnlessFocused(windDown, wind.z ?? 0);

  const kinematics = state.kinematics || {};
  const orientation = kinematics.orientation || {};
  const velocity = kinematics.linear_velocity || {};
  kinematicsValue.textContent = [
    `6-DOF: ${kinematics.source || "unknown"}`,
    `roll ${formatNumber(orientation.roll)} rad`,
    `pitch ${formatNumber(orientation.pitch)} rad`,
    `yaw ${formatNumber(orientation.yaw)} rad`,
    `v ${formatVector(velocity)}`,
  ].join(" | ");

  const threat = state.observation.nearest_dynamic_obstacle;
  threatValue.textContent = threat
    ? `${threat.threat_level} / ${threat.predicted_distance.toFixed(1)}`
    : "none";
  threatValue.style.color = threatColor(threat ? threat.threat_level : "low");

  const latest = state.history[0];
  observationText.textContent = latest ? latest.observation : state.observation.text;
  decisionJson.textContent = latest ? JSON.stringify(latest.decision, null, 2) : "{}";
  feedbackText.textContent = latest ? latest.feedback : state.feedback;
  stepBtn.disabled = runtimePending;
  autoBtn.disabled = runtimePending || state.status !== "active";
  resetBtn.disabled = runtimePending;

  historyList.innerHTML = "";
  if (!state.history.length) {
    const empty = document.createElement("p");
    empty.className = "empty-log";
    empty.textContent = "等待第一次 Agent 决策。";
    historyList.appendChild(empty);
    return;
  }

  state.history.forEach((entry) => {
    const item = document.createElement("article");
    item.className = "history-item";
    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = `STEP ${entry.step}`;
    const action = document.createElement("code");
    const proposed = entry.decision.proposed_action;
    action.textContent = proposed && proposed !== entry.decision.action
      ? `${proposed} -> ${entry.decision.action}`
      : entry.decision.action;
    if (entry.decision.safety_intervened) {
      action.classList.add("intervened");
    }
    header.append(title, action);

    const reason = document.createElement("p");
    reason.textContent = entry.decision.reason;
    const feedback = document.createElement("p");
    feedback.textContent = entry.feedback;
    item.append(header, reason, feedback);
    historyList.appendChild(item);
  });
}


function renderRuntimeControls() {
  if (!state) {
    return;
  }
  const runtime = state.runtime || {};
  const connected = Boolean(runtime.connected);
  runtimeStatus.textContent = runtimePending
    ? `${state.runtime_backend} / switching`
    : `${state.runtime_backend} / ${connected ? "connected" : "offline"}`;
  runtimeStatus.style.color = connected ? "#62d57b" : "#ff6b57";
  airsimPanel.classList.toggle("is-hidden", state.runtime_backend !== "airsim");
  runtimeButtons.forEach((button) => {
    const selected = button.dataset.runtime === state.runtime_backend;
    button.setAttribute("aria-checked", String(selected));
    button.classList.toggle("active", selected);
  });
}


function setInputUnlessFocused(input, value) {
  if (document.activeElement !== input) {
    input.value = String(value);
  }
}


function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "--";
}


function formatVector(vector) {
  return `(${formatNumber(vector.x)}, ${formatNumber(vector.y)}, ${formatNumber(vector.z)})`;
}


function threatColor(level) {
  if (level === "critical" || level === "high") {
    return "#ff6b57";
  }
  if (level === "medium") {
    return "#f2b84b";
  }
  return "#62d57b";
}


function toWorld(point) {
  const half = state ? (state.grid_size - 1) / 2 : 9.5;
  return new THREE.Vector3(
    (Number(point.x) - half) * WORLD_SCALE,
    Number(point.z) * ALTITUDE_SCALE,
    (Number(point.y) - half) * WORLD_SCALE,
  );
}


function setCameraMode(mode) {
  cameraMode = mode;
  overviewBtn.classList.toggle("active", mode === "overview");
  followBtn.classList.toggle("active", mode === "follow");
  controls.enabled = mode === "overview";
  if (mode === "overview") {
    camera.position.set(40, 42, 48);
    controls.target.set(0, 6, 0);
  }
}


function animate() {
  timer.update();
  const delta = Math.min(timer.getDelta(), 0.05);
  const elapsed = timer.getElapsed();

  if (uavGroup) {
    if (activeTrajectory) {
      const progress = Math.min(
        1,
        (performance.now() - trajectoryStartedAt) / trajectoryDurationMs,
      );
      const position = activeTrajectory.getPoint(progress);
      const tangent = activeTrajectory.getTangent(Math.min(1, progress + 0.01));
      uavGroup.position.copy(position);
      if (Math.abs(tangent.x) + Math.abs(tangent.z) > 0.001) {
        uavGroup.rotation.y = Math.atan2(tangent.x, tangent.z);
      }
      const bank = THREE.MathUtils.clamp(-tangent.x * 0.12, -0.16, 0.16);
      uavGroup.rotation.z = bank + Math.sin(elapsed * 2.2) * 0.018;
      if (progress >= 1) {
        activeTrajectory = null;
        uavGroup.position.copy(uavGroup.userData.targetPosition);
      }
    } else if (uavGroup.userData.targetPosition) {
      uavGroup.position.lerp(uavGroup.userData.targetPosition, 1 - Math.exp(-5.5 * delta));
      uavGroup.rotation.z = Math.sin(elapsed * 2.2) * 0.025;
    }
    rotorMeshes.forEach((rotor) => {
      rotor.rotation.y += delta * 42 * rotor.userData.speed;
    });
  }

  birdMeshes.forEach((bird, id) => {
    const target = birdTargets.get(id);
    if (target) {
      bird.position.lerp(target, 1 - Math.exp(-3.6 * delta));
    }
    bird.userData.wings.forEach((wing, index) => {
      wing.rotation.z = Math.sin(elapsed * 8.5 + index * Math.PI) * 0.5;
    });
  });

  if (targetMarker) {
    targetMarker.rotation.y += delta * 0.35;
  }

  if (cameraMode === "follow" && uavGroup) {
    const desired = uavGroup.position.clone().add(new THREE.Vector3(9, 7, 12));
    camera.position.lerp(desired, 1 - Math.exp(-3.2 * delta));
    camera.lookAt(uavGroup.position);
  } else {
    controls.update();
  }

  renderer.render(scene, camera);
}


function onResize() {
  const width = sceneContainer.clientWidth;
  const height = sceneContainer.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}


stepBtn.addEventListener("click", stepSimulation);
autoBtn.addEventListener("click", toggleAuto);
resetBtn.addEventListener("click", resetSimulation);
scenarioSelect.addEventListener("change", () => switchScenario(scenarioSelect.value));
overviewBtn.addEventListener("click", () => setCameraMode("overview"));
followBtn.addEventListener("click", () => setCameraMode("follow"));
runtimeButtons.forEach((button) => {
  button.addEventListener("click", () => switchRuntime(button.dataset.runtime));
});
reconnectAirSimBtn.addEventListener("click", () => switchRuntime("airsim", true));
applyWindBtn.addEventListener("click", applyWind);
window.addEventListener("resize", onResize);

renderer.setAnimationLoop(animate);
loadState().catch((error) => {
  feedbackText.textContent = error.message;
});
