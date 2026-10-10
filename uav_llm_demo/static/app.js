import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { sampleTraffic } from "./traffic.js";
import { buildUrbanCity, urbanSurfaceTexture } from "./urban_city.js";


const sceneContainer = document.getElementById("sceneContainer");
const planBtn = document.getElementById("planBtn");
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const overviewBtn = document.getElementById("overviewBtn");
const followBtn = document.getElementById("followBtn");
const agentButtons = [...document.querySelectorAll("[data-agent]")];
const runtimeButtons = [...document.querySelectorAll("[data-runtime]")];
const agentKeyBtn = document.getElementById("agentKeyBtn");
const airsimConfigBtn = document.getElementById("airsimConfigBtn");
const vehicleName = document.getElementById("vehicleName");
const airsimHost = document.getElementById("airsimHost");
const airsimPort = document.getElementById("airsimPort");
const reconnectAirSimBtn = document.getElementById("reconnectAirSimBtn");
const windNorth = document.getElementById("windNorth");
const windEast = document.getElementById("windEast");
const windDown = document.getElementById("windDown");
const applyWindBtn = document.getElementById("applyWindBtn");
const scenarioSelect = document.getElementById("scenarioSelect");

const simulator = document.getElementById("simulator")
  || document.querySelector(".simulator");
const agentDialog = document.getElementById("agentDialog");
const agentForm = document.getElementById("agentForm");
const closeAgentBtn = document.getElementById("closeAgentBtn");
const cancelAgentBtn = document.getElementById("cancelAgentBtn");
const deepseekApiKey = document.getElementById("deepseekApiKey");
const deepseekModel = document.getElementById("deepseekModel");
const deepseekBaseUrl = document.getElementById("deepseekBaseUrl");
const saveAgentBtn = document.getElementById("saveAgentBtn");
const agentDialogError = document.getElementById("agentDialogError");
const airsimDialog = document.getElementById("airsimDialog");
const closeAirSimBtn = document.getElementById("closeAirSimBtn");
const closeAirSimFooterBtn = document.getElementById("closeAirSimFooterBtn");
const currentStepValue = document.getElementById("currentStepValue");
const goalDirectionValue = document.getElementById("goalDirectionValue");
const goalDistanceValue = document.getElementById("goalDistanceValue");
const staticRiskValue = document.getElementById("staticRiskValue");
const dynamicRiskValue = document.getElementById("dynamicRiskValue");
const agentActionValue = document.getElementById("agentActionValue");
const agentRiskValue = document.getElementById("agentRiskValue");
const agentReasonValue = document.getElementById("agentReasonValue");
const proposedActionValue = document.getElementById("proposedActionValue");
const executedActionValue = document.getElementById("executedActionValue");
const guardStatus = document.getElementById("guardStatus");
const executionPositionValue = document.getElementById("executionPositionValue");
const resultActionValue = document.getElementById("resultActionValue");

const positionValue = document.getElementById("positionValue");
const altitudeValue = document.getElementById("altitudeValue");
const batteryValue = document.getElementById("batteryValue");
const telemetryPhaseValue = document.getElementById("telemetryPhaseValue");
const phaseValue = document.getElementById("phaseValue");
const threatValue = document.getElementById("threatValue");
const backendValue = document.getElementById("backendValue");
const feedbackText = document.getElementById("feedbackText");
const missionProgressText = document.getElementById("missionProgressText");
const recoveryText = document.getElementById("recoveryText");
const runtimeStatus = document.getElementById("runtimeStatus");

const WORLD_SCALE = 3;
const ALTITUDE_SCALE = 5;
const timer = new THREE.Timer();

let state = null;
let trafficMotion = null;
let commandPending = false;
let pollActive = false;
let lastSequence = -1;
let commandError = "";
let plannedRoute = null;
let plannedRouteKey = "";
let avoidanceRoute = null;
let pathKey = "";
let cameraMode = "overview";
let followOffset = new THREE.Vector3(9, 7, 12);
const followAnchor = new THREE.Vector3();
let cameraObstacles = [];
let environmentReady = false;
let displayedGridSize = 0;
let uavGroup = null;
let pathLine = null;
let goalStar = null;
const goalReward = document.createElement("div");
goalReward.className = "goal-reward";
goalReward.setAttribute("role", "status");
goalReward.setAttribute("aria-live", "polite");
goalReward.hidden = true;
sceneContainer.appendChild(goalReward);
let birdMeshes = new Map();
let hoveredBird = null;
const birdPointer = new THREE.Vector2(2, 2);
const birdRaycaster = new THREE.Raycaster();
const birdTooltip = document.createElement("div");
birdTooltip.className = "bird-tooltip";
birdTooltip.hidden = true;
sceneContainer.appendChild(birdTooltip);
let rotorMeshes = [];
let scenariosLoaded = false;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0xbfc5c6, 0.002);

const camera = new THREE.PerspectiveCamera(
  54,
  sceneContainer.clientWidth / sceneContainer.clientHeight,
  0.1,
  400,
);
camera.position.set(55, 60, 65);

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
renderer.toneMappingExposure = 0.95;
sceneContainer.appendChild(renderer.domElement);
renderer.domElement.addEventListener("pointermove", (event) => {
  const bounds = renderer.domElement.getBoundingClientRect();
  birdPointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1,
    -(event.clientY - bounds.top) / bounds.height * 2 + 1);
});
renderer.domElement.addEventListener("pointerleave", () => birdPointer.set(2, 2));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.maxPolarAngle = Math.PI * 0.48;
controls.minDistance = 16;
controls.maxDistance = 115;
controls.target.set(0, 17, 0);

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


function acceptState(payload) {
  if (!Number.isFinite(payload.sequence) || payload.sequence <= lastSequence) {
    return false;
  }
  const initial = !state
    || payload.diagnostics?.run_id !== state.diagnostics?.run_id
    || payload.scenario_id !== state.scenario_id
    || payload.runtime_backend !== state.runtime_backend
    || ["x", "y", "z"].some((axis) => payload.target?.[axis] !== state.target?.[axis])
    || (payload.flight?.plan_id === 0 && state.flight?.plan_id !== 0)
    || (payload.continuous_path || payload.path || []).length
      < (state.continuous_path || state.path || []).length;
  lastSequence = payload.sequence;
  state = payload;
  if (hasOwn(payload, "traffic_motion")) {
    trafficMotion = payload.traffic_motion;
  }
  if (initial) clearEnvironment();
  syncScene(initial);
  renderUi();
  return true;
}


async function pollState() {
  if (pollActive) return;
  pollActive = true;
  while (pollActive) {
    try {
      acceptState(await requestJson("/api/state"));
    } catch {
      setText(feedbackText, "Telemetry unavailable; reconnecting");
    }
    await sleep(125);
  }
}


// Mutations share one gate; reads remain independent during planner latency.
async function runCommand(action) {
  if (commandPending || !state) return false;
  commandPending = true;
  commandError = "";
  renderUi();
  try {
    await action();
    return true;
  } catch {
    // Do not echo provider errors that might contain credentials.
    commandError = "Request failed. Check the connection and configuration.";
    return false;
  } finally {
    commandPending = false;
    renderUi();
  }
}


async function postState(url, body) {
  const payload = await requestJson(url, {
    method: "POST",
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  acceptState(payload);
}


async function toggleMission() {
  return runCommand(() => postState(
    state.flight?.running || state.flight?.planning
      ? "/api/runtime/stop" : "/api/mission/start",
  ));
}


async function planMission() {
  return runCommand(() => postState("/api/mission/plan"));
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


async function resetSimulation() {
  return runCommand(() => postState("/api/reset"));
}


async function switchScenario(scenarioId) {
  if (scenarioId === state?.scenario_id) return;
  return runCommand(() => postState("/api/scenario", { scenario_id: scenarioId }));
}


async function switchAgent(backend, force = false, config = {}) {
  if (!state || commandPending) return false;
  const normalized = String(backend || "").trim().toLowerCase();
  if (!["mock", "deepseek"].includes(normalized)) return false;
  if (!force && normalized === state.agent_backend) return true;
  if (normalized === "deepseek" && !config.api_key && !state.deepseek_key_configured) {
    openAgentDialog();
    return false;
  }
  const switched = await runCommand(async () => {
    // Config preserves the route and running state in the backend, including
    // invalidating any old planner generation. No frontend pause/resume race.
    await postState("/api/config", { backend: normalized, ...config });
    closeDialog(agentDialog);
  });
  if (!switched && agentDialogError) {
    setText(agentDialogError, commandError);
    agentDialogError.hidden = false;
  }
  return switched;
}


async function configureDeepSeekFromDialog(event) {
  event?.preventDefault();
  if (commandPending) {
    return;
  }
  const config = {
    model: deepseekModel?.value.trim(),
    base_url: deepseekBaseUrl?.value.trim(),
  };
  const apiKey = deepseekApiKey?.value.trim();
  if (apiKey) {
    config.api_key = apiKey;
  }
  const switched = await switchAgent("deepseek", true, config);
  if (switched) {
    if (deepseekApiKey) {
      deepseekApiKey.value = "";
    }
    if (agentDialogError) {
      agentDialogError.hidden = true;
      agentDialogError.textContent = "";
    }
  }
}


function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}


async function switchRuntime(backend, force = false) {
  if (commandPending || !state || (!force && backend === state.runtime_backend)) return;
  if (!window.confirm(`Switch to ${backend.toUpperCase()} and reset the mission?`)) return;
  return runCommand(() => postState("/api/runtime", {
    backend,
    config: backend === "airsim" ? airsimConfig() : {},
  }));
}


async function applyWind() {
  return runCommand(() => postState("/api/runtime/wind", {
    north: Number(windNorth.value),
    east: Number(windEast.value),
    down: Number(windDown.value),
  }));
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


function setupLighting() {
  const skyLight = new THREE.HemisphereLight(0xb6c9db, 0x4d4942, 0.85);
  scene.add(skyLight);

  const sun = new THREE.DirectionalLight(0xffdfae, 3.2);
  sun.position.set(-90, 110, 60);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  sun.shadow.camera.left = -55;
  sun.shadow.camera.right = 55;
  sun.shadow.camera.top = 55;
  sun.shadow.camera.bottom = -55;
  sun.shadow.camera.near = 1;
  sun.shadow.camera.far = 300;
  sun.shadow.bias = -0.0002;
  sun.shadow.normalBias = 0.04;
  scene.add(sun);
}


function setupAtmosphere() {
  const canvas = document.createElement("canvas");
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext("2d");
  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
  sky.addColorStop(0, "#738da3");
  sky.addColorStop(0.3, "#a9b8c3");
  sky.addColorStop(0.49, "#b9c4cf");
  sky.addColorStop(0.56, "#a8aaa3");
  sky.addColorStop(1, "#6c746e");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  // Distant layered clouds replace the low floating spheres in the flight scene.
  for (let layer = 0; layer < 5; layer += 1) {
    for (let cloud = 0; cloud < 13; cloud += 1) {
      const x = cloud * 180 + Math.sin(cloud * 2 + layer) * 95;
      const y = 260 + layer * 50 + Math.sin(cloud * 1.7 + layer) * 25;
      const gradient = ctx.createRadialGradient(x, y, 2, x, y, 130);
      gradient.addColorStop(0, "rgba(245,239,225,0.48)");
      gradient.addColorStop(1, "rgba(245,239,225,0)");
      ctx.save(); ctx.translate(x, y); ctx.scale(1.8, 0.35); ctx.translate(-x, -y);
      ctx.fillStyle = gradient;
      ctx.fillRect(x - 130, y - 130, 260, 260);
      ctx.restore();
    }
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.mapping = THREE.EquirectangularReflectionMapping;
  scene.background = texture;
  scene.environment = texture;
  scene.environmentIntensity = 0.35;
}


function buildEnvironment() {
  if (environmentReady) {
    return;
  }

  const urbanParcels = state.scenario?.urban_parcels || [];
  const mapScale = Math.max(1, state.grid_size / 20);
  controls.maxDistance = 115 * mapScale;
  scene.fog.density = 0.002 / mapScale;
  if (displayedGridSize !== state.grid_size) {
    displayedGridSize = state.grid_size;
    if (cameraMode === "overview") setCameraMode("overview");
    const sun = scene.children.find((object) => object.isDirectionalLight);
    const shadow = sun.shadow.camera;
    shadow.left = shadow.bottom = -55 * mapScale;
    shadow.right = shadow.top = 55 * mapScale;
    shadow.far = 300 * mapScale;
    shadow.updateProjectionMatrix();
  }
  const beaconHeight = Math.max(36, ...state.scene_layout
    .filter((item) => item.name.startsWith("building_"))
    .map((item) => item.position.z * ALTITUDE_SCALE + item.size.z / 2 + 12));
  const surfaceTextures = new Map();
  state.scene_layout.forEach((item) => {
    if (urbanParcels.length && item.name.startsWith("building_")) return;
    let map = null;
    if (urbanParcels.length && (item.name === "ground" || item.name.startsWith("road_")
        || item.name.startsWith("sidewalk_"))) {
      const kind = item.name.startsWith("road_") ? "asphalt" : "concrete";
      const key = `${kind}:${item.size.x}:${item.size.y}`;
      if (!surfaceTextures.has(key)) {
        const texture = urbanSurfaceTexture(kind);
        texture.repeat.set(item.size.x / 8, item.size.y / 8);
        surfaceTextures.set(key, texture);
      }
      map = surfaceTextures.get(key);
    }
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(item.size.x, item.size.z, item.size.y),
      new THREE.MeshStandardMaterial({ color: map ? "#ffffff" : item.color, map, roughness: 0.9 }),
    );
    mesh.position.copy(toWorld(item.position));
    mesh.castShadow = item.name.startsWith("building_");
    mesh.receiveShadow = true;
    mesh.userData.environment = true;
    mesh.userData.cameraOccluder = item.name.startsWith("building_");
    scene.add(mesh);
    if (item.name === "base_pad" || item.name === "target_pad") {
      scene.add(createEndpointBeacon(item, beaconHeight));
    }
  });
  if (urbanParcels.length) {
    const roads = state.scene_layout.filter(item => item.name.startsWith("road_"));
    scene.add(buildUrbanCity(state.scenario, toWorld, WORLD_SCALE, ALTITUDE_SCALE, roads));
  }
  scene.updateMatrixWorld(true);
  scene.traverse((object) => {
    if (object.userData.cameraOccluder) {
      cameraObstacles.push(new THREE.Box3().setFromObject(object).expandByScalar(0.15));
    }
  });

  goalStar = createGoalStar();
  scene.add(goalStar);

  uavGroup = createDrone();
  uavGroup.position.copy(toWorld(state.uav));
  scene.add(uavGroup);
  if (cameraMode === "follow") setCameraMode("follow");

  state.dynamic_objects.forEach((item) => {
    const bird = createBird(item);
    bird.position.copy(toWorld(item.position));
    scene.add(bird);
    birdMeshes.set(item.id, bird);
  });

  environmentReady = true;
}


function createGoalStar() {
  const group = new THREE.Group();
  group.name = "goal-star";
  group.userData.environment = true;
  group.position.copy(toWorld(state.target));
  const shape = new THREE.Shape();
  for (let i = 0; i < 10; i += 1) {
    const angle = Math.PI / 2 + i * Math.PI / 5;
    const radius = i % 2 === 0 ? 1.6 : 0.72;
    const x = Math.cos(angle) * radius, y = Math.sin(angle) * radius;
    if (i === 0) shape.moveTo(x, y); else shape.lineTo(x, y);
  }
  shape.closePath();
  const geometry = new THREE.ExtrudeGeometry(shape, {
    depth: 0.35, bevelEnabled: true, bevelSize: 0.1, bevelThickness: 0.1, bevelSegments: 2, steps: 1,
  });
  geometry.center();
  const material = new THREE.MeshBasicMaterial({color: 0xffd34d, transparent: true,
    depthTest: false, depthWrite: false, toneMapped: false});
  const star = new THREE.Mesh(geometry, material);
  star.renderOrder = 7;
  group.add(star);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(2.1, 0.055, 8, 48), material.clone());
  ring.rotation.x = Math.PI / 2;
  ring.renderOrder = 6;
  group.add(ring);
  const sparks = [];
  for (let i = 0; i < 16; i += 1) {
    const spark = new THREE.Mesh(new THREE.OctahedronGeometry(0.15), material.clone());
    spark.renderOrder = 8;
    spark.visible = false;
    group.add(spark);
    sparks.push(spark);
  }
  group.userData.reward = {star, ring, sparks, collected: Boolean(state.milestones?.target_reached), age: null};
  group.visible = !group.userData.reward.collected;
  return group;
}


function updateGoalStar(delta, elapsed, active) {
  if (!goalStar) return;
  const reward = goalStar.userData.reward;
  if (state.milestones?.target_reached && !reward.collected) {
    reward.collected = true;
    reward.age = 0;
    reward.startedAt = elapsed;
    goalReward.textContent = "★ +1 · Target reached";
    goalReward.hidden = !active;
  }
  if (!reward.collected) {
    goalStar.visible = active;
    reward.star.position.y = Math.sin(elapsed * 2.2) * 0.28;
    reward.star.rotation.y = elapsed * 1.2;
    reward.ring.scale.setScalar(1 + Math.sin(elapsed * 2.2) * 0.08);
    return;
  }
  if (reward.age === null) { goalStar.visible = false; return; }
  reward.age = Math.max(0, elapsed - reward.startedAt);
  const t = Math.min(1, reward.age / 1.8);
  goalStar.visible = active && t < 1;
  goalReward.hidden = !active || t >= 1;
  reward.star.position.y = 3 * t;
  reward.star.rotation.y += delta * 7;
  reward.star.scale.setScalar(1 + Math.sin(t * Math.PI) * 0.55);
  reward.star.material.opacity = 1 - t;
  reward.ring.scale.setScalar(1 + t * 2.5);
  reward.ring.material.opacity = 1 - t;
  reward.sparks.forEach((spark, index) => {
    const angle = index * Math.PI * 2 / reward.sparks.length;
    spark.visible = t < 1;
    spark.position.set(Math.cos(angle) * t * 4, Math.sin(index * 2.4) * t * 2 + t,
      Math.sin(angle) * t * 4);
    spark.material.opacity = 1 - t;
    spark.scale.setScalar(1 - t * 0.8);
  });
}


function createEndpointBeacon(item, height) {
  const group = new THREE.Group();
  group.name = item.name + "_beacon";
  group.userData.environment = true;
  group.position.copy(toWorld(item.position));
  group.position.y += item.size.z / 2;
  // Overlay the glow so buildings cannot hide the destination in follow mode.
  const material = new THREE.MeshBasicMaterial({
    color: item.color, transparent: true, opacity: 0.14,
    side: THREE.DoubleSide, depthWrite: false, depthTest: false,
    fog: false, toneMapped: false,
  });
  const geometry = new THREE.BoxGeometry(item.size.x, height, item.size.y);
  // Keep only the four vertical faces, leaving the light column open at both ends.
  const indices = geometry.index.array;
  geometry.setIndex([...indices.slice(0, 12), ...indices.slice(24)]);
  geometry.clearGroups();
  const column = new THREE.Mesh(geometry, material);
  column.position.y = height / 2;
  column.renderOrder = 3;
  group.add(column);
  const halfWidth = item.size.x / 2;
  const halfDepth = item.size.y / 2;
  const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-halfWidth, 0.025, -halfDepth),
    new THREE.Vector3(halfWidth, 0.025, -halfDepth),
    new THREE.Vector3(halfWidth, 0.025, halfDepth),
    new THREE.Vector3(-halfWidth, 0.025, halfDepth),
  ]), new THREE.LineBasicMaterial({
    color: item.color, transparent: true, opacity: 0.8,
    depthWrite: false, depthTest: false, fog: false, toneMapped: false,
  }));
  ring.renderOrder = 4;
  group.add(ring);
  return group;
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
  // Telemetry z=0 denotes ground contact; keep the model above its contact plane.
  group.children.forEach(child => { child.position.y += 0.72; });
  group.scale.setScalar(0.72);
  return group;
}


function createBird(item) {
  const group = new THREE.Group();
  const anatomy = new THREE.Group();
  const radius = item.radius * WORLD_SCALE;
  anatomy.scale.setScalar(radius / 1.65);
  group.add(anatomy);
  const dark = new THREE.MeshStandardMaterial({ color: 0x35414a, roughness: 0.8 });
  const light = new THREE.MeshStandardMaterial({ color: 0xe4e8e5, roughness: 0.85 });
  const beakMaterial = new THREE.MeshStandardMaterial({ color: 0xd9b25c, roughness: 0.6 });
  const part = (geometry, material, position, scale) => {
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position);
    if (scale) mesh.scale.set(...scale);
    mesh.castShadow = true;
    anatomy.add(mesh);
    return mesh;
  };
  part(new THREE.SphereGeometry(0.33, 16, 12), light, [0, 0, 0], [0.7, 0.8, 1.7]);
  part(new THREE.SphereGeometry(0.22, 14, 10), dark, [0, 0.18, 0.45]);
  const beak = part(new THREE.ConeGeometry(0.09, 0.3, 6), beakMaterial, [0, 0.15, 0.73]);
  beak.rotation.x = Math.PI / 2;
  for (const side of [-1, 1]) {
    part(new THREE.SphereGeometry(0.035, 8, 6), light, [side * 0.18, 0.23, 0.53]);
    const tail = part(new THREE.BoxGeometry(0.17, 0.045, 0.48), dark, [side * 0.09, 0.02, -0.61]);
    tail.rotation.y = side * 0.18;
  }
  const wings = [-1, 1].map((side) => {
    const pivot = new THREE.Group();
    pivot.position.set(side * 0.15, 0.09, 0.02);
    const shape = new THREE.Shape();
    shape.moveTo(0, 0.22);
    shape.quadraticCurveTo(0.65, 0.35, 1.38, -0.28);
    shape.lineTo(1.12, -0.46);
    shape.lineTo(0.5, -0.3);
    shape.lineTo(0, -0.2);
    shape.closePath();
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: 0.045, bevelEnabled: false, curveSegments: 8 });
    geometry.rotateX(Math.PI / 2);
    const mesh = new THREE.Mesh(geometry, dark);
    mesh.scale.x = side;
    mesh.castShadow = true;
    pivot.add(mesh);
    anatomy.add(pivot);
    return pivot;
  });
  const safetyBubble = new THREE.Mesh(new THREE.SphereGeometry(radius, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0xf2b84b, transparent: true, opacity: 0.12, depthWrite: false }));
  safetyBubble.visible = false;
  group.add(safetyBubble);
  group.userData = { wings, anatomy, safetyBubble, id: item.id, radius, yaw: null };
  return group;
}




function syncScene(initial) {
  buildEnvironment();

  const htmlRuntimeActive = state.runtime_backend === "html";
  const uavTarget = toWorld(state.uav);
  uavGroup.userData.targetPosition = uavTarget;
  if (initial) uavGroup.position.copy(uavTarget);

  state.dynamic_objects.forEach((item) => {
    let bird = birdMeshes.get(item.id);
    if (!bird) {
      bird = createBird(item);
      scene.add(bird);
      birdMeshes.set(item.id, bird);
    }
    const target = toWorld(item.position);
    if (initial) {
      bird.position.copy(target);
    }
  });

  updatePath();
  updateWaypoints();
  setSceneActorVisibility(htmlRuntimeActive);
}


function updatePath() {
  const nextKey = JSON.stringify(state.continuous_path || state.path || []);
  if (nextKey === pathKey) return;
  pathKey = nextKey;
  if (pathLine) {
    scene.remove(pathLine);
    pathLine.geometry.dispose();
    pathLine.material.dispose();
    pathLine = null;
  }
  if (state.runtime_backend !== "html") {
    return;
  }
  const points = (state.continuous_path || state.path || []).map((point) => toWorld(point));
  if (points.length < 2) {
    return;
  }
  const curve = points.length > 2
    ? new THREE.CatmullRomCurve3(points, false, "centripetal", 0.25)
    : new THREE.LineCurve3(points[0], points[1]);
  const geometry = new THREE.TubeGeometry(
    curve,
    Math.max(12, points.length * 8),
    0.14,
    8,
    false,
  );
  const material = new THREE.MeshBasicMaterial({
    color: 0x19f3eb,
    transparent: true,
    opacity: 0.98,
    depthTest: false,
  });
  pathLine = new THREE.Mesh(geometry, material);
  pathLine.renderOrder = 5;
  scene.add(pathLine);
}


function disposeRoute() {
  if (plannedRoute) {
    plannedRoute.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    scene.remove(plannedRoute);
  }
  plannedRoute = null;
  plannedRouteKey = "";
}


function updateWaypoints() {
  updateAvoidanceRoute();
  const waypoints = state.flight?.waypoints || [];
  const planStart = state.flight?.plan_start;
  const key = JSON.stringify([state.flight?.plan_id, planStart, waypoints]);
  if (key === plannedRouteKey) return;
  disposeRoute();
  plannedRouteKey = key;
  if (!waypoints.length) return;
  plannedRoute = new THREE.Group();
  const points = waypoints.map(toWorld);
  // The origin belongs to the accepted plan, not the moving UAV or mission base.
  const routePoints = planStart ? [toWorld(planStart), ...points] : points;
  const line = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(routePoints),
    new THREE.LineDashedMaterial({ color: 0xffc857, dashSize: 0.7, gapSize: 0.4 }),
  );
  line.computeLineDistances();
  plannedRoute.add(line);
  points.forEach((point) => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0xffc857 }),
    );
    marker.position.copy(point);
    plannedRoute.add(marker);
  });
  scene.add(plannedRoute);
}


function updateAvoidanceRoute() {
  const avoidance = state.flight?.avoidance;
  if (!avoidanceRoute) {
    avoidanceRoute = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineDashedMaterial({ color: 0xff9f43, dashSize: 0.25, gapSize: 0.15, depthTest: false }),
    );
    avoidanceRoute.renderOrder = 6;
    scene.add(avoidanceRoute);
  }
  avoidanceRoute.visible = state.runtime_backend === "html" && Boolean(avoidance?.active && avoidance.target);
  if (avoidanceRoute.visible) {
    const positions = avoidanceRoute.geometry.attributes.position;
    [toWorld(state.uav), toWorld(avoidance.target)].forEach((point, index) => positions.setXYZ(index, point.x, point.y, point.z));
    positions.needsUpdate = true;
    avoidanceRoute.geometry.computeBoundingSphere();
    avoidanceRoute.computeLineDistances();
    avoidanceRoute.material.color.set(avoidance.limited ? 0xff5555 : 0xff9f43);
  }
}


function setSceneActorVisibility(htmlRuntimeActive) {
  if (plannedRoute) plannedRoute.visible = htmlRuntimeActive;
  if (avoidanceRoute && !htmlRuntimeActive) avoidanceRoute.visible = false;
  if (uavGroup) {
    uavGroup.visible = htmlRuntimeActive;
  }
  birdMeshes.forEach((bird) => {
    bird.visible = htmlRuntimeActive;
  });
  if (pathLine) {
    pathLine.visible = htmlRuntimeActive;
  }
}


function clearEnvironment() {
  goalReward.hidden = true;
  goalStar = null;
  disposeRoute();
  if (avoidanceRoute) {
    avoidanceRoute.geometry.dispose();
    avoidanceRoute.material.dispose();
    scene.remove(avoidanceRoute);
    avoidanceRoute = null;
  }
  pathKey = "";
  [...scene.children].forEach((child) => {
    if (child.userData.environment || child === uavGroup || birdMeshesHas(child) || child === pathLine) {
      child.traverse((object) => {
        object.geometry?.dispose();
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => {
          material?.map?.dispose();
          material?.dispose();
        });
      });
      scene.remove(child);
    }
  });
  birdMeshes.clear();
  cameraObstacles = [];
  followOffset.set(9, 7, 12);
  followAnchor.set(0, 0, 0);
  rotorMeshes = [];
  uavGroup = null;
  pathLine = null;
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
  renderToolbarState();
  if (!state) {
    return;
  }
  renderTelemetry();
  renderRuntimeControls();
  renderDecisionTrace(getHistoryEntries()[0] || null);
}


function renderToolbarState() {
  simulator?.setAttribute("data-operation", commandError ? "error" : state?.flight?.mode || "idle");

  const missionPhase = { fly_to_target: "Outbound", returning_home: "Returning", completed: "Completed" }[state?.phase] || "--";
  setText(phaseValue, `${missionPhase} / ${state?.flight?.stage || "--"} / ${state?.flight?.mode || "idle"}`);
  const traceDecision = getHistoryEntries()[0]?.decision;
  const traceBackend = traceDecision?.agent_backend || state?.agent_backend;
  const traceModel = traceDecision?.agent_backend
    ? traceDecision.agent_model
    : state?.agent_model;
  setText(backendValue, traceModel ? `${traceBackend}:${traceModel}` : traceBackend);
  const flight = state?.flight || {};
  const active = Boolean(flight.running || flight.planning);
  const unavailable = !state || state.status !== "active";
  planBtn.disabled = commandPending || unavailable || active || flight.mode === "braking";
  startBtn.disabled = commandPending || unavailable || flight.mode === "braking";
  resetBtn.disabled = commandPending || !state;
  scenarioSelect.disabled = commandPending || !state;
  agentButtons.forEach((button) => {
    const selected = button.dataset.agent === state?.agent_backend;
    button.setAttribute("aria-checked", String(selected));
    button.classList.toggle("active", selected);
    button.disabled = commandPending;
  });
  if (agentKeyBtn) {
    agentKeyBtn.disabled = commandPending;
  }
  startBtn.setAttribute("aria-pressed", String(active));

  const iconChanged = updateCommandButton(
    startBtn,
    active ? "Pause" : "Start",
    active ? "pause" : "play",
  );
  if (iconChanged) {
    createLucideIcons();
  }
}


function renderTelemetry() {
  if (!state) {
    return;
  }
  const uav = isRecord(state.uav) ? state.uav : {};
  setText(positionValue, formatPoint(uav));

  const altitudeScale = Number(
    state.runtime?.meters_per_altitude_level
      || state.meters_per_altitude_level
      || 5,
  );
  const altitude = uav.z === null || uav.z === undefined
    ? Number.NaN
    : Number(uav.z);
  setText(
    altitudeValue,
    Number.isFinite(altitude) && Number.isFinite(altitudeScale)
      ? `${(altitude * altitudeScale).toFixed(1)} m`
      : "--",
  );
  setText(
    batteryValue,
    state.battery === null || state.battery === undefined
      ? "--"
      : `${Math.round(state.battery)}%`,
  );
  setText(telemetryPhaseValue, state.flight?.mode);

  const observation = isRecord(state.observation) ? state.observation : {};
  const threat = isRecord(observation.nearest_dynamic_obstacle)
    ? observation.nearest_dynamic_obstacle
    : null;
  setText(
    threatValue,
    threat
      ? `${valueOrDash(threat.threat_level)} / ${formatNumber(threat.predicted_distance)}`
      : hasOwn(observation, "nearest_dynamic_obstacle") ? "none" : "--",
  );
  if (threatValue) {
    threatValue.style.color = threatColor(threat?.threat_level || "low");
  }
}


function renderDecisionTrace(entry = state?.history?.[0] || null) {
  if (!state) {
    return;
  }
  const entries = getHistoryEntries();
  const latestEntry = entries[0] || null;
  const isLatestEntry = Boolean(
    entry
      && latestEntry
      && String(entry.step) === String(latestEntry.step),
  );
  const structuredEntryObservation = isRecord(entry?.observation)
    ? entry.observation
    : null;
  const observation = ((!entry || isLatestEntry) && isRecord(state.observation)
    ? state.observation : structuredEntryObservation) || {};
  const hasObservation = Boolean(
    entry?.observation || Object.keys(observation).length,
  );
  setTraceStageStatus(
    "observation",
    hasObservation ? "active" : "waiting",
    hasObservation ? "OBSERVED" : "WAITING",
  );

  setText(
    currentStepValue,
    `#${state.flight?.plan_id ?? 0}`,
  );
  setText(goalDirectionValue, observation.goal_direction);
  setText(goalDistanceValue, formatNumber(observation.goal_distance));
  setText(
    staticRiskValue,
    formatStaticRisk(
      observation.nearest_static_obstacle,
      hasOwn(observation, "nearest_static_obstacle"),
    ),
  );
  setText(
    dynamicRiskValue,
    formatDynamicRisk(
      observation.nearest_dynamic_obstacle,
      hasOwn(observation, "nearest_dynamic_obstacle"),
    ),
  );

  const decision = isRecord(entry?.decision) ? entry.decision : {};
  const waypoints = Array.isArray(decision.waypoints) ? decision.waypoints : [];
  const flight = state.flight || {};
  const avoidance = flight.avoidance || {};
  setText(agentActionValue, waypoints.length ? waypoints.map(formatPoint).join(" -> ") : "--");
  setText(document.getElementById("plannerProgress"), `${flight.planner_calls ?? 0} planner calls | ${avoidance.replans ?? 0} local replans | ${Math.min(flight.waypoint_index ?? 0, flight.waypoints?.length ?? 0)}/${flight.waypoints?.length ?? 0} reached`);
  setText(agentRiskValue, decision.risk);
  setText(agentReasonValue, decision.reason);

  const safetyIntervention = decision.safety_intervention;
  const hasDecision = Object.keys(decision).length > 0;
  setTraceStageStatus(
    "agent",
    flight.planning ? "active" : decision.provider_error ? "error" : hasDecision ? "pass" : "waiting",
    flight.planning ? "PLANNING" : decision.provider_error ? "API ERROR" : hasDecision ? "PLANNED" : "WAITING",
  );
  const guardOverrode = Boolean(
    avoidance.active || decision.safety_intervened || safetyIntervention,
  );
  setText(proposedActionValue, decision.expected_result);
  setText(executedActionValue, formatPoint(avoidance.active ? avoidance.target : flight.waypoints?.[flight.waypoint_index]));
  const guardLabel = avoidance.active ? "LOCAL AVOIDANCE" : decision.provider_error ? "API ERROR" : hasDecision ? (guardOverrode ? "OVERRIDE" : "PASS") : "WAITING";
  setText(guardStatus, guardLabel);
  const guardContainer = guardStatus?.closest(".decision-guard");
  guardContainer?.setAttribute(
    "data-status",
    decision.provider_error ? "error" : guardOverrode ? "override" : hasDecision ? "pass" : "waiting",
  );
  guardStatus?.setAttribute(
    "data-status",
    decision.provider_error ? "error" : guardOverrode ? "override" : hasDecision ? "pass" : "waiting",
  );

  setTraceStageStatus(
    "execution",
    avoidance.active ? (avoidance.limited ? "error" : "active") : flight.error ? "error" : flight.running || flight.planning ? "active" : "pass",
    String(flight.mode || "idle").toUpperCase(),
  );
  setText(resultActionValue, `${flight.mode || "idle"} / ${formatNumber(flight.speed_mps)} m/s`);
  setText(executionPositionValue, formatPoint(state.uav));
  const milestones = state.milestones || {};
  setText(missionProgressText, `${milestones.target_reached ? "★ Collected · Target reached" : "☆ Target pending"} · ${milestones.base_returned ? "Returned to base" : "Return pending"}`);
  const recovery = flight.recovery || {};
  const recoveryLabels = {
    refresh_observation_and_retry_provider: "Waiting for API recovery",
    repair_response: "Requesting a valid response",
    repair_rejected_segments: "Repairing rejected route",
    choose_different_corridor: "Requesting a different corridor",
    refresh_observation_and_change_route: "Replanning after stalled progress",
    track_accepted_route: "Following accepted route",
  };
  setText(recoveryText, `${recoveryLabels[recovery.next_action] || "No retry pending"}${flight.retry_in_seconds > 0 ? ` · retry in ${flight.retry_in_seconds.toFixed(1)}s` : ""}`);
  setText(feedbackText, commandError || (avoidance.active
    ? [avoidance.message, flight.error].filter(Boolean).join(" ")
    : flight.error || state.feedback || entry?.feedback));

}


function renderRuntimeControls() {
  if (!state) {
    return;
  }
  const runtime = state.runtime || {};
  const connected = Boolean(runtime.connected);
  setText(runtimeStatus, commandPending
    ? `${state.runtime_backend} / pending`
    : `${state.runtime_backend} / ${connected ? "connected" : "offline"}`);
  runtimeStatus.setAttribute("aria-busy", String(commandPending));
  runtimeStatus.setAttribute("data-connected", String(connected));
  runtimeStatus.style.color = connected ? "#62d57b" : "#ff6b57";
  runtimeButtons.forEach((button) => {
    const selected = button.dataset.runtime === state.runtime_backend;
    button.setAttribute("aria-checked", String(selected));
    button.classList.toggle("active", selected);
    button.disabled = commandPending;
  });
  if (airsimConfigBtn) {
    airsimConfigBtn.disabled = commandPending;
  }

  scenarioSelect.value = state.scenario_id || "";
  if (runtime.host !== null && runtime.host !== undefined) {
    setInputUnlessFocused(airsimHost, runtime.host);
  }
  if (runtime.port !== null && runtime.port !== undefined) {
    setInputUnlessFocused(airsimPort, runtime.port);
  }
  if (runtime.vehicle_name !== null && runtime.vehicle_name !== undefined) {
    setInputUnlessFocused(vehicleName, runtime.vehicle_name);
  }
  const wind = isRecord(runtime.wind_ned) ? runtime.wind_ned : {};
  setInputUnlessFocused(windNorth, wind.x ?? 0);
  setInputUnlessFocused(windEast, wind.y ?? 0);
  setInputUnlessFocused(windDown, wind.z ?? 0);

  reconnectAirSimBtn.disabled = commandPending;
  applyWindBtn.disabled = commandPending
    || !connected
    || !Boolean(runtime.supports_wind);
}


function openDialog(dialog) {
  if (!dialog) {
    return;
  }
  if (typeof dialog.showModal === "function") {
    if (!dialog.open) {
      dialog.showModal();
    }
  } else {
    dialog.setAttribute("open", "");
  }
}


function closeDialog(dialog) {
  if (!dialog) {
    return;
  }
  if (typeof dialog.close === "function") {
    if (dialog.open) {
      dialog.close();
    }
  } else {
    dialog.removeAttribute("open");
  }
}


function openAgentDialog() {
  if (commandPending) {
    return;
  }
  if (deepseekApiKey) {
    deepseekApiKey.value = "";
  }
  if (deepseekModel && state?.agent_model) {
    deepseekModel.value = String(state.agent_model);
  }
  if (agentDialogError) {
    agentDialogError.hidden = true;
    agentDialogError.textContent = "";
  }
  openDialog(agentDialog);
  window.setTimeout(() => deepseekApiKey?.focus(), 0);
}


function openAirSimDialog() {
  if (commandPending) {
    return;
  }
  renderRuntimeControls();
  openDialog(airsimDialog);
}


function setInputUnlessFocused(input, value) {
  if (input && document.activeElement !== input) {
    input.value = String(value);
  }
}


function setText(element, value) {
  if (element) {
    element.textContent = valueOrDash(value);
  }
}


function setTraceStageStatus(stageName, status, label) {
  const stage = document.querySelector(`[data-stage="${stageName}"]`);
  if (!stage) {
    return;
  }
  stage.setAttribute("data-status", status);
  setText(stage.querySelector(".stage-status"), label);
}


function valueOrDash(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  return String(value);
}


function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}


function hasOwn(object, key) {
  return isRecord(object) && Object.prototype.hasOwnProperty.call(object, key);
}


function formatPoint(point) {
  if (Array.isArray(point)) {
    if (point.length < 3) {
      return "--";
    }
    return `(${formatNumber(point[0])}, ${formatNumber(point[1])}, ${formatNumber(point[2])})`;
  }
  if (!isRecord(point)) {
    return "--";
  }
  if (!["x", "y", "z"].some((axis) => hasOwn(point, axis))) {
    return "--";
  }
  return `(${formatNumber(point.x)}, ${formatNumber(point.y)}, ${formatNumber(point.z)})`;
}


function formatStaticRisk(obstacle, fieldPresent) {
  if (!fieldPresent) {
    return "--";
  }
  if (!obstacle) {
    return "clear";
  }
  if (!isRecord(obstacle)) {
    return valueOrDash(obstacle);
  }
  const parts = [];
  if (obstacle.direction) {
    parts.push(String(obstacle.direction));
  }
  if (obstacle.distance !== null && obstacle.distance !== undefined) {
    parts.push(`distance ${formatNumber(obstacle.distance)}`);
  }
  if (obstacle.height !== null && obstacle.height !== undefined) {
    parts.push(`height ${formatNumber(obstacle.height)}`);
  }
  return parts.length ? parts.join(" | ") : "nearby";
}


function formatDynamicRisk(obstacle, fieldPresent) {
  if (!fieldPresent) {
    return "--";
  }
  if (!obstacle) {
    return "clear";
  }
  if (!isRecord(obstacle)) {
    return valueOrDash(obstacle);
  }
  const parts = [];
  if (obstacle.threat_level) {
    parts.push(String(obstacle.threat_level));
  }
  if (obstacle.predicted_distance !== null && obstacle.predicted_distance !== undefined) {
    parts.push(`predicted ${formatNumber(obstacle.predicted_distance)}`);
  }
  if (obstacle.id) {
    parts.push(String(obstacle.id));
  }
  return parts.length ? parts.join(" | ") : "nearby";
}


function getHistoryEntries() {
  return Array.isArray(state?.history) ? state.history : [];
}


function updateCommandButton(button, label, iconName) {
  if (!button) {
    return false;
  }
  const labelElement = button.querySelector("span");
  if (labelElement) {
    labelElement.textContent = label;
  } else {
    const textNode = [...button.childNodes].find(
      (node) => node.nodeType === 3 && node.nodeValue.trim(),
    );
    if (textNode) {
      textNode.nodeValue = label;
    } else {
      const span = document.createElement("span");
      span.textContent = label;
      button.appendChild(span);
    }
  }
  button.setAttribute("aria-label", label);
  button.setAttribute("title", label);

  const icon = button.querySelector("[data-lucide]");
  if (!icon || icon.getAttribute("data-lucide") === iconName) {
    return false;
  }
  icon.setAttribute("data-lucide", iconName);
  return true;
}


function createLucideIcons() {
  if (window.lucide?.createIcons) {
    window.lucide.createIcons();
  }
}


function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "--";
}


function formatVector(vector) {
  const value = isRecord(vector) ? vector : {};
  return `(${formatNumber(value.x)}, ${formatNumber(value.y)}, ${formatNumber(value.z)})`;
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


function visibleFollowDistance(target, offset) {
  const distance = offset.length();
  if (distance < 1e-6) return 0;
  const ray = new THREE.Ray(target, offset.clone().divideScalar(distance));
  const hit = new THREE.Vector3();
  let visible = distance;
  for (const obstacle of cameraObstacles) {
    if (ray.intersectBox(obstacle, hit)) {
      visible = Math.min(visible, hit.distanceTo(target));
    }
  }
  return visible;
}


function chooseFollowOffset(target) {
  // Retain even a shortened safe boom; do not reselect tied directions every frame.
  if (visibleFollowDistance(target, followOffset) >= followOffset.length() - 0.4) {
    return followOffset;
  }
  const base = new THREE.Vector3(9, 7, 12);
  const axis = new THREE.Vector3(0, 1, 0);
  let best = null;
  let bestScore = Infinity;
  let fallback = null;
  let fallbackDistance = 0;
  for (const height of [7, 12, 20, 36]) {
    const offsets = [new THREE.Vector3(0, height, 15), new THREE.Vector3(0, height, -15),
      new THREE.Vector3(15, height, 0), new THREE.Vector3(-15, height, 0)];
    for (const angle of [0, Math.PI / 4, -Math.PI / 4, Math.PI / 2,
      -Math.PI / 2, 3 * Math.PI / 4, -3 * Math.PI / 4, Math.PI]) {
      offsets.push(new THREE.Vector3(base.x, height, base.z).applyAxisAngle(axis, angle));
    }
    for (const offset of offsets) {
      const visible = visibleFollowDistance(target, offset);
      if (visible > fallbackDistance) {
        fallbackDistance = visible;
        fallback = offset.clone().multiplyScalar(Math.max(0.1, visible - 1.2) / offset.length());
      }
      if (visible < offset.length() - 0.4) continue;
      const score = offset.distanceToSquared(followOffset) + 0.15 * offset.distanceToSquared(base);
      if (score < bestScore) {
        bestScore = score;
        best = offset;
      }
    }
  }
  followOffset = best || fallback || base;
  return followOffset;
}

function updateFollowCamera(delta) {
  const target = uavGroup.position;
  // Transport the rig with its subject before damping the boom, avoiding polling lag.
  camera.position.add(target.clone().sub(followAnchor));
  followAnchor.copy(target);
  const offset = camera.position.clone().sub(target);
  offset.lerp(chooseFollowOffset(target), 1 - Math.exp(-3.2 * delta));
  const length = offset.length();
  const visible = visibleFollowDistance(target, offset);
  // Retract along the current sightline instead of teleporting to another angle.
  if (visible < length && length > 1e-6) {
    offset.multiplyScalar(Math.max(0.1, visible - 1.2) / length);
  }
  camera.position.copy(target).add(offset);
  camera.lookAt(target);
}


function setCameraMode(mode) {
  cameraMode = mode;
  overviewBtn.classList.toggle("active", mode === "overview");
  followBtn.classList.toggle("active", mode === "follow");
  controls.enabled = mode === "overview";
  if (mode === "overview") {
    const mapScale = Math.max(1, (state?.grid_size || 20) / 20);
    const urban = Boolean(state?.scenario?.urban_parcels?.length);
    camera.position.set((urban ? 50 : 55) * mapScale,
      (urban ? 42 : 60) * mapScale, (urban ? 56 : 65) * mapScale);
    controls.target.set(0, urban ? 24 : 17, 0);
  } else if (uavGroup) {
    followOffset.set(9, 7, 12);
    followAnchor.copy(uavGroup.position);
    camera.position.copy(uavGroup.position).add(chooseFollowOffset(uavGroup.position));
    camera.lookAt(uavGroup.position);
  }
}


function animate() {
  timer.update();
  const delta = Math.min(timer.getDelta(), 0.05);
  const elapsed = timer.getElapsed();
  const htmlRuntimeActive = state?.runtime_backend === "html";

  if (uavGroup && htmlRuntimeActive) {
    const grounded = state.flight?.stage === "landed";
    if (uavGroup.userData.targetPosition) {
      const target = uavGroup.userData.targetPosition;
      const direction = target.clone().sub(uavGroup.position);
      uavGroup.position.lerp(target, 1 - Math.exp(-20 * delta));
      if (Math.abs(direction.x) + Math.abs(direction.z) > 0.001) {
        uavGroup.rotation.y = Math.atan2(direction.x, direction.z);
      }
      uavGroup.rotation.z = grounded ? 0 : Math.sin(elapsed * 2.2) * 0.018;
    }
    rotorMeshes.forEach((rotor) => {
      if (!grounded || state.flight?.running) rotor.rotation.y += delta * 42 * rotor.userData.speed;
    });
  }

  if (htmlRuntimeActive) {
    birdMeshes.forEach((bird, id) => {
      const model = trafficMotion?.models[id];
      if (model) {
        const seconds = trafficMotion.elapsed; // Shared backend simulation clock; no wall-time extrapolation.
        const { position, velocity } = sampleTraffic(model, seconds);
        bird.position.lerp(toWorld({ x: position[0], y: position[1], z: position[2] }), 1 - Math.exp(-20 * delta));
        const speed = Math.hypot(velocity[0] * WORLD_SCALE, velocity[1] * WORLD_SCALE, velocity[2] * ALTITUDE_SCALE);
        if (speed > 0.02) {
          const yaw = Math.atan2(velocity[0], velocity[1]);
          const change = bird.userData.yaw === null ? 0 : Math.atan2(Math.sin(yaw - bird.userData.yaw), Math.cos(yaw - bird.userData.yaw));
          bird.rotation.y = yaw;
          bird.userData.anatomy.rotation.z = THREE.MathUtils.lerp(bird.userData.anatomy.rotation.z,
            THREE.MathUtils.clamp(-change / Math.max(delta, 0.001) * 0.3, -0.45, 0.45), 1 - Math.exp(-5 * delta));
          bird.userData.yaw = yaw;
          bird.userData.anatomy.rotation.x = -Math.atan2(velocity[2] * ALTITUDE_SCALE,
            Math.hypot(velocity[0] * WORLD_SCALE, velocity[1] * WORLD_SCALE));
        }
        const glide = Math.sin(seconds * 0.65 + model.phase) > 0.5 && speed > 0.8;
        bird.userData.wings.forEach((wing, index) => {
          wing.rotation.z = (index === 0 ? -1 : 1) * (glide ? 0.08 : Math.sin(seconds * 9 + model.phase) * 0.45);
        });
      }
      const separation = bird.position.distanceTo(toWorld(state.uav));
      const high = separation < bird.userData.radius + 1;
      const near = separation < bird.userData.radius + 4;
      bird.userData.safetyBubble.visible = near || hoveredBird === bird;
      bird.userData.safetyBubble.material.color.set(high ? 0xff6b57 : near ? 0xf2b84b : 0x74d8df);
    });
  }
  updateGoalStar(delta, elapsed, htmlRuntimeActive);
  updateBirdHover(htmlRuntimeActive);

  if (cameraMode === "follow" && uavGroup) {
    updateFollowCamera(delta);
  } else {
    controls.update();
  }

  renderer.render(scene, camera);
}

function updateBirdHover(active) {
  hoveredBird = null;
  if (active && Math.abs(birdPointer.x) <= 1 && Math.abs(birdPointer.y) <= 1) {
    birdRaycaster.setFromCamera(birdPointer, camera);
    const hit = birdRaycaster.intersectObjects(scene.children, true)
      .find((entry) => entry.object.visible && ![...birdMeshes.values()].some((bird) => entry.object === bird.userData.safetyBubble));
    let object = hit?.object;
    while (object) {
      if (object.userData.id && birdMeshes.get(object.userData.id) === object) {
        hoveredBird = object;
        break;
      }
      object = object.parent;
    }
  }
  birdTooltip.hidden = !hoveredBird;
  if (hoveredBird) {
    birdTooltip.textContent = `${hoveredBird.userData.id} · ${hoveredBird.position.y.toFixed(1)} m`;
    const width = sceneContainer.clientWidth;
    const height = sceneContainer.clientHeight;
    birdTooltip.style.left = `${Math.min(width - 150, Math.max(8, (birdPointer.x + 1) * width / 2 + 12))}px`;
    birdTooltip.style.top = `${Math.min(height - 36, Math.max(8, (1 - birdPointer.y) * height / 2 - 28))}px`;
  }
}


function onResize() {
  const width = sceneContainer.clientWidth;
  const height = sceneContainer.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}


planBtn.addEventListener("click", planMission);
startBtn.addEventListener("click", toggleMission);
resetBtn.addEventListener("click", resetSimulation);
scenarioSelect.addEventListener("change", () => switchScenario(scenarioSelect.value));
overviewBtn.addEventListener("click", () => setCameraMode("overview"));
followBtn.addEventListener("click", () => setCameraMode("follow"));
agentButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const backend = button.dataset.agent;
    if (backend === "deepseek" && !state?.deepseek_key_configured) {
      openAgentDialog();
      return;
    }
    void switchAgent(backend);
  });
});
runtimeButtons.forEach((button) => {
  button.addEventListener("click", () => switchRuntime(button.dataset.runtime));
});
agentKeyBtn?.addEventListener("click", openAgentDialog);
airsimConfigBtn?.addEventListener("click", openAirSimDialog);
reconnectAirSimBtn.addEventListener("click", () => switchRuntime("airsim", true));
applyWindBtn.addEventListener("click", applyWind);
agentForm?.addEventListener("submit", configureDeepSeekFromDialog);
closeAgentBtn?.addEventListener("click", () => closeDialog(agentDialog));
cancelAgentBtn?.addEventListener("click", () => closeDialog(agentDialog));
closeAirSimBtn?.addEventListener("click", () => closeDialog(airsimDialog));
closeAirSimFooterBtn?.addEventListener("click", () => closeDialog(airsimDialog));
window.addEventListener("resize", onResize);

createLucideIcons();
renderToolbarState();
renderer.setAnimationLoop(animate);
void pollState();
void loadScenarios().catch(() => setText(feedbackText, "Scenario list unavailable"));
