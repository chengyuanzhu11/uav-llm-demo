import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { buildUrbanCity, urbanSurfaceTexture } from "./urban_city.js";
import { buildSemanticCampus } from "./semantic_campus.js";
import { MotionTimeline } from "./render_motion.js";
import { FlightTrail } from "./flight_trail.js";
import { FollowCameraRig } from "./follow_camera.js";
import { AvoidanceView } from "./avoidance_view.js";


const sceneContainer = document.getElementById("sceneContainer");
const planBtn = document.getElementById("planBtn");
const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const avoidanceStatus = document.getElementById("avoidanceStatus");
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
const semanticPanel = document.getElementById("semanticPanel");
const taskForm = document.getElementById("semanticTaskForm");
const taskInstruction = document.getElementById("taskInstruction");
const submitTaskBtn = document.getElementById("submitTaskBtn");
const recommendedTasks = document.getElementById("recommendedTasks");
const semanticProgram = document.getElementById("semanticProgram");
const semanticSteps = document.getElementById("semanticSteps");
const flightDiagnostics = document.getElementById("flightDiagnostics");

const simulator = document.getElementById("simulator")
  || document.querySelector(".simulator");
const agentDialog = document.getElementById("agentDialog");
const agentForm = document.getElementById("agentForm");
const closeAgentBtn = document.getElementById("closeAgentBtn");
const cancelAgentBtn = document.getElementById("cancelAgentBtn");
const agentProvider = document.getElementById("agentProvider");
const deepseekApiKey = document.getElementById("deepseekApiKey");
const deepseekModel = document.getElementById("deepseekModel");
const deepseekBaseUrl = document.getElementById("deepseekBaseUrl");
const saveAgentBtn = document.getElementById("saveAgentBtn");
const agentDialogError = document.getElementById("agentDialogError");
const agentKeyStatus = document.getElementById("agentKeyStatus");
const agentDialogStatus = document.getElementById("agentDialogStatus");
const agentAdvanced = document.getElementById("agentAdvanced");
const loadModelsBtn = document.getElementById("loadModelsBtn");
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
let plannedTrail = null;
let plannedRouteKey = "";
let avoidanceView = null;
let displayedAvoidance = null;
const motionTimeline = new MotionTimeline();
const followRig = new FollowCameraRig(THREE);
let flightTrail = null;
let trailHead = null;
let cameraMode = "overview";
let cameraObstacles = [];
let campusPlaceLabels = null;
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
birdTooltip.style.whiteSpace = "pre-line";
birdTooltip.style.maxWidth = "270px";
sceneContainer.appendChild(birdTooltip);
let rotorMeshes = [];
let scenariosLoaded = false;
let renderedMissionMode = null;
let recommendationKey = "";
let semanticStepsKey = "";
let agentConfigCatalog = null;
let agentConfigEpoch = 0;
let agentModelRequest = 0;
let agentKeyRevision = 0;
let agentModelsContext = "";
let agentConfigLoading = false;
let agentModelsLoading = false;
let agentConfigSaving = false;

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
    const error = new Error(payload.error || `Request failed: ${response.status}`);
    error.status = response.status;
    error.code = payload.error_code;
    throw error;
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
    || (payload.scenario?.mission_mode !== "semantic"
      && (["x", "y", "z"].some((axis) => payload.target?.[axis] !== state.target?.[axis])
        || (payload.flight?.plan_id === 0 && state.flight?.plan_id !== 0)))
    || (payload.continuous_path || payload.path || []).length
      < (state.continuous_path || state.path || []).length;
  const changedRun = state && (payload.diagnostics?.run_id !== state.diagnostics?.run_id
    || payload.scenario_id !== state.scenario_id);
  lastSequence = payload.sequence;
  state = payload;
  if (changedRun) commandError = "";
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
async function runCommand(action, { semanticValidation = false } = {}) {
  if (commandPending || !state) return false;
  commandPending = true;
  commandError = "";
  renderUi();
  try {
    await action();
    return true;
  } catch (error) {
    // Only the task endpoint's curated validation messages are safe to display.
    // Provider and transport failures never expose their raw exception text.
    commandError = semanticValidation && error.status === 400
      ? error.message
      : isSemanticScenario() ? "请求未成功，请检查连接和 API 配置。" : "Request failed. Check the connection and configuration.";
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
    (isSemanticScenario() ? state.flight?.running : state.flight?.running || state.flight?.planning)
      ? "/api/runtime/stop" : "/api/mission/start",
  ));
}


async function planMission() {
  if (isSemanticScenario()) return false;
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
  if (state) scenarioSelect.value = state.scenario_id;
}


async function resetSimulation() {
  return runCommand(() => postState("/api/reset"));
}


async function switchScenario(scenarioId) {
  if (scenarioId === state?.scenario_id) return;
  return runCommand(() => postState("/api/scenario", { scenario_id: scenarioId }));
}


async function switchAgent(backend) {
  if (!state || commandPending) return false;
  const normalized = String(backend || "").trim().toLowerCase();
  if (!["mock", "deepseek"].includes(normalized)) return false;
  if (isSemanticScenario() && normalized !== "deepseek") return false;
  if (normalized === "deepseek") {
    openAgentDialog();
    return false;
  }
  if (normalized === state.agent_backend) return true;
  return runCommand(() => postState("/api/config", { backend: normalized }));
}


async function configureDeepSeekFromDialog(event) {
  event?.preventDefault();
  if (commandPending || agentConfigLoading || agentModelsLoading || agentConfigSaving) return;
  if (!deepseekModel.value || agentModelsContext !== agentConfigContext()) {
    showAgentConfigError("请先读取可用模型，再选择模型保存。");
    return;
  }
  const epoch = agentConfigEpoch;
  const config = {
    backend: "deepseek",
    provider: agentProvider.value,
    model: deepseekModel.value,
    base_url: deepseekBaseUrl.value.trim(),
    verify: true,
  };
  if (deepseekApiKey.value.trim()) config.api_key = deepseekApiKey.value.trim();
  let configurationError = "";
  agentConfigSaving = true;
  showAgentConfigError("");
  setText(agentDialogStatus, "正在验证配置…");
  refreshAgentConfigControls();
  const saved = await runCommand(async () => {
    try {
      await postState("/api/config", config);
    } catch (error) {
      configurationError = agentConfigurationError(error);
      throw error;
    }
  });
  if (epoch !== agentConfigEpoch || !agentDialog.open) return;
  agentConfigSaving = false;
  if (saved) {
    closeAgentDialog();
  } else {
    agentDialogStatus.textContent = "";
    showAgentConfigError(configurationError || "配置未保存，请重试。");
    refreshAgentConfigControls();
  }
}


function sleep(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}


async function switchRuntime(backend, force = false) {
  if (isSemanticScenario() && backend !== "html") return false;
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


function setupAtmosphere(campus = false) {
  const canvas = document.createElement("canvas");
  canvas.width = 2048;
  canvas.height = 1024;
  const ctx = canvas.getContext("2d");
  const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
  sky.addColorStop(0, campus ? "#7eaac7" : "#738da3");
  sky.addColorStop(0.3, campus ? "#b3d1e2" : "#a9b8c3");
  sky.addColorStop(0.49, campus ? "#d0dee4" : "#b9c4cf");
  sky.addColorStop(0.56, campus ? "#c4cfc7" : "#a8aaa3");
  sky.addColorStop(1, campus ? "#9bab98" : "#6c746e");
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
  scene.background?.dispose();
  scene.background = texture;
  scene.environment = texture;
  scene.environmentIntensity = campus ? 0.5 : 0.35;
}


function buildEnvironment() {
  if (environmentReady) {
    return;
  }

  const semantic = isSemanticScenario();
  setupAtmosphere(semantic);
  const sunlight = scene.children.find(object => object.isDirectionalLight);
  sunlight.color.set(semantic ? 0xf5f5ed : 0xffdfae);
  sunlight.intensity = semantic ? 2.8 : 3.2;
  const skylight = scene.children.find(object => object.isHemisphereLight);
  skylight.intensity = semantic ? 1.2 : 0.85;
  skylight.groundColor.set(semantic ? 0x68776a : 0x4d4942);
  renderer.toneMappingExposure = semantic ? 1.05 : 0.95;
  const urbanParcels = state.scenario?.urban_parcels || [];
  const mapScale = Math.max(1, state.grid_size / 20);
  controls.maxDistance = 115 * mapScale;
  camera.far = Math.max(400, state.grid_size * WORLD_SCALE * 4);
  camera.updateProjectionMatrix();
  scene.fog.density = 0.002 / mapScale;
  scene.fog.color.set(semantic ? 0xcddce0 : 0xbfc5c6);
  if (displayedGridSize !== state.grid_size) {
    displayedGridSize = state.grid_size;
    if (cameraMode === "overview") setCameraMode("overview");
    const sun = scene.children.find((object) => object.isDirectionalLight);
    // Keep the whole campus in front of the shadow camera on larger maps.
    const lightScale = semantic ? Math.max(1, mapScale / 2) : 1;
    sun.position.set(-90 * lightScale, 110 * lightScale, 60 * lightScale);
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
    if ((semantic || urbanParcels.length) && item.name.startsWith("building_")) return;
    // Campus architecture and landscape are drawn from the shared campus layout.
    if (semantic && item.name !== "ground" && item.name !== "base_pad") return;
    let map = null;
    if (urbanParcels.length && ((!semantic && item.name === "ground") || item.name.startsWith("road_")
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
    if (!semantic && (item.name === "base_pad" || item.name === "target_pad")) {
      scene.add(createEndpointBeacon(item, beaconHeight));
    }
  });
  if (!semantic && urbanParcels.length) {
    const roads = state.scene_layout.filter(item => item.name.startsWith("road_"));
    scene.add(buildUrbanCity(state.scenario, toWorld, WORLD_SCALE, ALTITUDE_SCALE, roads));
  }
  if (semantic) {
    const campus = new THREE.Group();
    campus.name = "semantic-campus";
    campus.userData.environment = true;
    buildSemanticCampus(THREE, campus, state.scenario);
    campusPlaceLabels = campus.userData.placeLabels;
    scene.add(campus);
  }
  scene.updateMatrixWorld(true);
  scene.traverse((object) => {
    if (object.userData.cameraOccluder) {
      cameraObstacles.push(new THREE.Box3().setFromObject(object).expandByScalar(0.15));
    }
  });

  if (!semantic) {
    goalStar = createGoalStar();
    scene.add(goalStar);
  }

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


function updateGoalStar(delta, elapsed, active, displayTime = Infinity) {
  if (!goalStar) return;
  const reward = goalStar.userData.reward;
  const reachedAt = state.milestones?.target_reached_at;
  if (state.milestones?.target_reached && !reward.collected
      && (!Number.isFinite(reachedAt) || displayTime >= reachedAt)) {
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
  const safetyBubble = new THREE.Mesh(new THREE.SphereGeometry(1, 20, 12),
    new THREE.MeshBasicMaterial({ color: 0xf2b84b, transparent: true, opacity: 0.12, depthWrite: false }));
  safetyBubble.visible = false;
  const contactBubble = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 10),
    new THREE.MeshBasicMaterial({ color: 0xff6b57, wireframe: true, transparent: true, opacity: 0.2, depthWrite: false }));
  contactBubble.visible = false;
  group.add(safetyBubble, contactBubble);
  group.userData = { wings, anatomy, safetyBubble, contactBubble,
    highlightMaterials: [dark, light, beakMaterial], id: item.id, yaw: null };
  updateBirdEnvelope(group, item);
  return group;
}


function updateBirdEnvelope(bird, item) {
  // Keep the visible anatomy readable for stored pre-metric replays. Safety
  // thresholds are accepted only from the current API, never guessed in the UI.
  const radius = Number.isFinite(item.radius_m) ? item.radius_m
    : Number.isFinite(item.radius) ? item.radius * WORLD_SCALE : 0;
  bird.userData.anatomy.scale.setScalar(Math.max(0, radius) / 1.65);
  for (const name of ["radius_m", "contact_distance_m", "separation_distance_m", "near_miss_distance_m"]) {
    bird.userData[name] = Number.isFinite(item[name]) && item[name] >= 0 ? item[name] : null;
  }
  bird.userData.contactBubble.scale.setScalar(bird.userData.contact_distance_m || 0);
  bird.userData.safetyBubble.scale.setScalar(bird.userData.separation_distance_m || 0);
}


function currentDynamicRiskLevel(distance, envelope) {
  if (!Number.isFinite(distance) || !Number.isFinite(envelope?.contact_distance_m)
      || !Number.isFinite(envelope?.separation_distance_m)) return "unknown";
  if (distance < envelope.contact_distance_m) return "critical";
  if (distance < envelope.separation_distance_m) return "high";
  if (Number.isFinite(envelope.near_miss_distance_m) && distance < envelope.near_miss_distance_m) return "medium";
  return "low";
}


function dynamicDistanceM(obstacle) {
  if (Number.isFinite(obstacle?.distance_m)) return obstacle.distance_m;
  return obstacle?.distance_unit === "m" && Number.isFinite(obstacle.distance) ? obstacle.distance : null;
}


function syncScene(initial) {
  buildEnvironment();

  const htmlRuntimeActive = state.runtime_backend === "html";
  if (initial) motionTimeline.reset();
  motionTimeline.ingest(state.motion_samples, performance.now() / 1000);
  if (initial) uavGroup.position.copy(toWorld(state.uav));

  state.dynamic_objects.forEach((item) => {
    let bird = birdMeshes.get(item.id);
    if (!bird) {
      bird = createBird(item);
      scene.add(bird);
      birdMeshes.set(item.id, bird);
    }
    updateBirdEnvelope(bird, item);
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
  if (state.runtime_backend !== "html") return;
  if (!flightTrail) {
    flightTrail = new FlightTrail(THREE, toWorld);
    pathLine = flightTrail.object;
    scene.add(pathLine);
    trailHead = new THREE.Line(new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(), new THREE.Vector3(),
    ]), new THREE.LineBasicMaterial({color: 0x19f3eb, depthTest: false, depthWrite: false}));
    trailHead.renderOrder = 5;
    scene.add(trailHead);
  }
  flightTrail.update(state.continuous_path || state.path || [], state.path_start_index);
}


function disposeRoute() {
  plannedTrail?.dispose();
  plannedTrail = null;
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
  const continuousMotion = Boolean(state.semantic_mission?.motion_progress);
  const key = JSON.stringify([state.flight?.plan_id, planStart, waypoints, continuousMotion]);
  if (key === plannedRouteKey) return;
  disposeRoute();
  plannedRouteKey = key;
  if (!waypoints.length) return;
  plannedRoute = new THREE.Group();
  plannedRoute.name = "planned_route";
  // Reuse the bounded tube renderer: WebGL line width stays one pixel on many
  // devices. Planned and flown paths own separate geometry and materials.
  plannedTrail = new FlightTrail(THREE, toWorld);
  plannedTrail.radius = 0.095;
  plannedTrail.material.color.set(0xffc857);
  plannedTrail.material.opacity = 0.92;
  plannedTrail.object.name = "planned_route_path";
  // The origin belongs to the accepted plan, not the moving UAV or mission base.
  plannedTrail.update(planStart ? [planStart, ...waypoints] : waypoints);
  plannedTrail.object.traverse(object => {
    if (object.isMesh) {
      object.renderOrder = 4;
      object.name = "planned_route_segment";
      object.raycast = () => {};
    }
  });
  plannedRoute.add(plannedTrail.object);
  // Curve samples describe a continuous path; only endpoints need markers.
  const markerPoints = continuousMotion ? [waypoints[0], waypoints[waypoints.length - 1]] : waypoints;
  markerPoints.map(toWorld).forEach(point => {
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 10, 8),
      new THREE.MeshBasicMaterial({ color: 0xffc857, depthTest: false, depthWrite: false }),
    );
    marker.name = "planned_waypoint";
    marker.raycast = () => {};
    marker.renderOrder = 4;
    marker.position.copy(point);
    plannedRoute.add(marker);
  });
  scene.add(plannedRoute);
}


function updateAvoidanceRoute() {
  if (!avoidanceView) {
    avoidanceView = new AvoidanceView(THREE, toWorld);
    scene.add(avoidanceView.object);
  }
  avoidanceView.ingest(state.flight?.avoidance?.visualization);
}


function renderAvoidanceStatus(sample) {
  if (!avoidanceStatus) return;
  avoidanceStatus.hidden = !sample;
  if (!sample) return;
  const dynamic = Boolean(sample.threats?.length || sample.episode?.object_ids?.length);
  const status = sample.active ? (dynamic ? "动态避障" : "航迹调整")
    : dynamic ? (sample.episode?.status === "cleared" ? "恢复航行" : "避障结束") : "调整结束";
  const clearance = sample.episode?.min_separation_clearance_m;
  const value = Number.isFinite(clearance) ? `${clearance.toFixed(2)} m` : "未测";
  setText(document.getElementById("avoidanceStatusTitle"), status);
  setText(document.getElementById("avoidanceClearance"), `最小安全净空 ${value}`);
  avoidanceStatus.dataset.active = String(sample.active);
  avoidanceStatus.dataset.violation = String(Number.isFinite(clearance) && clearance < 0);
}

function setSceneActorVisibility(htmlRuntimeActive) {
  if (plannedRoute) plannedRoute.visible = htmlRuntimeActive;
  if (avoidanceView && !htmlRuntimeActive) avoidanceView.object.visible = false;
  if (uavGroup) {
    uavGroup.visible = htmlRuntimeActive;
  }
  birdMeshes.forEach((bird) => {
    bird.visible = htmlRuntimeActive;
  });
  if (pathLine) {
    pathLine.visible = htmlRuntimeActive;
  }
  if (trailHead && !htmlRuntimeActive) trailHead.visible = false;
}


function clearEnvironment() {
  goalReward.hidden = true;
  goalStar = null;
  disposeRoute();
  if (avoidanceView) {
    scene.remove(avoidanceView.object);
    avoidanceView.dispose();
    avoidanceView = null;
  }
  displayedAvoidance = null;
  renderAvoidanceStatus(null);
  if (flightTrail) {
    scene.remove(flightTrail.object);
    flightTrail.dispose();
    flightTrail = null;
    pathLine = null;
  }
  if (trailHead) {
    scene.remove(trailHead);
    trailHead.geometry.dispose();
    trailHead.material.dispose();
    trailHead = null;
  }
  motionTimeline.reset();
  [...scene.children].forEach((child) => {
    if (child.userData.environment || child === uavGroup || birdMeshesHas(child) || child === pathLine) {
      child.traverse((object) => {
        if (object.isInstancedMesh) object.dispose();
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
  campusPlaceLabels = null;
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
  renderSemanticMission();
}


function renderToolbarState() {
  const semantic = isSemanticScenario();
  simulator?.setAttribute("data-mission-mode", semantic ? "semantic" : "navigation");
  simulator?.setAttribute("data-operation", commandError ? "error" : state?.flight?.mode || "idle");

  const missionPhase = semantic ? "校园任务" : { fly_to_target: "Outbound", returning_home: "Returning", completed: "Completed" }[state?.phase] || "--";
  setText(phaseValue, `${missionPhase} / ${state?.flight?.stage || "--"} / ${state?.flight?.mode || "idle"}`);
  const traceDecision = getHistoryEntries()[0]?.decision;
  const traceBackend = semantic ? "DeepSeek" : traceDecision?.agent_backend || state?.agent_backend;
  const traceModel = semantic ? null : traceDecision?.agent_backend
    ? traceDecision.agent_model
    : state?.agent_model;
  setText(backendValue, traceModel ? `${traceBackend}:${traceModel}` : traceBackend);
  const flight = state?.flight || {};
  const active = Boolean(semantic ? flight.running : flight.running || flight.planning);
  const unavailable = !state || state.status !== "active";
  planBtn.hidden = semantic;
  planBtn.disabled = commandPending || unavailable || active || flight.mode === "braking";
  startBtn.disabled = commandPending || unavailable || flight.mode === "braking"
    || (semantic && !["interpreting", "running", "error"].includes(state.semantic_mission?.status));
  resetBtn.disabled = commandPending || !state;
  scenarioSelect.disabled = commandPending || !state;
  agentButtons.forEach((button) => {
    const selected = button.dataset.agent === state?.agent_backend;
    button.setAttribute("aria-checked", String(selected));
    button.classList.toggle("active", selected);
    button.hidden = semantic && button.dataset.agent === "mock";
    button.disabled = commandPending || (semantic && button.dataset.agent === "mock");
  });
  if (agentKeyBtn) {
    agentKeyBtn.disabled = commandPending;
  }
  startBtn.setAttribute("aria-pressed", String(active));

  const iconChanged = updateCommandButton(
    startBtn,
    semantic ? (active ? "暂停任务" : "继续任务") : (active ? "Pause" : "Start"),
    active ? "pause" : "play",
  );
  if (iconChanged) {
    createLucideIcons();
  }
}


function isSemanticScenario() {
  return state?.scenario?.mission_mode === "semantic";
}


function semanticStatusLabel(status) {
  return {
    idle: "待提交", interpreting: "解析中", needs_clarification: "待补充", generation_failed: "生成失败",
    running: "执行中", completed: "已完成", error: "任务异常", safety_wait: "等待处理",
  }[status] || "待提交";
}


async function submitSemanticTask(event) {
  event?.preventDefault();
  if (!isSemanticScenario() || commandPending) return false;
  const instruction = taskInstruction.value.trim();
  if (!instruction) return false;
  if (!state.deepseek_key_configured) {
    openAgentDialog();
    return false;
  }
  return runCommand(() => postState("/api/mission/task", { instruction }), { semanticValidation: true });
}


const SEMANTIC_ACTION_LABELS = {
  visit: "访问", inspect: "巡检", thermal_scan: "热观测", capture: "模拟采样", motion: "轨迹飞行",
  control: "位置控制", review: "执行评估", return_home: "返回并降落",
};
const STEP_STATUS_LABELS = {
  pending: "待执行", active: "进行中", completed: "已完成",
  skipped: "条件不满足，跳过", cancelled: "任务修改后取消",
};


function pointOfInterestLabel(id) {
  return state?.scenario?.points_of_interest?.find(point => point.id === id)?.label || id || "指定地点";
}


function semanticElement(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}


function semanticConditionLabel(expression, mission, revision = mission.plan_revision ?? mission.revision) {
  if (!isRecord(expression)) return "待确认条件";
  if (Array.isArray(expression.all)) return expression.all
    .map(condition => `（${semanticConditionLabel(condition, mission, revision)}）`).join("且");
  if (Array.isArray(expression.any)) return expression.any
    .map(condition => `（${semanticConditionLabel(condition, mission, revision)}）`).join("或");
  if (expression.not) return `不满足（${semanticConditionLabel(expression.not, mission, revision)}）`;
  const fields = {
    temperature_c: "温度", anomaly: "异常", captured: "采样完成", inspected: "巡检完成",
    visited: "到达地点", returned: "已返航", landed: "已降落",
    battery_percent: "电量", elapsed_seconds: "任务用时", completed: "动作完成", reviewed: "评估完成", state: "飞行状态",
    total_length_m: "轨迹总长", distance_m: "轨迹覆盖距离", progress: "轨迹进度",
  };
  const operator = { "<": "低于", "<=": "不超过", "==": "等于", "!=": "不等于", ">=": "至少", ">": "超过" };
  const field = fields[expression.field] || expression.field || "结果";
  const result = expression.evidence_id ? (mission.evidence || []).find(entry => entry.evidence_id === expression.evidence_id) : null;
  const dependency = result || (revision === (mission.plan_revision ?? mission.revision)
    ? semanticProgramNodes(mission.program).find(step => step.id === expression.step_id) : null)
    || (mission.steps || []).find(step => step.id === expression.step_id && (!revision || step.revision === revision));
  const location = expression.source === "telemetry" ? "" : `${pointOfInterestLabel(dependency?.target_id)} · `;
  const value = typeof expression.value === "boolean" ? (expression.value ? "是" : "否") : String(expression.value ?? "待确认");
  const unit = expression.field === "temperature_c" ? " °C"
    : expression.field === "battery_percent" ? "%" : expression.field === "elapsed_seconds" ? " 秒"
      : ["total_length_m", "distance_m"].includes(expression.field) ? " m" : "";
  return `${location}${field}${operator[expression.op] || "等于"} ${value}${unit}`;
}


function semanticProgramNodes(program) {
  const nodes = Array.isArray(program) ? program : Array.isArray(program?.steps) ? program.steps : [];
  return nodes.flatMap(node => [node,
    ...semanticProgramNodes(node.steps), ...semanticProgramNodes(node.then), ...semanticProgramNodes(node.else)]);
}


function semanticProgramLabel(node, mission) {
  const action = semanticStepLabel(node);
  if (node.action === "repeat") {
    const progress = (mission.loops || []).find(loop => loop.id === node.id && loop.revision === (mission.plan_revision ?? mission.revision));
    const total = Number.isInteger(node.count) ? `${node.count} 次` : "持续";
    const iteration = progress?.iteration > 0 ? ` · 第 ${progress.iteration} 次${progress.status === "completed" ? "，已结束" : ""}` : "";
    return `循环 ${total}${iteration}`;
  }
  if (node.action === "branch") return `条件分支 · ${semanticConditionLabel(node.condition, mission)}`;
  if (node.action === "motion") return [action, semanticMotionSummary(node)].filter(Boolean).join(" · ");
  if (node.action === "control") return [action, semanticControlSummary(node)].filter(Boolean).join(" · ");
  return action;
}


function semanticStepLabel(step) {
  if (step.action === "review") return SEMANTIC_ACTION_LABELS.review;
  const action = step.action === "control" && isRecord(step.control)
    ? {landed: "降落停留", airborne: "定点停留"}[step.control.state] || SEMANTIC_ACTION_LABELS.control
    : SEMANTIC_ACTION_LABELS[step.action] || "执行任务";
  return `${action} · ${pointOfInterestLabel(step.target_id)}`;
}


function semanticControlSummary(step) {
  const control = isRecord(step.control) ? step.control : {};
  const details = [];
  if (control.anchor === "center") details.push("场地中心");
  if (Array.isArray(control.offset_m)) {
    const directions = [["东", "西"], ["南", "北"], ["上", "下"]];
    control.offset_m.slice(0, 3).forEach((distance, axis) => {
      if (Number.isFinite(distance) && distance !== 0) details.push(`向${directions[axis][distance > 0 ? 0 : 1]} ${Math.abs(distance)} m`);
    });
  }
  if (control.state === "airborne" && Number.isFinite(control.altitude_m)) details.push(`高度 ${control.altitude_m} m`);
  if (Number.isFinite(control.duration_seconds) && control.duration_seconds > 0) details.push(`停留 ${control.duration_seconds} 秒`);
  if (Number.isFinite(control.yaw_deg)) details.push(`朝向 ${control.yaw_deg}°`);
  return details.join(" · ");
}


function semanticGoalMessage(mission) {
  const goal = isRecord(mission.goal_state) ? mission.goal_state : {};
  if (mission.status !== "completed" || !isRecord(mission.goal_state)) return "";
  if (goal.satisfied !== true) return "任务结束状态尚未确认。";
  if (!goal.target_id) return mission.message || "任务已完成，保持当前位置。";
  const location = pointOfInterestLabel(goal.target_id);
  if (goal.state === "landed") return `任务已完成，已在${location}降落停驻。`;
  if (goal.state === "airborne") return `任务已完成，正在${location}上空保持位置。`;
  return "任务结束状态尚未确认。";
}


function semanticMotionSummary(step) {
  const motion = isRecord(step.motion) ? step.motion : {};
  const path = isRecord(motion.path) ? motion.path : {};
  const details = [];
  if (path.kind === "boundary") {
    const direction = path.direction === "counterclockwise" ? "逆时针" : "顺时针";
    details.push(`${direction}${Number.isInteger(path.laps) && path.laps > 0 ? ` ${path.laps} 圈` : "沿边界飞行"}`);
  } else if (path.kind === "curve") details.push("三维曲线");
  else if (path.kind === "polyline") details.push("指定路径");
  if (Number.isFinite(motion.speed_mps)) details.push(`${motion.speed_mps} m/s`);
  if (motion.heading?.mode === "fixed" && Number.isFinite(motion.heading.yaw_deg)) {
    details.push(`固定朝向 ${motion.heading.yaw_deg}°`);
  } else if (motion.heading?.mode === "face_target") {
    details.push(`朝向${pointOfInterestLabel(motion.heading.target_id)}`);
  }
  return details.join(" · ");
}


function semanticMotionProgressLabel(step, mission) {
  const progress = mission.motion_progress;
  if (step.action !== "motion" || step.status !== "active" || !isRecord(progress)) return "";
  const isCurrent = mission.current_execution_id ? step.execution_id === mission.current_execution_id
    : step.id === mission.current_step_id && step.revision === (mission.plan_revision ?? mission.revision);
  if (!isCurrent || (progress.execution_id && progress.execution_id !== step.execution_id)) return "";
  if (progress.phase === "transit") return "前往轨迹起点";
  if (progress.phase === "complete") return "轨迹已通过，正在确认完成";
  if (progress.phase !== "tracking") return "";
  const percent = Number.isFinite(progress.progress)
    ? ` ${Math.floor(Math.max(0, Math.min(0.99, progress.progress)) * 100)}%` : "";
  const distance = Number.isFinite(progress.distance_m) && Number.isFinite(progress.total_length_m)
    && progress.distance_m >= 0 && progress.total_length_m > 0
    ? ` · ${Math.floor(progress.distance_m)} / ${Math.ceil(progress.total_length_m)} m` : "";
  const laps = Number.isInteger(progress.completed_laps) && Number.isInteger(progress.laps)
    && progress.completed_laps >= 0 && progress.laps > 0
    ? ` · 已完成 ${progress.completed_laps} / ${progress.laps} 圈` : "";
  return `轨迹进度${percent}${distance}${laps}`;
}


function semanticEvidenceForStep(step, mission) {
  if (step.status !== "completed") return null;
  return (mission.evidence || []).find(entry => isRecord(entry)
    && entry.mission_id === mission.mission_id && entry.vehicle_id === mission.vehicle_id
    && entry.target_id === step.target_id && entry.action === step.action
    && (step.execution_id ? entry.execution_id === step.execution_id && entry.revision === step.revision
      : !entry.execution_id && entry.step_id === step.id));
}


function renderSemanticMission() {
  const semantic = isSemanticScenario();
  semanticPanel.hidden = !semantic;
  if (semantic !== renderedMissionMode) {
    renderedMissionMode = semantic;
    flightDiagnostics.open = !semantic;
    document.getElementById("flightDiagnosticsSummary").hidden = !semantic;
    setText(document.getElementById("inspectorHeading"), semantic ? "任务工作台" : "Agent Trace");
  }
  if (!semantic) return;
  const mission = state.semantic_mission || {};
  const steps = Array.isArray(mission.steps) ? mission.steps : [];
  const evidence = Array.isArray(mission.evidence) ? mission.evidence : [];
  const configured = Boolean(state.deepseek_key_configured);
  const planRevision = mission.plan_revision ?? mission.revision;
  const revisionSteps = steps.filter(step => !planRevision || step.revision === planRevision);
  setText(document.getElementById("plannerProgress"), `当前任务已完成 ${revisionSteps.filter(step => step.status === "completed").length} 步 · ${state.flight?.planner_calls || 0} 次本地规划`);
  taskInstruction.disabled = commandPending;
  submitTaskBtn.disabled = commandPending || !configured || !taskInstruction.value.trim();
  submitTaskBtn.textContent = mission.status === "generation_failed" ? "重试任务"
    : mission.instruction || mission.revision > 0 || steps.length ? "修改任务" : "提交任务";
  setText(document.getElementById("instructionCount"), `${taskInstruction.value.length} / 2000`);
  taskForm.setAttribute("aria-busy", String(commandPending || mission.status === "interpreting"));

  const recommendations = Array.isArray(state.scenario.recommended_tasks) ? state.scenario.recommended_tasks : [];
  const nextRecommendationKey = JSON.stringify(recommendations);
  if (nextRecommendationKey !== recommendationKey) {
    recommendationKey = nextRecommendationKey;
    recommendedTasks.replaceChildren(...recommendations.map((instruction, index) => {
      const button = semanticElement("button", "recommended-task");
      button.type = "button";
      button.append(semanticElement("span", "recommendation-index", String(index + 1).padStart(2, "0")),
        semanticElement("span", "", String(instruction)));
      button.addEventListener("click", () => {
        taskInstruction.value = String(instruction);
        commandError = "";
        renderSemanticMission();
        taskInstruction.focus({ preventScroll: true });
      });
      return button;
    }));
  }
  const statusElement = document.getElementById("semanticStatus");
  statusElement.dataset.status = mission.status || "idle";
  const paused = ["running", "interpreting"].includes(mission.status) && !state.flight?.running;
  setText(statusElement, paused ? "已暂停" : semanticStatusLabel(mission.status));
  const defaults = {
    idle: "暂无任务",
    interpreting: "正在解析任务指令。",
    needs_clarification: "请补充具体地点或任务要求后重新提交。",
    generation_failed: "任务程序生成失败，可重试原指令。",
    running: "正在执行任务。",
    completed: "任务已完成。",
    error: "任务未完成，请调整指令后重新提交。",
    safety_wait: "任务已安全等待，请修改任务或提交返航指令。",
  };
  const message = document.getElementById("semanticMessage");
  message.dataset.error = String(Boolean(commandError || ["error", "generation_failed"].includes(mission.status)));
  const feedback = Array.isArray(mission.execution_feedback)
    ? [...mission.execution_feedback].reverse().find(entry => typeof entry?.message === "string" && entry.message.trim())?.message : "";
  const messageText = commandError || (paused ? "任务已暂停，可继续或重置。"
    : semanticGoalMessage(mission) || mission.message || feedback || defaults[mission.status] || defaults.idle);
  setText(message, messageText);
  const summary = document.getElementById("semanticSummary");
  const summaryText = typeof mission.summary === "string" ? mission.summary.trim() : "";
  summary.hidden = !summaryText || summaryText === messageText.trim();
  setText(summary, summary.hidden ? "" : summaryText);
  const constraintsElement = document.getElementById("semanticConstraints");
  const constraints = mission.constraints || {};
  const constraintLabels = [];
  if (Number.isFinite(constraints.max_altitude_m)) constraintLabels.push(`限高 ${constraints.max_altitude_m} m`);
  if (Array.isArray(constraints.avoid_regions)) {
    constraints.avoid_regions.forEach((id) => {
      const label = state.scenario.semantic_regions?.find(region => region.id === id)?.label || id;
      constraintLabels.push(`避开${label}`);
    });
  }
  constraintsElement.hidden = constraintLabels.length === 0;
  if (constraintsElement.dataset.value !== JSON.stringify(constraintLabels)) {
    constraintsElement.dataset.value = JSON.stringify(constraintLabels);
    constraintsElement.replaceChildren(...constraintLabels.map(label => semanticElement("span", "constraint-chip", label)));
  }

  const nextStepsKey = JSON.stringify([state.scenario_id, mission.mission_id, mission.vehicle_id,
    mission.revision, planRevision, mission.program, mission.loops, steps, mission.current_step_id, mission.current_execution_id, evidence,
    steps.map(step => semanticMotionProgressLabel(step, mission))]);
  if (nextStepsKey !== semanticStepsKey) {
    semanticStepsKey = nextStepsKey;
    const program = semanticProgramNodes(mission.program).filter(node => ["repeat", "branch"].includes(node.action));
    semanticProgram.hidden = program.length === 0;
    semanticProgram.replaceChildren(...program.map(node => {
      const item = semanticElement("div", "task-program-rule");
      item.append(semanticElement("strong", "", semanticProgramLabel(node, mission)));
      if (node.action === "repeat") {
        const labels = (node.steps || []).map(child => semanticProgramLabel(child, mission));
        item.append(semanticElement("span", "", labels.join(" → ")));
        if (node.until) item.append(semanticElement("span", "", `结束条件：${semanticConditionLabel(node.until, mission)}`));
        if (node.when) item.append(semanticElement("span", "", `执行条件：${semanticConditionLabel(node.when, mission)}`));
      } else {
        const branchLabel = nodes => (nodes || []).map(child => semanticProgramLabel(child, mission)).join(" → ");
        item.append(semanticElement("span", "", `满足时：${branchLabel(node.then) || "继续后续任务"}`));
        if (node.else?.length) item.append(semanticElement("span", "", `否则：${branchLabel(node.else)}`));
      }
      return item;
    }));
    const revisions = new Set(steps.map(step => step.revision || 1));
    let displayedRevision = null;
    const items = [];
    steps.forEach((step, index) => {
      const revision = step.revision || 1;
      if (revisions.size > 1 && displayedRevision !== revision) {
        const heading = semanticElement("li", "task-revision-heading",
          `${revision === planRevision ? "当前任务" : "任务记录"} · 第 ${revision} 版`);
        items.push(heading);
        displayedRevision = revision;
      }
      const item = semanticElement("li", "semantic-step");
      item.dataset.status = step.status || "pending";
      if (mission.current_execution_id ? step.execution_id === mission.current_execution_id
        : step.id === mission.current_step_id && step.status === "active") item.setAttribute("aria-current", "step");
      const marker = step.status === "completed" ? "✓" : ["skipped", "cancelled"].includes(step.status) ? "−" : String(index + 1);
      item.append(semanticElement("span", "task-step-marker", marker));
      const content = semanticElement("div", "task-step-content");
      content.append(semanticElement("strong", "", semanticStepLabel(step)));
      const detail = step.action === "motion" ? semanticMotionSummary(step)
        : step.action === "control" ? semanticControlSummary(step)
          : step.action === "review" && typeof step.instruction === "string" ? step.instruction.trim() : "";
      if (detail) content.append(semanticElement("span", "task-step-iteration", detail));
      const status = STEP_STATUS_LABELS[step.status] || "待执行";
      const duration = step.action !== "control" && Number(step.duration_seconds) > 0 ? ` · 停留 ${step.duration_seconds} 秒` : "";
      const result = semanticEvidenceForStep(step, mission);
      content.append(semanticElement("span", "task-step-status", result ? evidenceResultLabel(result)
        : semanticMotionProgressLabel(step, mission) || status + duration));
      const iterations = Object.values(step.loop_iterations || {});
      if (iterations.length) content.append(semanticElement("span", "task-step-iteration", `循环第 ${iterations.join(" / ")} 次`));
      if (step.when) {
        content.append(semanticElement("span", "task-step-condition", `条件：${semanticConditionLabel(step.when, mission, step.revision)}`));
      }
      item.append(content);
      items.push(item);
    });
    semanticSteps.replaceChildren(...items);
  }
}


function evidenceResultLabel(entry) {
  const result = isRecord(entry.result) ? entry.result : {};
  if (entry.action === "control") {
    if (result.completed !== true) return "目标状态尚未确认。";
    if (result.state === "landed" && result.landed === true) return "遥测已确认降落并停驻。";
    if (result.state === "airborne" && result.landed === false) return "遥测已确认到达并保持位置。";
    return "目标状态尚未确认。";
  }
  if (entry.action === "review") return result.reviewed === true ? "执行反馈已评估。" : "正在评估执行反馈。";
  if (entry.action === "thermal_scan") {
    const temperature = Number.isFinite(result.temperature_c) ? `${result.temperature_c.toFixed(1)} °C` : "温度未记录";
    return `${temperature} · ${result.anomaly === true ? "检测到模拟异常" : result.anomaly === false ? "未检测到模拟异常" : "异常状态未确认"}`;
  }
  if (entry.action === "motion") {
    if (result.completed !== true) return "轨迹完成尚未确认。";
    const distance = Number.isFinite(result.distance_m) && result.distance_m >= 0
      ? ` · 已覆盖 ${result.distance_m.toFixed(1)} m` : "";
    return `已完成规定轨迹${distance}`;
  }
  if (entry.action === "capture") return result.captured === true
    ? "模拟采样记录已生成；不生成照片。" : "模拟采样未确认完成。";
  if (entry.action === "inspect") return result.inspected === true ? "已完成仿真巡检。" : "巡检未确认完成。";
  if (entry.action === "return_home") return result.returned === true && result.landed === true ? "遥测已确认返回起点并降落。" : "返航或降落尚未确认。";
  return result.visited === true ? "遥测已确认到达指定位置。" : "动作结果尚未确认。";
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
      ? `${currentDynamicRiskLevel(dynamicDistanceM(threat), threat)} / ${formatNumber(dynamicDistanceM(threat))} m（当前）`
      : hasOwn(observation, "nearest_dynamic_obstacle") ? "none" : "--",
  );
  if (threatValue) {
    threatValue.style.color = threatColor(currentDynamicRiskLevel(dynamicDistanceM(threat), threat));
    threatValue.title = "当前中心距对应的风险等级；不包含未来轨迹预测。";
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
  setText(missionProgressText, isSemanticScenario()
    ? `${(state.semantic_mission?.steps || []).filter(step => step.status === "completed").length} 个步骤已验证完成 · 当前阶段 ${semanticStatusLabel(state.semantic_mission?.status)}`
    : `${milestones.target_reached ? "★ Collected · Target reached" : "☆ Target pending"} · ${milestones.base_returned ? "Returned to base" : "Return pending"}`);
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
    button.disabled = commandPending || (isSemanticScenario() && button.dataset.runtime === "airsim");
    button.title = isSemanticScenario() && button.dataset.runtime === "airsim"
      ? "园区任务目前使用 HTML 仿真观测" : "";
  });
  if (airsimConfigBtn) {
    airsimConfigBtn.disabled = commandPending || isSemanticScenario();
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

  reconnectAirSimBtn.disabled = commandPending || isSemanticScenario();
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


function showAgentConfigError(message) {
  agentDialogError.textContent = message;
  agentDialogError.hidden = !message;
}


function agentConfigurationError(error) {
  // Only these endpoints' fixed, server-curated errors may reach this form.
  const safeCodes = new Set(["authentication_failed", "model_unavailable",
    "endpoint_unreachable", "configuration_invalid", "missing_key",
    "provider_changed", "models_unavailable"]);
  return safeCodes.has(error?.code) ? error.message : "无法读取或验证配置，请检查连接后重试。";
}


function agentConfigContext() {
  return JSON.stringify([agentProvider.value, deepseekBaseUrl.value.trim().replace(/\/+$/, ""), agentKeyRevision]);
}


function agentProviderProfile() {
  return agentConfigCatalog?.profiles?.find(profile => profile.id === agentProvider.value) || null;
}


function agentKeyReusable() {
  const profile = agentProviderProfile();
  return Boolean(profile?.key_configured && profile.base_url?.replace(/\/+$/, "")
    === deepseekBaseUrl.value.trim().replace(/\/+$/, ""));
}


function setAgentModelOptions(models, selected = "") {
  deepseekModel.replaceChildren();
  for (const model of models.length ? models : [""]) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model || "请先读取可用模型";
    deepseekModel.appendChild(option);
  }
  if (models.includes(selected)) deepseekModel.value = selected;
}


function refreshAgentConfigControls() {
  const unavailable = agentConfigLoading || agentConfigSaving || !agentConfigCatalog;
  agentProvider.disabled = unavailable;
  deepseekApiKey.disabled = unavailable;
  deepseekBaseUrl.disabled = unavailable;
  const ready = agentModelsContext === agentConfigContext() && Boolean(deepseekModel.value);
  deepseekModel.disabled = unavailable || agentModelsLoading || !ready;
  loadModelsBtn.disabled = unavailable || agentModelsLoading || !deepseekBaseUrl.value.trim()
    || (!deepseekApiKey.value.trim() && !agentKeyReusable());
  saveAgentBtn.disabled = unavailable || agentModelsLoading || !ready;
  loadModelsBtn.textContent = agentModelsLoading ? "读取中…" : "读取模型";
  saveAgentBtn.textContent = agentConfigSaving ? "验证中…" : "验证并保存";
  deepseekApiKey.placeholder = agentKeyReusable() ? "已保存此服务 Key；留空复用" : "输入此服务的 API Key";
  setText(agentKeyStatus, agentKeyReusable()
    ? "已保存此服务的 Key；填写新 Key 可替换。" : "Key 仅保存在当前服务进程内。");
}


function invalidateAgentModels() {
  agentModelRequest += 1;
  agentModelsContext = "";
  agentModelsLoading = false;
  setAgentModelOptions([]);
  agentDialogStatus.textContent = "";
  showAgentConfigError("");
}


function selectAgentProvider() {
  deepseekApiKey.value = "";
  agentKeyRevision += 1;
  invalidateAgentModels();
  const provider = agentConfigCatalog?.providers?.find(entry => entry.id === agentProvider.value);
  deepseekBaseUrl.value = agentProviderProfile()?.base_url || provider?.default_base_url || "";
  agentAdvanced.open = agentProvider.value === "custom";
  refreshAgentConfigControls();
  if (agentKeyReusable()) void loadAgentModels();
}


async function loadAgentModels() {
  if (agentConfigLoading || agentConfigSaving || !agentConfigCatalog) return;
  const context = agentConfigContext();
  const epoch = agentConfigEpoch;
  const request = ++agentModelRequest;
  const body = {provider: agentProvider.value, base_url: deepseekBaseUrl.value.trim()};
  if (deepseekApiKey.value.trim()) body.api_key = deepseekApiKey.value.trim();
  if (!body.base_url || (!body.api_key && !agentKeyReusable())) {
    showAgentConfigError("请填写此服务的 API Key 和接口地址。");
    return;
  }
  agentModelsLoading = true;
  agentModelsContext = "";
  showAgentConfigError("");
  setText(agentDialogStatus, "正在读取可用模型…");
  refreshAgentConfigControls();
  try {
    const payload = await requestJson("/api/config/models", {method: "POST", body: JSON.stringify(body)});
    if (epoch !== agentConfigEpoch || request !== agentModelRequest || !agentDialog.open
      || context !== agentConfigContext()) return;
    if (payload.provider !== body.provider) throw new Error("Provider changed");
    const models = [...new Set((Array.isArray(payload.models) ? payload.models : [])
      .filter(model => typeof model === "string" && model.trim()))];
    if (!models.length) {
      setAgentModelOptions([]);
      agentDialogStatus.textContent = "";
      showAgentConfigError("此接口未返回可用模型，请检查 Key、地域和高级设置中的地址。");
      return;
    }
    const preferred = [agentProviderProfile()?.model, payload.current_model, payload.recommended_model]
      .find(model => models.includes(model)) || models[0];
    setAgentModelOptions(models, preferred);
    agentModelsContext = context;
    setText(agentDialogStatus, "已读取可用模型，请确认后保存。");
  } catch (error) {
    if (epoch !== agentConfigEpoch || request !== agentModelRequest || !agentDialog.open) return;
    setAgentModelOptions([]);
    agentDialogStatus.textContent = "";
    showAgentConfigError(agentConfigurationError(error));
  } finally {
    if (epoch === agentConfigEpoch && request === agentModelRequest && agentDialog.open) {
      agentModelsLoading = false;
      refreshAgentConfigControls();
    }
  }
}


function clearAgentDialog() {
  deepseekApiKey.value = "";
  agentConfigEpoch += 1;
  agentModelRequest += 1;
  agentKeyRevision += 1;
  agentModelsContext = "";
  agentConfigCatalog = null;
  agentConfigLoading = false;
  agentModelsLoading = false;
  agentConfigSaving = false;
}


function closeAgentDialog() {
  clearAgentDialog();
  closeDialog(agentDialog);
}


async function openAgentDialog() {
  if (commandPending) return;
  clearAgentDialog();
  const epoch = agentConfigEpoch;
  agentConfigLoading = true;
  showAgentConfigError("");
  setAgentModelOptions([]);
  setText(agentDialogStatus, "正在读取配置…");
  openDialog(agentDialog);
  refreshAgentConfigControls();
  try {
    const catalog = await requestJson("/api/config");
    if (epoch !== agentConfigEpoch || !agentDialog.open) return;
    agentConfigCatalog = catalog;
    agentProvider.replaceChildren();
    for (const provider of catalog.providers || []) {
      const option = document.createElement("option");
      option.value = provider.id;
      option.textContent = provider.label;
      agentProvider.appendChild(option);
    }
    const selected = catalog.active_provider || state?.agent_provider || "deepseek";
    if ((catalog.providers || []).some(provider => provider.id === selected)) agentProvider.value = selected;
    agentConfigLoading = false;
    selectAgentProvider();
    deepseekApiKey.focus({preventScroll: true});
  } catch (error) {
    if (epoch !== agentConfigEpoch || !agentDialog.open) return;
    agentConfigLoading = false;
    agentDialogStatus.textContent = "";
    showAgentConfigError(agentConfigurationError(error));
    refreshAgentConfigControls();
  }
}


function openAirSimDialog() {
  if (commandPending || isSemanticScenario()) {
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
  parts.push(`当前 ${currentDynamicRiskLevel(dynamicDistanceM(obstacle), obstacle)}`);
  parts.push(`中心距 ${formatNumber(dynamicDistanceM(obstacle))} m`);
  if (Number.isFinite(obstacle.separation_clearance_m)) {
    parts.push(`安全净空 ${formatNumber(obstacle.separation_clearance_m)} m`);
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
  if (level === "unknown") return "#94a3b8";
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


function worldFromSample(position) {
  return toWorld({x: position[0], y: position[1], z: position[2]});
}


function updateFollowCamera(delta, reset = false) {
  const target = uavGroup.position;
  if (reset) followRig.reset(target, cameraObstacles);
  camera.position.copy(followRig.update(target, delta, cameraObstacles,
    motionTimeline.preview().map(worldFromSample)));
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
    if (isSemanticScenario()) {
      // View from the south entrance, matching the campus reference photograph.
      const size = state.grid_size * WORLD_SCALE;
      const fit = Math.max(1, 1.1 / camera.aspect);
      camera.position.set(size * 0.12 * fit, size * 0.73 * fit, size * 0.88 * fit);
      controls.target.set(0, 0, size * 0.08);
    } else {
      camera.position.set((urban ? 50 : 55) * mapScale,
        (urban ? 42 : 60) * mapScale, (urban ? 56 : 65) * mapScale);
      controls.target.set(0, urban ? 24 : 17, 0);
    }
  } else if (uavGroup) {
    camera.position.copy(followRig.reset(uavGroup.position, cameraObstacles));
    camera.lookAt(uavGroup.position);
  }
}


function animate() {
  timer.update();
  const delta = Math.min(timer.getDelta(), 0.1);
  const elapsed = timer.getElapsed();
  const htmlRuntimeActive = state?.runtime_backend === "html";
  const frame = htmlRuntimeActive ? motionTimeline.sample(performance.now() / 1000) : null;

  if (uavGroup && frame) {
    uavGroup.position.copy(worldFromSample(frame.position));
    // Runtime yaw is measured from north; the camera pod faces local -z.
    uavGroup.rotation.y = -frame.yaw;
    uavGroup.rotation.z = frame.landed ? 0 : Math.sin(frame.time * 2.2) * 0.018;
    rotorMeshes.forEach((rotor) => {
      if (!frame.landed) rotor.rotation.y += delta * 42 * rotor.userData.speed;
    });
    flightTrail?.setVisibleEnd(frame.path_index);
    const head = (state.continuous_path || state.path || [])[frame.path_index - (state.path_start_index || 0)];
    if (trailHead) {
      trailHead.visible = Boolean(head && [head.x, head.y, head.z].every(Number.isFinite));
      if (trailHead.visible) {
        const attribute = trailHead.geometry.attributes.position;
        const start = toWorld(head), end = uavGroup.position;
        attribute.setXYZ(0, start.x, start.y, start.z);
        attribute.setXYZ(1, end.x, end.y, end.z);
        attribute.needsUpdate = true;
        trailHead.geometry.computeBoundingSphere();
      }
    }
  }

  displayedAvoidance = avoidanceView?.update(frame?.time, uavGroup?.position,
    state?.flight?.waypoints?.[state?.flight?.waypoint_index],
    Boolean(state?.flight?.running && !state.flight.planning && !state.flight.error),
    Boolean(htmlRuntimeActive && frame)) || null;
  renderAvoidanceStatus(displayedAvoidance);
  const highlightedBirds = new Set(displayedAvoidance
    ? [...(displayedAvoidance.threats || []).map(threat => threat.id), ...(displayedAvoidance.episode?.object_ids || [])] : []);

  if (htmlRuntimeActive && frame) {
    birdMeshes.forEach((bird, id) => {
      const sample = frame.traffic[id];
      bird.visible = Boolean(sample);
      if (!sample) return;
      const position = sample.slice(0, 3), velocity = sample.slice(3);
      const seconds = frame.time;
      bird.position.copy(worldFromSample(position));
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
      const phase = trafficMotion?.models[id]?.phase || 0;
      const glide = Math.sin(seconds * 0.65 + phase) > 0.5 && speed > 0.8;
      bird.userData.wings.forEach((wing, index) => {
        wing.rotation.z = (index === 0 ? -1 : 1) * (glide ? 0.08 : Math.sin(seconds * 9 + phase) * 0.45);
      });
      const separation = bird.position.distanceTo(uavGroup.position);
      const risk = currentDynamicRiskLevel(separation, bird.userData);
      const highlighted = highlightedBirds.has(id);
      bird.userData.avoidanceRelated = highlighted;
      bird.userData.highlightMaterials.forEach(material => {
        material.emissive.set(highlighted ? 0xff963e : 0x000000);
        material.emissiveIntensity = highlighted ? (displayedAvoidance.active ? .6 : .25) : 0;
      });
      const showEnvelope = risk !== "unknown" && (risk !== "low" || hoveredBird === bird || highlighted);
      bird.userData.safetyBubble.visible = showEnvelope;
      bird.userData.contactBubble.visible = showEnvelope;
      bird.userData.safetyBubble.material.color.set(highlighted && risk === "low" ? "#ff963e" : threatColor(risk));
      bird.userData.safetyBubble.material.opacity = highlighted && displayedAvoidance.active ? .23 : .12;
      bird.userData.contactBubble.material.color.set(risk === "critical" ? "#ff6b57" : "#94a3b8");
    });
  }
  updateGoalStar(delta, elapsed, htmlRuntimeActive, frame?.time);
  updateBirdHover(htmlRuntimeActive);

  if (cameraMode === "follow" && uavGroup) {
    updateFollowCamera(delta, Boolean(frame?.reset));
  } else {
    controls.update();
  }

  const mission = state?.semantic_mission;
  const activeStep = mission?.steps?.find(step => mission.current_execution_id
    ? step.execution_id === mission.current_execution_id
    : step.id === mission.current_step_id && step.status === "active");
  campusPlaceLabels?.update(camera, birdPointer, activeStep?.target_id,
    sceneContainer.clientWidth, sceneContainer.clientHeight);

  renderer.render(scene, camera);
}

function updateBirdHover(active) {
  hoveredBird = null;
  if (active && Math.abs(birdPointer.x) <= 1 && Math.abs(birdPointer.y) <= 1) {
    birdRaycaster.setFromCamera(birdPointer, camera);
    // Most pointer positions miss all birds. Avoid raycasting the entire city
    // and accumulated trail on every animation frame just to discover that.
    const birds = [...birdMeshes.values()].filter(bird => bird.visible);
    const bubbles = new Set(birds.flatMap(bird => [bird.userData.safetyBubble, bird.userData.contactBubble]));
    const candidates = birdRaycaster.intersectObjects(birds, true)
      .filter(entry => entry.object.visible && !bubbles.has(entry.object));
    if (!candidates.length) { birdTooltip.hidden = true; return; }
    const hit = birdRaycaster.intersectObjects(scene.children, true)
      .find((entry) => entry.object.visible && !bubbles.has(entry.object));
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
    const centerDistance = hoveredBird.position.distanceTo(uavGroup.position);
    const data = hoveredBird.userData;
    const text = `${data.id}${data.avoidanceRelated ? " · 避障关联目标" : ""} · 当前中心距 ${centerDistance.toFixed(1)} m\n内圈接触 ${formatNumber(data.contact_distance_m)} m · 外圈安全 ${formatNumber(data.separation_distance_m)} m`;
    if (birdTooltip.textContent !== text) birdTooltip.textContent = text;
    const width = sceneContainer.clientWidth;
    const height = sceneContainer.clientHeight;
    birdTooltip.style.left = `${Math.max(8, Math.min(width - 288, (birdPointer.x + 1) * width / 2 + 12))}px`;
    birdTooltip.style.top = `${Math.max(8, Math.min(height - 68, (1 - birdPointer.y) * height / 2 - 52))}px`;
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
    if (backend === "deepseek") {
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
closeAgentBtn?.addEventListener("click", closeAgentDialog);
cancelAgentBtn?.addEventListener("click", closeAgentDialog);
agentDialog?.addEventListener("cancel", clearAgentDialog);
agentDialog?.addEventListener("close", () => { if (!agentDialog.open) clearAgentDialog(); });
agentProvider?.addEventListener("change", selectAgentProvider);
loadModelsBtn?.addEventListener("click", loadAgentModels);
deepseekApiKey?.addEventListener("input", () => {
  agentKeyRevision += 1;
  invalidateAgentModels();
  refreshAgentConfigControls();
});
deepseekBaseUrl?.addEventListener("input", () => {
  invalidateAgentModels();
  refreshAgentConfigControls();
});
closeAirSimBtn?.addEventListener("click", () => closeDialog(airsimDialog));
closeAirSimFooterBtn?.addEventListener("click", () => closeDialog(airsimDialog));
window.addEventListener("resize", onResize);
new ResizeObserver(onResize).observe(sceneContainer);
taskForm?.addEventListener("submit", submitSemanticTask);
taskInstruction?.addEventListener("input", () => {
  commandError = "";
  renderSemanticMission();
});

createLucideIcons();
renderToolbarState();
renderer.setAnimationLoop(animate);
void pollState();
void loadScenarios().catch(() => setText(feedbackText, "Scenario list unavailable"));
