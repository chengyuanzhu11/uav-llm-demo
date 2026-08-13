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

const simulator = document.getElementById("simulator")
  || document.querySelector(".simulator");
const settingsBtn = document.getElementById("settingsBtn");
const settingsDialog = document.getElementById("settingsDialog");
const closeSettingsBtn = document.getElementById("closeSettingsBtn");
const inspectorToggleBtn = document.getElementById("inspectorToggleBtn");
const trajectoryBtn = document.getElementById("trajectoryBtn");
const traceView = document.getElementById("traceView");
const historyView = document.getElementById("historyView");
const historyBackBtn = document.getElementById("historyBackBtn");
const historyTableBody = document.getElementById("historyTableBody");
const historyEmpty = document.getElementById("historyEmpty");
const historyViewBtn = document.getElementById("historyViewBtn");
const operationBanner = document.getElementById("operationBanner");
const operationStatus = document.getElementById("operationStatus");
const sceneExecutionStatus = document.getElementById("sceneExecutionStatus");
const currentStepValue = document.getElementById("currentStepValue");
const goalDirectionValue = document.getElementById("goalDirectionValue");
const goalDistanceValue = document.getElementById("goalDistanceValue");
const staticRiskValue = document.getElementById("staticRiskValue");
const dynamicRiskValue = document.getElementById("dynamicRiskValue");
const agentActionValue = document.getElementById("agentActionValue");
const agentWaypointValue = document.getElementById("agentWaypointValue");
const agentRiskValue = document.getElementById("agentRiskValue");
const agentSkillValue = document.getElementById("agentSkillValue");
const agentReasonValue = document.getElementById("agentReasonValue");
const proposedActionValue = document.getElementById("proposedActionValue");
const executedActionValue = document.getElementById("executedActionValue");
const guardStatus = document.getElementById("guardStatus");
const guardReasonValue = document.getElementById("guardReasonValue");
const executionPositionValue = document.getElementById("executionPositionValue");
const executionRuntimeValue = document.getElementById("executionRuntimeValue");
const executionDurationValue = document.getElementById("executionDurationValue");
const executionBatteryValue = document.getElementById("executionBatteryValue");
const settingsAgentValue = document.querySelector('[data-settings-value="agent"]');
const settingsScenarioValue = document.querySelector('[data-settings-value="scenario"]');
const settingsRuntimeValue = document.querySelector('[data-settings-value="runtime"]');

const missionBadge = document.getElementById("missionBadge");
const positionValue = document.getElementById("positionValue");
const altitudeValue = document.getElementById("altitudeValue");
const batteryValue = document.getElementById("batteryValue");
const telemetryPhaseValue = document.getElementById("telemetryPhaseValue");
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
let operationState = "ready";
let operationMessage = "Ready";
let historyMode = false;
let selectedHistoryStep = null;
let trajectoryVisible = true;
let autoLoopActive = false;
let runtimeStopPending = false;
let currentStepSettled = Promise.resolve(null);
let settleCurrentStep = null;

let inspectorExpanded = false;
try {
  inspectorExpanded = sessionStorage.getItem("uav-inspector-expanded") === "true";
} catch (error) {
  inspectorExpanded = false;
}
simulator?.classList.toggle("inspector-expanded", inspectorExpanded);

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
  setOperationState("ready", "Ready");
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
    return currentStepSettled;
  }
  if (!state || state.status !== "active") {
    renderToolbarState();
    return null;
  }

  stepInFlight = true;
  const stepBackend = state.runtime_backend;
  const stepBefore = Number(state.step_count || 0);
  let outcome = {
    status: "pending",
    backend: stepBackend,
    stepBefore,
  };
  currentStepSettled = new Promise((resolve) => {
    settleCurrentStep = resolve;
  });
  setOperationState(
    "executing",
    `${String(stepBackend || "runtime").toUpperCase()} executing step #${stepBefore + 1}`,
  );
  try {
    state = await requestJson("/api/step", { method: "POST" });
    outcome = {
      status: "committed",
      backend: stepBackend,
      stepBefore,
      stepAfter: Number(state.step_count || stepBefore),
    };
    syncScene(false);
    renderUi();
    stopAutoWhenDone();
    if (state.status !== "active") {
      setOperationState(
        "ready",
        `Mission ${state.status} after step #${state.step_count}`,
      );
    } else if (autoRunning) {
      setOperationState(
        "executing",
        `Auto running after step #${state.step_count}`,
      );
    } else if (!runtimeStopPending) {
      setOperationState("ready", `Ready after step #${state.step_count}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    outcome = {
      status: "failed",
      backend: stepBackend,
      stepBefore,
      message,
    };
    autoRunning = false;
    setText(feedbackText, message);
    const expectedAirSimCancellation = runtimeStopPending
      && stepBackend === "airsim"
      && isCancellationError(message);
    if (!expectedAirSimCancellation) {
      setOperationState("error", message);
    }
  } finally {
    stepInFlight = false;
    const resolveStep = settleCurrentStep;
    settleCurrentStep = null;
    resolveStep?.(outcome);
    renderToolbarState();
  }
  return outcome;
}


async function resetSimulation() {
  await stopAuto();
  setOperationState("executing", "Resetting mission");
  try {
    state = await requestJson("/api/reset", { method: "POST" });
    selectedHistoryStep = null;
    showHistoryView(false);
    clearEnvironment();
    syncScene(true);
    renderUi();
    setOperationState("ready", "Mission reset");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText(feedbackText, message);
    setOperationState("error", message);
  }
}


async function switchScenario(scenarioId) {
  if (!state || runtimePending || scenarioId === state.scenario_id) {
    return;
  }
  await stopAuto(true);
  setRuntimePending(true);
  setOperationState("executing", `Switching scenario to ${scenarioId}`);
  try {
    state = await requestJson("/api/scenario", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    selectedHistoryStep = null;
    showHistoryView(false);
    clearEnvironment();
    syncScene(true);
    renderUi();
    setOperationState("ready", `Scenario ${scenarioId} ready`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText(feedbackText, message);
    scenarioSelect.value = state.scenario_id;
    setOperationState("error", message);
  } finally {
    setRuntimePending(false);
  }
}


function toggleAuto() {
  if (autoRunning) {
    void stopAuto(true);
    return;
  }
  if (
    runtimeStopPending
    || runtimePending
    || stepInFlight
    || !state
    || state.status !== "active"
  ) {
    return;
  }
  autoRunning = true;
  setOperationState("executing", "Auto running");
  startAutoLoop();
}


async function stopAuto(commandStop = false) {
  autoRunning = false;
  renderToolbarState();
  if (!commandStop || !state) {
    return;
  }

  const backend = state.runtime_backend;
  const stepBeforeStop = Number(state.step_count || 0);
  const pendingStep = stepInFlight ? currentStepSettled : null;
  runtimeStopPending = true;
  setOperationState(
    "executing",
    `Pausing ${String(backend || "runtime").toUpperCase()}`,
  );
  try {
    const stoppedState = await requestJson("/api/runtime/stop", { method: "POST" });
    const stepOutcome = pendingStep ? await pendingStep : null;
    if (stepOutcome?.status !== "committed") {
      state = stoppedState;
    }
    renderUi();

    const stoppedAtStep = Number(state.step_count || 0);
    if (stepOutcome?.status === "failed") {
      const cancelledBeforeCommit = backend === "airsim"
        && stoppedAtStep === stepBeforeStop
        && isCancellationError(stepOutcome.message);
      if (cancelledBeforeCommit) {
        setOperationState(
          "paused",
          `Paused; current action cancelled before commit at step #${stoppedAtStep} (AirSim)`,
        );
      } else {
        setText(feedbackText, stepOutcome.message);
        setOperationState("error", stepOutcome.message);
      }
    } else {
      setOperationState("paused", `Paused after step #${stoppedAtStep}`);
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText(feedbackText, message);
    setOperationState("error", message);
  } finally {
    runtimeStopPending = false;
    renderToolbarState();
  }
}


function stopAutoWhenDone() {
  if (state && state.status !== "active") {
    autoRunning = false;
    renderToolbarState();
  }
}


async function runAutoLoop() {
  while (autoRunning && state && state.status === "active") {
    const outcome = await stepSimulation();
    if (outcome?.status === "failed") {
      break;
    }
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


function startAutoLoop() {
  if (autoLoopActive) {
    return;
  }
  autoLoopActive = true;
  void runAutoLoop().finally(() => {
    autoLoopActive = false;
    if (autoRunning && state?.status === "active") {
      startAutoLoop();
    }
  });
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
  setOperationState("executing", `Switching runtime to ${backend}`);
  try {
    state = await requestJson("/api/runtime", {
      method: "POST",
      body: JSON.stringify({
        backend,
        config: backend === "airsim" ? airsimConfig() : {},
      }),
    });
    selectedHistoryStep = null;
    showHistoryView(false);
    clearEnvironment();
    syncScene(true);
    renderUi();
    setOperationState("ready", `${backend.toUpperCase()} runtime ready`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText(feedbackText, message);
    renderRuntimeControls();
    setOperationState("error", message);
  } finally {
    setRuntimePending(false);
  }
}


async function applyWind() {
  applyWindBtn.disabled = true;
  setOperationState("executing", "Applying AirSim wind");
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
    setOperationState("ready", "AirSim wind updated");
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    setText(feedbackText, message);
    setOperationState("error", message);
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
  renderToolbarState();
  renderRuntimeSettings();
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
  pathLine.visible = trajectoryVisible;
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
  renderToolbarState();
  if (!state) {
    return;
  }
  renderTelemetry();
  renderRuntimeSettings();
  renderHistory();

  const entries = getHistoryEntries();
  let selectedEntry = null;
  if (selectedHistoryStep !== null) {
    selectedEntry = entries.find(
      (entry) => String(entry?.step) === String(selectedHistoryStep),
    ) || null;
    if (!selectedEntry) {
      selectedHistoryStep = null;
    }
  }
  renderDecisionTrace(selectedEntry || entries[0] || null);
}


function renderToolbarState() {
  simulator?.setAttribute("data-operation", operationState);
  operationBanner?.setAttribute("data-operation", operationState);
  operationBanner?.setAttribute("aria-live", "polite");
  setText(operationStatus, operationMessage);
  setText(sceneExecutionStatus, operationMessage);

  const status = state?.status || "loading";
  setText(missionBadge, status);
  if (missionBadge) {
    missionBadge.style.color = status === "completed"
      ? "#62d57b"
      : status === "failed"
        ? "#ff6b57"
        : "#45d5d2";
  }
  setText(phaseValue, state?.phase);
  setText(
    backendValue,
    state?.agent_model
      ? `${state.agent_backend}:${state.agent_model}`
      : state?.agent_backend,
  );
  setText(stepCount, `${Number(state?.step_count || 0)} steps`);

  const metrics = isRecord(state?.metrics) ? state.metrics : {};
  setText(metricsValue, [
    `success ${metrics.success ? "yes" : "no"}`,
    `interventions ${metrics.safety_interventions ?? 0}`,
    `path ${formatNumber(metrics.path_length ?? 0)}`,
    `battery -${metrics.battery_used ?? 0}%`,
  ].join(" | "));

  const missionActive = state?.status === "active";
  stepBtn.disabled = runtimePending
    || runtimeStopPending
    || stepInFlight
    || autoRunning
    || !missionActive;
  autoBtn.disabled = runtimePending
    || runtimeStopPending
    || !missionActive
    || (stepInFlight && !autoRunning);
  resetBtn.disabled = runtimePending || stepInFlight;
  scenarioSelect.disabled = runtimePending || stepInFlight;
  autoBtn.setAttribute("aria-pressed", String(autoRunning));

  const iconChanged = updateCommandButton(
    autoBtn,
    autoRunning ? "Pause" : "Auto",
    autoRunning ? "pause" : "play",
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
      : `${state.battery}%`,
  );
  setText(telemetryPhaseValue, state.phase);

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
  const observation = structuredEntryObservation
    || ((!entry || isLatestEntry) && isRecord(state.observation)
      ? state.observation
      : {});
  const observationSummary = typeof entry?.observation === "string"
    ? entry.observation
    : observation.text;
  const hasObservation = Boolean(
    observationSummary || Object.keys(observation).length,
  );
  setTraceStageStatus(
    "observation",
    hasObservation ? "active" : "waiting",
    hasObservation ? "OBSERVED" : "WAITING",
  );

  setText(
    currentStepValue,
    `#${entry?.step ?? state.step_count ?? 0}`,
  );
  setText(observationText, observationSummary);
  setText(goalDirectionValue, observation.goal_direction);
  setText(goalDistanceValue, observation.goal_distance);
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
  setText(agentActionValue, decision.action);
  setText(agentWaypointValue, formatPoint(decision.waypoint));
  setText(agentRiskValue, decision.risk);
  setText(agentSkillValue, decision.skill);
  setText(agentReasonValue, decision.reason);
  setText(
    decisionJson,
    Object.keys(decision).length ? safeJson(decision) : "--",
  );

  const proposedAction = decision.proposed_action || decision.action;
  const executedAction = entry?.action ?? decision.action;
  const safetyIntervention = decision.safety_intervention;
  const hasDecision = Object.keys(decision).length > 0;
  setTraceStageStatus(
    "agent",
    hasDecision ? "pass" : "waiting",
    hasDecision ? "PROPOSED" : "WAITING",
  );
  const guardOverrode = Boolean(
    decision.safety_intervened
      || safetyIntervention
      || (
        proposedAction
        && executedAction
        && proposedAction !== executedAction
      ),
  );
  setText(proposedActionValue, proposedAction);
  setText(executedActionValue, executedAction);
  const guardLabel = hasDecision ? (guardOverrode ? "OVERRIDE" : "PASS") : "WAITING";
  setText(guardStatus, guardLabel);
  setText(guardReasonValue, safetyIntervention || decision.reason);
  setTraceStageStatus(
    "guard",
    hasDecision ? (guardOverrode ? "override" : "pass") : "waiting",
    guardLabel,
  );

  const execution = isRecord(entry?.execution) ? entry.execution : {};
  const useCurrentExecution = !entry || isLatestEntry;
  const kinematics = isRecord(entry?.kinematics)
    ? entry.kinematics
    : isRecord(execution.kinematics)
      ? execution.kinematics
      : useCurrentExecution && isRecord(state.kinematics)
        ? state.kinematics
        : {};
  const executionPosition = execution.position
    ?? entry?.position
    ?? (useCurrentExecution ? kinematics.position ?? state.uav : null);
  const executionRuntime = execution.runtime
    ?? entry?.runtime
    ?? (useCurrentExecution ? state.runtime_backend : null);
  const executionDuration = execution.duration
    ?? entry?.trajectory_duration
    ?? entry?.duration
    ?? (useCurrentExecution ? state.trajectory_duration : null);
  const executionBattery = execution.battery
    ?? entry?.battery
    ?? (useCurrentExecution ? state.battery : null);
  const executionFeedback = entry?.feedback
    ?? execution.feedback
    ?? (!entry ? state.feedback : null);
  const hasExecution = Boolean(
    entry && (
      executionFeedback
      || Object.keys(execution).length
      || Object.keys(kinematics).length
    ),
  );
  setTraceStageStatus(
    "execution",
    hasExecution ? "pass" : "waiting",
    hasExecution ? "EXECUTED" : "WAITING",
  );

  setText(executionPositionValue, formatPoint(executionPosition));
  setText(executionRuntimeValue, formatRuntimeValue(executionRuntime));
  setText(executionDurationValue, formatDuration(executionDuration));
  setText(
    executionBatteryValue,
    executionBattery === null || executionBattery === undefined
      ? "--"
      : `${executionBattery}%`,
  );
  setText(feedbackText, executionFeedback);
  setText(kinematicsValue, formatKinematics(kinematics));
}


function renderHistory() {
  const entries = getHistoryEntries();
  historyList?.replaceChildren();
  historyTableBody?.replaceChildren();
  renderHistoryNavigation();

  if (historyEmpty) {
    historyEmpty.hidden = entries.length > 0;
    historyEmpty.classList.toggle("is-hidden", entries.length > 0);
  }

  if (!entries.length) {
    if (historyList) {
      const empty = document.createElement("p");
      empty.className = "empty-log";
      empty.textContent = "Waiting for the first Agent decision.";
      historyList.appendChild(empty);
    }
    return;
  }

  entries.slice(0, 3).forEach((entry) => {
    if (!historyList) {
      return;
    }
    const decision = isRecord(entry?.decision) ? entry.decision : {};
    const actions = historyActions(entry);
    const item = document.createElement("article");
    item.className = "history-item";
    item.tabIndex = 0;
    item.setAttribute("role", "button");
    item.classList.toggle(
      "selected",
      selectedHistoryStep !== null
        && String(entry?.step) === String(selectedHistoryStep),
    );
    item.setAttribute("aria-label", `Inspect step ${valueOrDash(entry.step)}`);

    const header = document.createElement("header");
    const title = document.createElement("strong");
    title.textContent = `STEP ${valueOrDash(entry.step)}`;
    const action = document.createElement("code");
    action.textContent = actions.label;
    action.classList.toggle("intervened", actions.overrode);
    header.append(title, action);

    const reason = document.createElement("p");
    reason.textContent = valueOrDash(
      decision.safety_intervention || decision.reason,
    );
    const feedback = document.createElement("p");
    feedback.textContent = valueOrDash(entry?.feedback);
    item.append(header, reason, feedback);
    item.addEventListener("click", () => activateHistoryEntry(entry));
    item.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activateHistoryEntry(entry);
      }
    });
    historyList.appendChild(item);
  });

  entries.forEach((entry) => {
    if (!historyTableBody) {
      return;
    }
    const actions = historyActions(entry);
    const row = document.createElement("tr");
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-label", `Inspect step ${valueOrDash(entry.step)}`);
    row.classList.toggle(
      "selected",
      selectedHistoryStep !== null
        && String(entry?.step) === String(selectedHistoryStep),
    );
    appendTableCell(row, `#${valueOrDash(entry.step)}`);
    appendTableCell(row, actions.proposed);
    appendTableCell(row, actions.executed);
    appendTableCell(
      row,
      actions.hasDecision ? (actions.overrode ? "OVERRIDE" : "PASS") : "--",
    );
    appendTableCell(row, entry?.feedback);
    row.addEventListener("click", () => activateHistoryEntry(entry));
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activateHistoryEntry(entry);
      }
    });
    historyTableBody.appendChild(row);
  });
}


function renderRuntimeSettings() {
  if (!state) {
    return;
  }
  const runtime = state.runtime || {};
  const connected = Boolean(runtime.connected);
  setText(runtimeStatus, runtimePending
    ? `${state.runtime_backend} / switching`
    : `${state.runtime_backend} / ${connected ? "connected" : "offline"}`);
  setText(
    settingsAgentValue,
    state.agent_model
      ? `${state.agent_backend}:${state.agent_model}`
      : state.agent_backend,
  );
  setText(
    settingsScenarioValue,
    state.scenario?.name || state.scenario_id,
  );
  setText(
    settingsRuntimeValue,
    `${String(state.runtime_backend || "--").toUpperCase()} / ${connected ? "connected" : "offline"}`,
  );
  runtimeStatus.setAttribute("aria-busy", String(runtimePending));
  runtimeStatus.setAttribute("data-connected", String(connected));
  runtimeStatus.style.color = connected ? "#62d57b" : "#ff6b57";
  airsimPanel.classList.toggle("is-hidden", state.runtime_backend !== "airsim");
  airsimPanel.hidden = state.runtime_backend !== "airsim";
  runtimeButtons.forEach((button) => {
    const selected = button.dataset.runtime === state.runtime_backend;
    button.setAttribute("aria-checked", String(selected));
    button.classList.toggle("active", selected);
    button.disabled = runtimePending;
  });

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

  reconnectAirSimBtn.disabled = runtimePending;
  applyWindBtn.disabled = runtimePending || !connected;
}


function renderRuntimeControls() {
  renderRuntimeSettings();
}


function setOperationState(nextState, message) {
  operationState = nextState || "ready";
  operationMessage = valueOrDash(message);
  renderToolbarState();
}


function toggleInspectorWidth() {
  if (!simulator) {
    return;
  }
  inspectorExpanded = !simulator.classList.contains("inspector-expanded");
  simulator.classList.toggle("inspector-expanded", inspectorExpanded);
  renderInspectorToggle();
  try {
    sessionStorage.setItem(
      "uav-inspector-expanded",
      String(inspectorExpanded),
    );
  } catch (error) {
    // Storage can be denied in hardened or private browser contexts.
  }
  window.requestAnimationFrame(onResize);
}


function showHistoryView(show) {
  historyMode = Boolean(show);
  traceView?.classList.toggle("is-hidden", historyMode);
  historyView?.classList.toggle("is-hidden", !historyMode);
  traceView?.setAttribute("aria-hidden", String(historyMode));
  historyView?.setAttribute("aria-hidden", String(!historyMode));

  if (!state) {
    return;
  }
  if (historyMode) {
    renderHistory();
    return;
  }
  const entries = getHistoryEntries();
  const selectedEntry = selectedHistoryStep === null
    ? null
    : entries.find(
      (entry) => String(entry?.step) === String(selectedHistoryStep),
    ) || null;
  renderDecisionTrace(selectedEntry || entries[0] || null);
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


function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (error) {
    return "--";
  }
}


function formatDuration(value) {
  if (value === null || value === undefined || value === "") {
    return "--";
  }
  const number = Number(value);
  return Number.isFinite(number) ? `${number.toFixed(2)} s` : "--";
}


function formatRuntimeValue(runtime) {
  if (isRecord(runtime)) {
    const name = runtime.backend || runtime.source || runtime.name;
    if (!name) {
      return "--";
    }
    if (runtime.connected === null || runtime.connected === undefined) {
      return String(name);
    }
    return `${name} / ${runtime.connected ? "connected" : "offline"}`;
  }
  return valueOrDash(runtime);
}


function formatKinematics(kinematics) {
  if (!isRecord(kinematics) || !Object.keys(kinematics).length) {
    return "--";
  }
  const orientation = isRecord(kinematics.orientation)
    ? kinematics.orientation
    : {};
  const velocity = isRecord(kinematics.linear_velocity)
    ? kinematics.linear_velocity
    : {};
  return [
    `6-DOF: ${valueOrDash(kinematics.source)}`,
    `roll ${formatNumber(orientation.roll)} rad`,
    `pitch ${formatNumber(orientation.pitch)} rad`,
    `yaw ${formatNumber(orientation.yaw)} rad`,
    `v ${formatVector(velocity)}`,
  ].join(" | ");
}


function historyActions(entry) {
  const decision = isRecord(entry?.decision) ? entry.decision : {};
  const hasDecision = Object.keys(decision).length > 0;
  const proposed = decision.proposed_action || decision.action;
  const executed = entry?.action ?? decision.action;
  const overrode = Boolean(
    decision.safety_intervened
      || decision.safety_intervention
      || (proposed && executed && proposed !== executed),
  );
  return {
    hasDecision,
    proposed: valueOrDash(proposed),
    executed: valueOrDash(executed),
    overrode,
    label: proposed && executed && proposed !== executed
      ? `${proposed} -> ${executed}`
      : valueOrDash(executed || proposed),
  };
}


function getHistoryEntries() {
  return Array.isArray(state?.history) ? state.history : [];
}


function activateHistoryEntry(entry) {
  if (!entry || entry.step === null || entry.step === undefined) {
    return;
  }
  selectedHistoryStep = entry.step;
  showHistoryView(false);
  renderHistory();
}


function appendTableCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = valueOrDash(value);
  row.appendChild(cell);
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


function renderInspectorToggle() {
  if (!inspectorToggleBtn) {
    return;
  }
  const label = inspectorExpanded
    ? "Collapse decision inspector"
    : "Expand decision inspector";
  inspectorToggleBtn.setAttribute("aria-label", label);
  inspectorToggleBtn.setAttribute("title", label);
  const icon = inspectorToggleBtn.querySelector("[data-lucide]");
  const iconName = inspectorExpanded
    ? "panel-right-close"
    : "panel-right-open";
  const iconChanged = Boolean(
    icon && icon.getAttribute("data-lucide") !== iconName,
  );
  icon?.setAttribute("data-lucide", iconName);
  if (iconChanged) {
    createLucideIcons();
  }
}


function renderHistoryNavigation() {
  if (!historyViewBtn) {
    return;
  }
  const showingSelectedTrace = selectedHistoryStep !== null && !historyMode;
  const iconChanged = updateCommandButton(
    historyViewBtn,
    showingSelectedTrace ? "Current" : "History",
    showingSelectedTrace ? "arrow-left" : "history",
  );
  if (iconChanged) {
    createLucideIcons();
  }
}


function isCancellationError(message) {
  return /\b(cancelled|canceled|aborted)\b/i.test(String(message || ""));
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
settingsBtn?.addEventListener("click", () => {
  renderRuntimeSettings();
  if (typeof settingsDialog?.showModal === "function") {
    if (!settingsDialog.open) {
      settingsDialog.showModal();
    }
  } else {
    settingsDialog?.setAttribute("open", "");
  }
});
closeSettingsBtn?.addEventListener("click", () => {
  if (typeof settingsDialog?.close === "function") {
    settingsDialog.close();
  } else {
    settingsDialog?.removeAttribute("open");
  }
});
inspectorToggleBtn?.addEventListener("click", toggleInspectorWidth);
historyViewBtn?.addEventListener("click", () => {
  if (selectedHistoryStep !== null && !historyMode) {
    selectedHistoryStep = null;
    showHistoryView(false);
    renderHistory();
    return;
  }
  showHistoryView(true);
});
historyBackBtn?.addEventListener("click", () => {
  selectedHistoryStep = null;
  showHistoryView(false);
  renderHistory();
});
trajectoryBtn?.addEventListener("click", () => {
  trajectoryVisible = !trajectoryVisible;
  if (pathLine) {
    pathLine.visible = trajectoryVisible;
  }
  trajectoryBtn.classList.toggle("active", trajectoryVisible);
  trajectoryBtn.setAttribute("aria-pressed", String(trajectoryVisible));
  trajectoryBtn.setAttribute(
    "aria-label",
    trajectoryVisible ? "Hide trajectory" : "Show trajectory",
  );
  trajectoryBtn.setAttribute(
    "title",
    trajectoryVisible ? "Hide trajectory" : "Show trajectory",
  );
});
window.addEventListener("resize", onResize);

renderInspectorToggle();
trajectoryBtn?.classList.toggle("active", trajectoryVisible);
trajectoryBtn?.setAttribute("aria-pressed", String(trajectoryVisible));
showHistoryView(false);
createLucideIcons();
renderToolbarState();
renderer.setAnimationLoop(animate);
loadState().catch((error) => {
  const message = error instanceof Error ? error.message : String(error);
  setText(feedbackText, message);
  setOperationState("error", message);
});
