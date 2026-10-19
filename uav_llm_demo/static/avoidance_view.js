// Controller rollouts are diagnostic samples, not navigation instructions.
// Render them on the same delayed simulation clock as the aircraft and traffic.
const validPoint = point => point && [point.x, point.y, point.z].every(Number.isFinite);
const copyPath = value => Array.isArray(value) && value.length >= 2 && value.every(validPoint)
  ? value.map(point => ({x: point.x, y: point.y, z: point.z})) : [];
const interpolatePath = (a, b, fraction) => a.length === b.length
  ? a.map((point, index) => Object.fromEntries(["x", "y", "z"].map(axis =>
    [axis, point[axis] + (b[index][axis] - point[axis]) * fraction]))) : a;

export class AvoidanceView {
  constructor(THREE, toWorld) {
    this.THREE = THREE;
    this.toWorld = toWorld;
    this.object = new THREE.Group();
    this.object.name = "avoidance-visualization";
    const makeLine = (name, color, dashed = false) => {
      const options = {color, transparent: true, opacity: .9, depthTest: false, depthWrite: false};
      const material = dashed
        ? new THREE.LineDashedMaterial({...options, dashSize: .5, gapSize: .3})
        : new THREE.LineBasicMaterial(options);
      const line = new THREE.Line(new THREE.BufferGeometry(), material);
      line.name = name;
      line.renderOrder = name === "execution-leg" ? 5 : 6;
      line.visible = false;
      line.raycast = () => {};
      this.object.add(line);
      return line;
    };
    this.executionLine = makeLine("execution-leg", 0x80bdff, true);
    this.nominalLine = makeLine("nominal-rollout", 0xc8c8a9, true);
    this.selectedLine = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      color: 0xff963e, transparent: true, opacity: .95, depthTest: false, depthWrite: false,
      side: THREE.DoubleSide, forceSinglePass: true,
    }));
    this.selectedLine.name = "selected-rollout";
    this.selectedLine.renderOrder = 6;
    this.selectedLine.visible = false;
    this.selectedLine.raycast = () => {};
    this.object.add(this.selectedLine);
    this.tubeDirection = new THREE.Vector3();
    this.tubeNormal = new THREE.Vector3();
    this.tubeBinormal = new THREE.Vector3();
    this.tubeAxis = new THREE.Vector3();
    this.reset();
  }

  reset() {
    this.samples = [];
    this.object.children.forEach(line => { line.visible = false; });
    this.object.visible = false;
  }

  ingest(visualization) {
    const value = visualization;
    if (!value || !Number.isFinite(value.simulation_time)
        || !Number.isInteger(value.sequence) || value.sequence < 0) return;
    if (value.sequence === 0 && !value.episode) { this.reset(); return; }
    const previous = this.samples.at(-1);
    if (previous && (value.sequence <= previous.sequence || value.simulation_time < previous.simulation_time)) return;
    const episode = value.episode && typeof value.episode.id !== "undefined"
      ? {...value.episode, object_ids: (Array.isArray(value.episode.object_ids)
        ? value.episode.object_ids : []).filter(id => typeof id === "string")} : null;
    const sample = {sequence: value.sequence, simulation_time: value.simulation_time,
      active: value.active === true, limited: value.limited === true,
      nominal_path: copyPath(value.nominal_path), selected_path: copyPath(value.selected_path),
      threats: (Array.isArray(value.threats) ? value.threats : []).filter(threat => typeof threat?.id === "string"), episode};
    // An ended episode may omit trajectories. Retain its last measured rollout
    // briefly, without constructing an artificial straight avoidance segment.
    if (!sample.active && episode && previous?.episode?.id === episode.id) {
      if (!sample.nominal_path.length) sample.nominal_path = previous.nominal_path;
      if (!sample.selected_path.length) sample.selected_path = previous.selected_path;
    }
    if (previous?.simulation_time === sample.simulation_time) this.samples.pop();
    this.samples.push(sample);
    if (this.samples.length > 128) this.samples.splice(0, this.samples.length - 128);
  }

  sample(time) {
    if (!Number.isFinite(time)) return null;
    let index = this.samples.length - 1;
    while (index >= 0 && this.samples[index].simulation_time > time + 1e-8) index--;
    if (index < 0) return null;
    const current = this.samples[index];
    const next = this.samples[index + 1];
    const endedAt = current.episode?.ended_at;
    const lingering = !current.active && Number.isFinite(endedAt) && time >= endedAt && time <= endedAt + 3;
    if (!current.active && !lingering) return null;
    // Blend only consecutive rollouts within one active episode. Changes of
    // episode/status wait for their confirmed timestamp, including late packets.
    if (current.active && next?.active && current.episode?.id === next.episode?.id
        && next.simulation_time - current.simulation_time <= .5) {
      const fraction = (time - current.simulation_time) / (next.simulation_time - current.simulation_time);
      return {...current,
        nominal_path: interpolatePath(current.nominal_path, next.nominal_path, fraction),
        selected_path: interpolatePath(current.selected_path, next.selected_path, fraction)};
    }
    return current;
  }

  updateLine(line, points) {
    if (line.isMesh) { this.updateTube(line, points); return; }
    line.userData.sampleCount = points.length;
    if (points.length < 2) { line.visible = false; return; }
    let positions = line.geometry.getAttribute("position");
    if (!positions || positions.count < points.length) {
      positions = new this.THREE.BufferAttribute(new Float32Array(Math.max(points.length, 64) * 3), 3);
      positions.setUsage(this.THREE.DynamicDrawUsage);
      line.geometry.setAttribute("position", positions);
    }
    points.forEach((point, index) => positions.setXYZ(index, point.x, point.y, point.z));
    positions.needsUpdate = true;
    line.geometry.setDrawRange(0, points.length);
    // Unused capacity can contain old positions. Disable culling instead of
    // deriving a bounding sphere from stale vertices outside the draw range.
    line.frustumCulled = false;
    if (line.material.isLineDashedMaterial) {
      let distances = line.geometry.getAttribute("lineDistance");
      if (!distances || distances.count < positions.count) {
        distances = new this.THREE.BufferAttribute(new Float32Array(positions.count), 1);
        distances.setUsage(this.THREE.DynamicDrawUsage);
        line.geometry.setAttribute("lineDistance", distances);
      }
      let distance = 0;
      points.forEach((point, index) => {
        if (index) distance += point.distanceTo(points[index - 1]);
        distances.setX(index, distance);
      });
      distances.needsUpdate = true;
    }
    line.visible = true;
  }

  updateTube(mesh, points) {
    mesh.userData.sampleCount = points.length;
    if (points.length < 2) { mesh.visible = false; return; }
    const sides = 6, radius = .075, required = (points.length - 1) * sides * 2;
    let positions = mesh.geometry.getAttribute("position");
    if (!positions || positions.count < required) {
      const segments = Math.max(points.length - 1, 64);
      positions = new this.THREE.BufferAttribute(new Float32Array(segments * sides * 2 * 3), 3);
      positions.setUsage(this.THREE.DynamicDrawUsage);
      mesh.geometry.setAttribute("position", positions);
      const indices = new Uint16Array(segments * sides * 6);
      for (let segment = 0; segment < segments; segment++) {
        for (let side = 0; side < sides; side++) {
          const a = segment * sides * 2 + side, b = segment * sides * 2 + (side + 1) % sides;
          indices.set([a, b, a + sides, b, b + sides, a + sides], (segment * sides + side) * 6);
        }
      }
      mesh.geometry.setIndex(new this.THREE.BufferAttribute(indices, 1));
    }
    let segment = 0;
    for (let index = 1; index < points.length; index++) {
      const start = points[index - 1], end = points[index];
      this.tubeDirection.subVectors(end, start);
      if (this.tubeDirection.lengthSq() < 1e-12) continue;
      this.tubeDirection.normalize();
      this.tubeAxis.set(Math.abs(this.tubeDirection.y) > .9 ? 1 : 0,
        Math.abs(this.tubeDirection.y) > .9 ? 0 : 1, 0);
      this.tubeNormal.crossVectors(this.tubeDirection, this.tubeAxis).normalize();
      this.tubeBinormal.crossVectors(this.tubeDirection, this.tubeNormal).normalize();
      for (let side = 0; side < sides; side++) {
        const cosine = Math.cos(side / sides * Math.PI * 2) * radius;
        const sine = Math.sin(side / sides * Math.PI * 2) * radius;
        const x = this.tubeNormal.x * cosine + this.tubeBinormal.x * sine;
        const y = this.tubeNormal.y * cosine + this.tubeBinormal.y * sine;
        const z = this.tubeNormal.z * cosine + this.tubeBinormal.z * sine;
        positions.setXYZ(segment * sides * 2 + side, start.x + x, start.y + y, start.z + z);
        positions.setXYZ(segment * sides * 2 + sides + side, end.x + x, end.y + y, end.z + z);
      }
      segment++;
    }
    positions.needsUpdate = true;
    mesh.geometry.setDrawRange(0, segment * sides * 6);
    mesh.frustumCulled = false;
    mesh.visible = segment > 0;
  }

  update(time, uavPosition, waypoint, executing, enabled = true) {
    this.object.visible = enabled;
    if (!enabled) return null;
    const sample = this.sample(time);
    this.updateLine(this.executionLine, executing && !sample?.active && validPoint(waypoint)
      ? [uavPosition, this.toWorld(waypoint)] : []);
    this.updateLine(this.nominalLine, sample ? sample.nominal_path.map(this.toWorld) : []);
    this.updateLine(this.selectedLine, sample ? sample.selected_path.map(this.toWorld) : []);
    this.selectedLine.material.color.set(sample?.limited ? 0xff7657 : 0xff963e);
    this.nominalLine.material.opacity = sample?.active ? .85 : .38;
    this.selectedLine.material.opacity = sample?.active ? .95 : .5;
    return sample;
  }

  dispose() {
    this.object.children.forEach(line => { line.geometry.dispose(); line.material.dispose(); });
    this.samples = [];
  }
}
