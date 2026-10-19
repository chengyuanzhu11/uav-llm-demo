// Camera motion consumes rendered positions only; it never predicts vehicle motion.
const BASE = [9, 7, 12];
const MIN_BOOM = 0.35;
const LOOKAHEAD_MARGIN = 2;
const CAMERA_MARGIN = 0.18;

export class FollowCameraRig {
  constructor(THREE) {
    this.THREE = THREE;
    this.position = new THREE.Vector3();
    this.direction = new THREE.Vector3(...BASE).normalize();
    this.preferredDirection = this.direction.clone();
    this.baseLength = Math.hypot(...BASE);
    this.length = this.baseLength;
    this.ray = new THREE.Ray();
    this.hit = new THREE.Vector3();
    this.obstacles = null;
    this.expanded = [];
    this.physical = [];
    this.elapsed = 0;
    this.nextChoice = 0;
    this.radialVelocity = 0;
    this.turnVelocity = 0;
    this.initialized = false;
    this._diagnostics = { emergencyClamps: 0, sweepClamps: 0, directionChanges: 0 };
  }

  get diagnostics() { return { ...this._diagnostics, boomLength: this.length }; }

  setObstacles(obstacles) {
    if (obstacles === this.obstacles) return;
    this.obstacles = obstacles || [];
    this.expanded = this.obstacles.map(box => box.clone().expandByScalar(LOOKAHEAD_MARGIN));
    this.physical = this.obstacles.map(box => box.clone().expandByScalar(CAMERA_MARGIN));
  }

  // The inflated envelope provides warning before a building corner enters view.
  // An origin already inside that envelope may still look away from the building.
  freeLength(origin, direction, length, boxes, originals = boxes, margin = LOOKAHEAD_MARGIN) {
    this.ray.set(origin, direction);
    let distance = length;
    for (let i = 0; i < boxes.length; i += 1) {
      const box = boxes[i];
      if (box.containsPoint(origin)) {
        if (originals[i].containsPoint(origin)) return 0;
        if (!this.ray.intersectBox(originals[i], this.hit)) continue;
        distance = Math.min(distance, Math.max(0, this.hit.distanceTo(origin) - margin));
      } else if (this.ray.intersectBox(box, this.hit)) {
        distance = Math.min(distance, this.hit.distanceTo(origin));
      }
    }
    return distance;
  }

  previewPoints(target, previewTargets) {
    const points = [target];
    let previous = target;
    for (const next of previewTargets || []) {
      if (!next || ![next.x, next.y, next.z].every(Number.isFinite)) continue;
      // Check segments too: a pair of individually clear samples can straddle a corner.
      const count = Math.max(1, Math.ceil(previous.distanceTo(next) / 0.6));
      for (let i = 1; i <= count && points.length < 96; i += 1) {
        points.push(previous.clone().lerp(next, i / count));
      }
      previous = next;
      if (points.length >= 96) break;
    }
    return points;
  }

  clearance(points, direction, length = this.baseLength) {
    let distance = length;
    for (const point of points) {
      distance = Math.min(distance,
        this.freeLength(point, direction, length, this.expanded, this.obstacles));
    }
    return distance >= length ? length : Math.max(MIN_BOOM, distance - 0.1);
  }

  candidates() {
    const { Vector3 } = this.THREE;
    const candidates = [new Vector3(...BASE).normalize()];
    for (const height of [7, 12, 20]) {
      for (let i = 0; i < 8; i += 1) {
        const angle = i * Math.PI / 4;
        candidates.push(new Vector3(15 * Math.sin(angle), height, 15 * Math.cos(angle)).normalize());
      }
    }
    candidates.push(new Vector3(0, 1, 0));
    return candidates;
  }

  chooseDirection(points, initializing = false) {
    const available = this.clearance(points, this.preferredDirection);
    if (!initializing && (available >= 6 || this.elapsed < this.nextChoice)) return;
    let choice = this.preferredDirection;
    let score = -Infinity;
    for (const candidate of this.candidates()) {
      const distance = this.clearance(points, candidate);
      // A new direction must offer useful space; this prevents tie-driven oscillation.
      if (!initializing && distance < Math.max(8, available + 3)) continue;
      if (!initializing) {
        let corridorClear = true;
        for (let j = 1; j <= 8; j += 1) {
          const intermediate = this.direction.clone().lerp(candidate, j / 8).normalize();
          if (this.clearance(points, intermediate) < Math.min(this.length, 5)) {
            corridorClear = false;
            break;
          }
        }
        if (!corridorClear) continue;
      }
      const candidateScore = distance - 5 * (1 - candidate.dot(this.direction));
      if (candidateScore > score) { score = candidateScore; choice = candidate; }
    }
    if (choice.angleTo(this.preferredDirection) > 0.01) {
      this.preferredDirection.copy(choice);
      if (!initializing) this._diagnostics.directionChanges += 1;
    }
    this.nextChoice = this.elapsed + 2;
  }

  reset(target, obstacles = []) {
    this.setObstacles(obstacles);
    this.elapsed = 0;
    this.nextChoice = 0;
    this.radialVelocity = 0;
    this.turnVelocity = 0;
    this._diagnostics = { emergencyClamps: 0, sweepClamps: 0, directionChanges: 0 };
    this.direction.set(...BASE).normalize();
    this.preferredDirection.copy(this.direction);
    if (this.clearance([target], this.direction) < this.baseLength - 0.2) {
      this.chooseDirection([target], true);
      this.direction.copy(this.preferredDirection);
    }
    this.length = Math.min(this.baseLength, this.clearance([target], this.direction));
    this.position.copy(target).addScaledVector(this.direction, this.length);
    this.initialized = true;
    return this.position;
  }

  update(target, delta, obstacles = [], previewTargets = []) {
    if (!this.initialized) this.reset(target, obstacles);
    this.setObstacles(obstacles);
    if (!Number.isFinite(delta) || delta <= 0) return this.position;
    const dt = Math.min(delta, 0.1);
    this.elapsed += dt;
    const points = this.previewPoints(target, previewTargets);
    this.chooseDirection(points);

    const angle = this.direction.angleTo(this.preferredDirection);
    const desiredTurnSpeed = Math.min(0.8, Math.sqrt(2 * 2.4 * angle));
    this.turnVelocity += Math.max(-2.4 * dt, Math.min(2.4 * dt, desiredTurnSpeed - this.turnVelocity));
    const turn = Math.min(1, this.turnVelocity * dt / Math.max(angle, 1e-9));
    const nextDirection = this.direction.clone().lerp(this.preferredDirection, turn).normalize();
    // Reserve the full near-term turn corridor before rotating a long boom around a wall.
    let desiredLength = this.clearance(points, this.direction);
    for (let i = 1; i <= 4; i += 1) {
      const intermediate = this.direction.clone().lerp(this.preferredDirection, i / 4).normalize();
      desiredLength = Math.min(desiredLength, this.clearance(points, intermediate));
    }
    const error = desiredLength - this.length;
    const stoppingSpeed = Math.sqrt(2 * 60 * Math.abs(error));
    const desiredSpeed = Math.sign(error) * Math.min(error < 0 ? 24 : 8, stoppingSpeed, Math.abs(error) * (error < 0 ? 4 : 1.4));
    const previousSpeed = this.radialVelocity;
    this.radialVelocity += Math.max(-60 * dt, Math.min(60 * dt, desiredSpeed - this.radialVelocity));
    let nextLength = this.length + (previousSpeed + this.radialVelocity) * 0.5 * dt;
    if (nextLength < MIN_BOOM || nextLength > this.baseLength) {
      nextLength = Math.max(MIN_BOOM, Math.min(this.baseLength, nextLength));
      this.radialVelocity = 0;
    }
    const next = target.clone().addScaledVector(nextDirection, nextLength);

    // Ordinary motion is already clear thanks to the preview envelope. Retain two
    // final physical guards for abrupt resets / absent lookahead / newly added walls.
    const physicalLength = this.freeLength(target, nextDirection, nextLength, this.physical, this.obstacles, CAMERA_MARGIN);
    if (physicalLength < nextLength - 1e-6) {
      next.copy(target).addScaledVector(nextDirection, Math.max(0, physicalLength - 0.03));
      this._diagnostics.emergencyClamps += 1;
      this.radialVelocity = 0;
    }
    const step = next.clone().sub(this.position);
    const stepLength = step.length();
    if (stepLength > 1e-8) {
      const swept = this.freeLength(this.position, step.divideScalar(stepLength), stepLength,
        this.physical, this.obstacles, CAMERA_MARGIN);
      if (swept < stepLength - 1e-6) {
        next.copy(this.position).addScaledVector(step, Math.max(0, swept - 0.03));
        this._diagnostics.sweepClamps += 1;
        this.radialVelocity = 0;
      }
    }
    this.position.copy(next);
    this.direction.copy(next).sub(target);
    this.length = this.direction.length();
    if (this.length > 1e-8) this.direction.divideScalar(this.length);
    else this.direction.copy(this.preferredDirection);
    return this.position;
  }
}
