// Camera motion consumes rendered positions only; it never predicts vehicle motion.
const BASE = [9, 7, 12];
const MIN_BOOM = 0.35;
const LOOKAHEAD_MARGIN = 2;
const CAMERA_MARGIN = 0.18;
const EXTENSION_DELAY = 1.2;

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
    this.angularVelocity = new THREE.Vector3();
    this.heldLength = this.length;
    this.extensionClearSince = null;
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

  clearance(points, direction, length = this.baseLength, physical = false) {
    let distance = length;
    for (const point of points) {
      distance = Math.min(distance,
        this.freeLength(point, direction, length, physical ? this.physical : this.expanded,
          this.obstacles, physical ? CAMERA_MARGIN : LOOKAHEAD_MARGIN));
    }
    return distance >= length ? length : Math.max(MIN_BOOM, distance - 0.1);
  }

  turnClearance(points, candidate, length) {
    const axis = this.direction.clone().cross(candidate);
    const angle = Math.atan2(axis.length(), this.direction.dot(candidate));
    axis.normalize();
    const count = Math.max(1, Math.ceil(angle * length / 0.5));
    let available = length;
    const directions = [];
    for (let i = 0; i <= count; i += 1) {
      const direction = this.direction.clone().applyAxisAngle(axis, angle * i / count);
      available = Math.min(available, this.clearance(points, direction, length, true));
      directions.push(direction);
    }
    // A blocked long sightline already determines a smaller radius. Testing
    // the hypothetical long camera arc inside that wall would incorrectly
    // discard all the usable space before it.
    if (available < length - 1e-6) return available;
    let previous = directions[0].clone().multiplyScalar(length);
    for (let i = 1; i < directions.length; i += 1) {
      const direction = directions[i];
      const offset = direction.clone().multiplyScalar(length);
      // Visibility at the arc samples alone does not protect the camera's
      // movement between them, particularly near a thin building corner.
      const segment = offset.clone().sub(previous);
      const distance = segment.length();
      if (distance > 1e-8) {
        segment.divideScalar(distance);
        for (const point of points) {
          const origin = point.clone().add(previous);
          if (this.freeLength(origin, segment, distance, this.physical,
            this.obstacles, CAMERA_MARGIN) < distance - 1e-6) return 0;
        }
      }
      previous = offset;
    }
    return available;
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
    const available = Math.min(this.clearance(points, this.preferredDirection),
      this.clearance(points, this.direction));
    if (!initializing && available >= this.baseLength - 0.2) return;
    if (!initializing && this.elapsed < this.nextChoice
        && this.clearance(points, this.direction, this.length, true) >= this.length) return;
    if (!initializing && this.clearance(points, this.preferredDirection) >= this.baseLength
        && this.turnClearance(points, this.preferredDirection, this.length) >= this.length) {
      this.nextChoice = this.elapsed + 0.25;
      return;
    }
    let choice = this.preferredDirection;
    const candidates = [this.preferredDirection, ...this.candidates()]
      .map(direction => ({ direction, distance: this.clearance(points, direction, this.baseLength, initializing) }))
      .sort((a, b) => b.direction.dot(this.direction) - a.direction.dot(this.direction));
    // First find a connected orbit with the normal radius. Retraction is only
    // considered when no such direction is reachable at the current radius.
    let full = false;
    for (const candidate of candidates) {
      if (candidate.distance < this.baseLength) continue;
      if (!initializing && this.turnClearance(points, candidate.direction, this.length) < this.length) continue;
      choice = candidate.direction;
      full = true;
      break;
    }
    if (!full) {
      let score = available;
      for (const candidate of candidates) {
        const distance = initializing ? candidate.distance : Math.min(candidate.distance,
          this.turnClearance(points, candidate.direction, Math.min(this.length, candidate.distance)));
        const candidateScore = distance - 2 * (1 - candidate.direction.dot(this.direction));
        if (candidateScore > score + 0.5) { score = candidateScore; choice = candidate.direction; }
      }
    }
    if (choice.angleTo(this.preferredDirection) > 0.01) {
      this.preferredDirection.copy(choice);
      if (!initializing) this._diagnostics.directionChanges += 1;
    }
    this.nextChoice = this.elapsed + 0.25;
  }

  reset(target, obstacles = []) {
    this.setObstacles(obstacles);
    this.elapsed = 0;
    this.nextChoice = 0;
    this.radialVelocity = 0;
    this.angularVelocity.set(0, 0, 0);
    this._diagnostics = { emergencyClamps: 0, sweepClamps: 0, directionChanges: 0 };
    this.direction.set(...BASE).normalize();
    this.preferredDirection.copy(this.direction);
    if (this.clearance([target], this.direction) < this.baseLength - 0.2) {
      this.chooseDirection([target], true);
      this.direction.copy(this.preferredDirection);
    }
    this.length = Math.min(this.baseLength, this.clearance([target], this.direction, this.baseLength, true));
    this.heldLength = this.length;
    this.extensionClearSince = null;
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

    const axis = this.direction.clone().cross(this.preferredDirection);
    const angle = Math.atan2(axis.length(), this.direction.dot(this.preferredDirection));
    const desiredTurnSpeed = Math.min(0.8, Math.sqrt(2 * 2.4 * angle), angle * 4);
    const previousAngularVelocity = this.angularVelocity.clone();
    const desiredAngularVelocity = axis.normalize().multiplyScalar(desiredTurnSpeed);
    this.angularVelocity.add(desiredAngularVelocity.sub(this.angularVelocity).clampLength(0, 2.4 * dt));
    // Preserve the turning velocity vector when choosing a different route.
    // Keeping only its magnitude would instantly change the camera's velocity.
    const rotation = previousAngularVelocity.add(this.angularVelocity).multiplyScalar(dt * 0.5);
    const nextDirection = this.direction.clone().applyAxisAngle(rotation.clone().normalize(), rotation.length()).normalize();
    // The warning envelope selects an orbit early. Only the immediate physical
    // corridor may shorten the boom; a distant turn must not cause zooming now.
    const safeLength = this.turnClearance(points, nextDirection, this.baseLength);
    if (safeLength < this.heldLength) {
      this.heldLength = Math.max(MIN_BOOM, safeLength);
      this.extensionClearSince = null;
    } else if (safeLength >= this.heldLength + 0.2 && angle < 0.1) {
      this.extensionClearSince ??= this.elapsed;
      if (this.elapsed - this.extensionClearSince >= EXTENSION_DELAY) this.heldLength = safeLength;
    } else {
      this.extensionClearSince = null;
    }
    const desiredLength = this.heldLength;
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
    if (physicalLength < nextLength - 1e-6 || this.length < nextLength - 1e-6) {
      this.heldLength = Math.min(this.heldLength, this.length);
      this.extensionClearSince = null;
    }
    if (this.length > 1e-8) this.direction.divideScalar(this.length);
    else this.direction.copy(this.preferredDirection);
    return this.position;
  }
}
