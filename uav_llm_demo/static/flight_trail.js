/** Incremental, bounded flight history. Owns its geometries and shared material. */
export class FlightTrail {
  constructor(THREE, toWorld) {
    this.THREE = THREE;
    this.toWorld = toWorld;
    this.object = new THREE.Group();
    this.object.name = "flight_trail";
    this.material = new THREE.MeshBasicMaterial({
      color: 0x19f3eb,
      transparent: true,
      opacity: 0.98,
      depthTest: false,
      depthWrite: false,
      side: THREE.DoubleSide,
      forceSinglePass: true,
    });
    this.radius = 0.14;
    this.radialSegments = 6;
    this.chunkCapacity = 256;
    this.maxPoints = 10000;
    this._coordinates = new Float64Array(this.maxPoints * 3);
    this._chunks = [];
    this._start = 0;
    this._end = 0;
    this._hasHistory = false;
    this._last = new THREE.Vector3();
    this._lastValid = false;
    this._direction = new THREE.Vector3();
    this._normal = new THREE.Vector3();
    this._binormal = new THREE.Vector3();
    this._boundsPoint = new THREE.Vector3();
    this._cos = Array.from({ length: this.radialSegments }, (_, i) => Math.cos(i * Math.PI * 2 / this.radialSegments));
    this._sin = Array.from({ length: this.radialSegments }, (_, i) => Math.sin(i * Math.PI * 2 / this.radialSegments));
    this._disposed = false;
    this._visibleEnd = Infinity;
  }

  /**
   * points is a contiguous window of {x,y,z}; startIndex is its absolute index.
   * Overlap is compared without allocating, so replaced histories are detected.
   * Only appended samples are converted to world coordinates or uploaded.
   * Recreate this object when the coordinate system / map changes.
   */
  update(points, startIndex = 0) {
    if (this._disposed) return;
    if (!Array.isArray(points)) points = [];
    if (!Number.isSafeInteger(startIndex) || startIndex < 0) startIndex = 0;
    const offset = Math.max(0, points.length - this.maxPoints);
    const start = startIndex + offset;
    const end = startIndex + points.length;
    if (!Number.isSafeInteger(end)) throw new RangeError("Flight trail index exceeds the safe integer range");
    let replace = !this._hasHistory || start < this._start || start > this._end || end < this._end;
    if (!replace) {
      const overlapEnd = Math.min(end, this._end);
      for (let index = start; index < overlapEnd; index++) {
        const point = points[index - startIndex];
        const cache = (index % this.maxPoints) * 3;
        if (!Object.is(this._coordinates[cache], this._coordinate(point?.x))
            || !Object.is(this._coordinates[cache + 1], this._coordinate(point?.y))
            || !Object.is(this._coordinates[cache + 2], this._coordinate(point?.z))) {
          replace = true;
          break;
        }
      }
    }
    if (replace || points.length === 0) {
      this._clear();
      this._start = start;
      this._end = start;
      this._hasHistory = points.length > 0;
    }
    this._trim(start);
    if (start >= this._end) this._lastValid = false;
    for (let index = this._end; index < end; index++) {
      const point = points[index - startIndex];
      const x = this._coordinate(point?.x);
      const y = this._coordinate(point?.y);
      const z = this._coordinate(point?.z);
      const cache = (index % this.maxPoints) * 3;
      this._coordinates[cache] = x;
      this._coordinates[cache + 1] = y;
      this._coordinates[cache + 2] = z;
      // Invalid samples break the line; never connect across missing telemetry.
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
        this._lastValid = false;
        continue;
      }
      const world = this.toWorld(point);
      if (!world || !Number.isFinite(world.x) || !Number.isFinite(world.y) || !Number.isFinite(world.z)
          || Math.abs(world.x) >= 1e20 || Math.abs(world.y) >= 1e20 || Math.abs(world.z) >= 1e20) {
        this._lastValid = false;
        continue;
      }
      if (this._lastValid && this._last.distanceToSquared(world) > 1e-12) {
        this._append(this._last, world, index);
      }
      this._last.copy(world);
      this._lastValid = true;
    }
    this._start = start;
    this._end = end;
    this._hasHistory = points.length > 0;
    this._flush();
    this._applyVisibleRanges();
  }

  _coordinate(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : NaN;
  }

  _createChunk() {
    const THREE = this.THREE;
    const verticesPerSegment = this.radialSegments * 2;
    const indicesPerSegment = this.radialSegments * 6;
    const positions = new Float32Array(this.chunkCapacity * verticesPerSegment * 3);
    const indices = new Uint16Array(this.chunkCapacity * indicesPerSegment);
    for (let segment = 0; segment < this.chunkCapacity; segment++) {
      const vertex = segment * verticesPerSegment;
      for (let side = 0; side < this.radialSegments; side++) {
        const next = (side + 1) % this.radialSegments;
        const at = segment * indicesPerSegment + side * 6;
        indices[at] = vertex + side;
        indices[at + 1] = vertex + next;
        indices[at + 2] = vertex + side + this.radialSegments;
        indices[at + 3] = vertex + next;
        indices[at + 4] = vertex + next + this.radialSegments;
        indices[at + 5] = vertex + side + this.radialSegments;
      }
    }
    const geometry = new THREE.BufferGeometry();
    const attribute = new THREE.BufferAttribute(positions, 3);
    attribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute("position", attribute);
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.setDrawRange(0, 0);
    geometry.boundingBox = new THREE.Box3();
    geometry.boundingSphere = new THREE.Sphere();
    const mesh = new THREE.Mesh(geometry, this.material);
    mesh.renderOrder = 5;
    mesh.name = "flight_trail_chunk";
    this.object.add(mesh);
    const chunk = { mesh, geometry, positions, attribute, count: 0, first: 0,
      endIndices: new Float64Array(this.chunkCapacity), dirtyStart: Infinity, dirtyEnd: 0 };
    this._chunks.push(chunk);
    return chunk;
  }

  _append(from, to, endIndex) {
    let chunk = this._chunks[this._chunks.length - 1];
    if (!chunk || chunk.count === this.chunkCapacity) chunk = this._createChunk();
    const direction = this._direction.subVectors(to, from).normalize();
    const normal = this._normal;
    // A stable perpendicular basis, including vertical takeoff and landing.
    if (Math.abs(direction.y) < 0.9) normal.set(-direction.z, 0, direction.x).normalize();
    else normal.set(direction.y, -direction.x, 0).normalize();
    const binormal = this._binormal.crossVectors(direction, normal).normalize();
    const firstComponent = chunk.count * this.radialSegments * 6;
    let at = firstComponent;
    for (let end = 0; end < 2; end++) {
      const center = end ? to : from;
      for (let side = 0; side < this.radialSegments; side++) {
        const u = this.radius * this._cos[side];
        const v = this.radius * this._sin[side];
        chunk.positions[at++] = center.x + normal.x * u + binormal.x * v;
        chunk.positions[at++] = center.y + normal.y * u + binormal.y * v;
        chunk.positions[at++] = center.z + normal.z * u + binormal.z * v;
      }
      const point = this._boundsPoint;
      chunk.geometry.boundingBox.expandByPoint(point.set(center.x - this.radius, center.y - this.radius, center.z - this.radius));
      chunk.geometry.boundingBox.expandByPoint(point.set(center.x + this.radius, center.y + this.radius, center.z + this.radius));
    }
    chunk.endIndices[chunk.count++] = endIndex;
    chunk.dirtyStart = Math.min(chunk.dirtyStart, firstComponent);
    chunk.dirtyEnd = at;
  }

  _trim(start) {
    while (this._chunks.length) {
      const chunk = this._chunks[0];
      while (chunk.first < chunk.count && chunk.endIndices[chunk.first] <= start) chunk.first++;
      if (chunk.first < chunk.count) {
        break;
      }
      this.object.remove(chunk.mesh);
      chunk.geometry.dispose();
      this._chunks.shift();
    }
  }

  _flush() {
    for (const chunk of this._chunks) {
      if (chunk.dirtyStart === Infinity) continue;
      // Multiple telemetry messages may arrive before the next render. Merge
      // pending uploads instead of dropping earlier changes or growing a list.
      let start = chunk.dirtyStart;
      let end = chunk.dirtyEnd;
      for (const range of chunk.attribute.updateRanges) {
        start = Math.min(start, range.start);
        end = Math.max(end, range.start + range.count);
      }
      chunk.attribute.clearUpdateRanges();
      chunk.attribute.addUpdateRange(start, end - start);
      chunk.attribute.needsUpdate = true;
      chunk.geometry.boundingBox.getBoundingSphere(chunk.geometry.boundingSphere);
      chunk.dirtyStart = Infinity;
      chunk.dirtyEnd = 0;
    }
  }

  /** Reveal history only through this absolute point index (inclusive). */
  setVisibleEnd(endAbsoluteIndex = Infinity) {
    if (this._disposed) return;
    const next = endAbsoluteIndex === Infinity ? Infinity
      : Number.isFinite(endAbsoluteIndex) ? Math.floor(endAbsoluteIndex) : -1;
    if (next === this._visibleEnd) return;
    this._visibleEnd = next;
    this._applyVisibleRanges();
  }

  _applyVisibleRanges() {
    const indicesPerSegment = this.radialSegments * 6;
    for (const chunk of this._chunks) {
      let low = chunk.first;
      let high = chunk.count;
      while (low < high) {
        const middle = (low + high) >>> 1;
        if (chunk.endIndices[middle] <= this._visibleEnd) low = middle + 1;
        else high = middle;
      }
      chunk.geometry.setDrawRange(chunk.first * indicesPerSegment, (low - chunk.first) * indicesPerSegment);
      chunk.mesh.visible = low > chunk.first;
    }
  }

  _clear() {
    for (const chunk of this._chunks) {
      this.object.remove(chunk.mesh);
      chunk.geometry.dispose();
    }
    this._chunks.length = 0;
    this._lastValid = false;
    this._hasHistory = false;
  }

  get stats() {
    const segments = this._chunks.reduce((sum, chunk) => sum + chunk.count - chunk.first, 0);
    return { points: this._end - this._start, segments, triangles: segments * this.radialSegments * 2,
      visibleTriangles: this._chunks.reduce((sum, chunk) => sum + chunk.geometry.drawRange.count / 3, 0),
      chunks: this._chunks.length, allocatedSegments: this._chunks.length * this.chunkCapacity };
  }

  dispose() {
    if (this._disposed) return;
    this._clear();
    this.material.dispose();
    this.object.removeFromParent();
    this._start = this._end = 0;
    this._disposed = true;
  }
}
