// Display confirmed control samples on one delayed clock. Never extrapolate a
// vehicle or feed this display state back into navigation or collision checks.
const mix = (a, b, fraction) => a + (b - a) * fraction;
const vector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);

export class MotionTimeline {
  constructor(delaySeconds = 0.35) {
    this.delay = delaySeconds;
    this.reset();
  }

  reset() {
    this.samples = [];
    this.time = null;
    this.lastWall = null;
    this.rate = 1;
    this.buffering = false;
    this.discontinuity = true;
  }

  ingest(samples, now) {
    if (!Array.isArray(samples) || !Number.isFinite(now)) return;
    const incoming = [];
    for (const sample of samples) {
      if (!sample || !Number.isFinite(sample.t) || !vector(sample.position)
        || !vector(sample.velocity) || !Number.isFinite(sample.yaw)) continue;
      if (incoming.length && sample.t <= incoming.at(-1).t) continue;
      const traffic = {};
      for (const [id, values] of Object.entries(sample.traffic || {})) {
        if (Array.isArray(values) && values.length === 6 && values.every(Number.isFinite)) traffic[id] = [...values];
      }
      incoming.push({t: sample.t, position: [...sample.position], velocity: [...sample.velocity],
        yaw: sample.yaw, landed: sample.landed === true, traffic,
        path_index: Number.isInteger(sample.path_index) ? sample.path_index : 0});
    }
    if (!incoming.length || (this.samples.length && incoming.at(-1).t < this.samples.at(-1).t)) return;
    const latest = incoming.at(-1).t;
    const previous = this.samples.at(-1)?.t;
    // A gap larger than the retained telemetry window has no confirmed path.
    // Resynchronize explicitly instead of drawing an invented cross-map segment.
    const gap = previous !== undefined && incoming[0].t > previous + 0.11;
    if (gap) this.reset();
    for (const sample of incoming) {
      const tail = this.samples.at(-1);
      if (!tail || sample.t > tail.t) this.samples.push(sample);
      else if (sample.t === tail.t) this.samples[this.samples.length - 1] = sample;
    }
    if (this.samples.length > 96) this.samples.splice(0, this.samples.length - 96);
    if (this.time === null || this.time < this.samples[0].t || latest - this.time > 1.25) {
      this.time = Math.max(this.samples[0].t, latest - this.delay);
      this.lastWall = now;
      this.rate = 1;
      this.buffering = latest - this.time < this.delay - 1e-6;
      this.discontinuity = true;
    }
  }

  sample(now) {
    if (!this.samples.length || !Number.isFinite(now)) return null;
    const latest = this.samples.at(-1).t;
    // This is a display clock, not a control integrator. Capping long frames
    // would make it fall behind repeatedly and force visible resynchronizations.
    const delta = Math.max(0, now - this.lastWall);
    this.lastWall = now;
    const available = latest - this.time;
    if (this.buffering && available >= this.delay - 1e-6) this.buffering = false;
    if (!this.buffering) {
      // Correct clock drift gently; packet arrival jitter must not become an
      // instantaneous change of animation speed or position.
      const desiredRate = Math.max(0.92, Math.min(1.08, 1 + (available - this.delay) * 0.12));
      this.rate = mix(this.rate, desiredRate, 1 - Math.exp(-2 * delta));
      this.time = Math.min(latest, this.time + delta * this.rate);
      if (this.time >= latest - 1e-9) this.buffering = true;
    }
    const frame = this.at(this.time);
    frame.reset = this.discontinuity;
    frame.buffering = this.buffering;
    frame.delay = latest - this.time;
    this.discontinuity = false;
    return frame;
  }

  at(time) {
    const samples = this.samples;
    if (!samples.length) return null;
    time = Math.max(samples[0].t, Math.min(samples.at(-1).t, time));
    let index = 0;
    while (index + 1 < samples.length && samples[index + 1].t <= time) index++;
    const a = samples[index], b = samples[Math.min(index + 1, samples.length - 1)];
    const fraction = b.t > a.t ? (time - a.t) / (b.t - a.t) : 0;
    const traffic = {};
    for (const [id, values] of Object.entries(a.traffic)) {
      const next = b.traffic[id] || values;
      traffic[id] = values.map((value, axis) => mix(value, next[axis], fraction));
    }
    const yawDelta = Math.atan2(Math.sin(b.yaw - a.yaw), Math.cos(b.yaw - a.yaw));
    return {time, position: a.position.map((value, axis) => mix(value, b.position[axis], fraction)),
      velocity: a.velocity.map((value, axis) => mix(value, b.velocity[axis], fraction)),
      yaw: a.yaw + yawDelta * fraction, landed: a.landed, path_index: a.path_index, traffic};
  }

  preview(horizon = 0.6) {
    if (this.time === null || !this.samples.length) return [];
    const end = Math.min(this.samples.at(-1).t, this.time + horizon);
    // Preserve every control corner, including vertical takeoff and landing.
    return this.samples.filter(sample => sample.t > this.time && sample.t < end)
      .map(sample => sample.position).concat([this.at(end).position]);
  }
}
