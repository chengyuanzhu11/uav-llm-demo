/** Export the actual browser scenery with Three's maintained glTF exporter. */
const {chromium} = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');

async function main() {
  const [input, output] = process.argv.slice(2);
  if (!input || !output) throw new Error('Usage: node export_gazebo_visual.cjs scene.json scenery.glb');
  const root = path.resolve(__dirname, '..');
  const payload = JSON.parse(fs.readFileSync(input, 'utf8'));
  const server = http.createServer((request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    if (pathname === '/') {
      response.setHeader('Content-Type', 'text/html');
      response.end('<!doctype html><script type="importmap">{"imports":{"three":"/static/vendor/three.module.js"}}</script>');
      return;
    }
    const prefix = pathname.startsWith('/static/') ? 'uav_llm_demo/static/'
      : pathname.startsWith('/three/') ? 'node_modules/three/' : null;
    const base = prefix && path.resolve(root, prefix);
    const filename = base && path.resolve(base, '.' + pathname.slice(pathname.indexOf('/', 1)));
    if (!filename || !filename.startsWith(base + path.sep) || !fs.existsSync(filename)) {
      response.writeHead(404).end(); return;
    }
    response.setHeader('Content-Type', 'text/javascript');
    fs.createReadStream(filename).pipe(response);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  try {
    browser = await chromium.launch({headless: true,
      ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE ? {executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE} : {}),
      args: ['--no-sandbox']});
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    const exported = await page.evaluate(async ({scenario, objects}) => {
      const THREE = await import('/static/vendor/three.module.js');
      const {GLTFExporter} = await import('/three/examples/jsm/exporters/GLTFExporter.js');
      const {mergeGeometries} = await import('/three/examples/jsm/utils/BufferGeometryUtils.js');
      const source = new THREE.Group();
      const half = (scenario.grid_size - 1) / 2;
      if (scenario.campus_layout) {
        const {buildSemanticCampus} = await import('/static/semantic_campus.js');
        buildSemanticCampus(THREE, source, scenario);
      }
      for (const item of objects) {
        // Match the browser: detailed campus geometry replaces coarse display cells.
        if (scenario.campus_layout && item.name !== 'ground' && item.name !== 'base_pad') continue;
        const size = item.size;
        const geometry = item.shape === 'ellipsoid'
          ? new THREE.SphereGeometry(1, 24, 16).scale(size.x / 2, size.z / 2, size.y / 2)
          : item.shape === 'cylinder'
            ? new THREE.CylinderGeometry(size.x / 2, size.x / 2, size.z, 20)
            : new THREE.BoxGeometry(size.x, size.z, size.y);
        const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({color: item.color, roughness: .9}));
        mesh.name = item.name;
        mesh.position.set((item.position.x - half) * 3, item.position.z * 5, (item.position.y - half) * 3);
        source.add(mesh);
      }
      source.updateMatrixWorld(true);
      const world = new THREE.Group();
      // Browser east/up/south -> Gazebo east/north/up, in metres at map origin.
      const toGazebo = new THREE.Matrix4().makeRotationX(Math.PI / 2);
      toGazebo.setPosition(half * 3, -half * 3, 0);
      let meshCount = 0, triangleCount = 0;
      const batches = new Map();
      source.traverseVisible(object => {
        // Labels, task overlays and camera-facing sprites are not physical scenery.
        if (!object.isMesh) return;
        const count = object.isInstancedMesh ? object.count : 1;
        for (let index = 0; index < count; index++) {
          const transform = object.matrixWorld.clone();
          if (object.isInstancedMesh) {
            const instance = new THREE.Matrix4(); object.getMatrixAt(index, instance);
            transform.multiply(instance);
          }
          meshCount++;
          // Bake transforms and batch by material. Thousands of tiny GL buffers
          // are costly (and unstable on WSLg's D3D12 backend); keep the same triangles.
          const geometry = object.geometry.index ? object.geometry.toNonIndexed() : object.geometry.clone();
          geometry.applyMatrix4(toGazebo.clone().multiply(transform));
          const groups = Array.isArray(object.material) ? geometry.groups
            : [{start: 0, count: geometry.attributes.position.count, materialIndex: 0}];
          for (const group of groups) {
            const material = Array.isArray(object.material) ? object.material[group.materialIndex] : object.material;
            const part = new THREE.BufferGeometry();
            for (const [name, attribute] of Object.entries(geometry.attributes)) {
              const values = attribute.array.slice(group.start * attribute.itemSize,
                (group.start + group.count) * attribute.itemSize);
              part.setAttribute(name, new THREE.BufferAttribute(values, attribute.itemSize, attribute.normalized));
            }
            const key = material.uuid + ':' + Object.keys(part.attributes).sort().join(',');
            if (!batches.has(key)) batches.set(key, {material, parts: []});
            batches.get(key).parts.push(part);
          }
          triangleCount += (object.geometry.index?.count || object.geometry.attributes.position.count) / 3;
        }
      });
      for (const {material, parts} of batches.values()) {
        const geometry = mergeGeometries(parts, false);
        if (!geometry) throw new Error('Cannot merge scenery attributes');
        world.add(new THREE.Mesh(geometry, material));
      }
      const bounds = new THREE.Box3().setFromObject(world);
      const binary = await new GLTFExporter().parseAsync(world, {binary: true, maxTextureSize: 2048});
      const data = await new Promise(resolve => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result.split(',')[1]);
        reader.readAsDataURL(new Blob([binary]));
      });
      return {data, meshCount, drawMeshCount: world.children.length, triangleCount,
        bounds: [bounds.min.toArray(), bounds.max.toArray()]};
    }, payload);
    fs.mkdirSync(path.dirname(path.resolve(output)), {recursive: true});
    fs.writeFileSync(output, Buffer.from(exported.data, 'base64'));
    delete exported.data;
    console.log(JSON.stringify(exported));
  } finally {
    if (browser) await browser.close();
    await new Promise(resolve => server.close(resolve));
  }
}
main().catch(error => {console.error(error.message); process.exitCode = 1;});
