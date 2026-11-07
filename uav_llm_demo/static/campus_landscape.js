// Static test scenery. Appearance belongs to the camera image, not the task catalog.
// Every furnishing stays inside its declared footprint/height used by safety checks.
export function addCampusLandscape(THREE, parent, layout, {X, Z, instance, material}) {
  const sphere = new THREE.IcosahedronGeometry(1, 2);
  const cone = new THREE.CylinderGeometry(0, 1, 1, 10);
  const trunk = new THREE.CylinderGeometry(.13, .19, 1, 8);
  const bark = material('#70533c');
  const foliage = {
    broadleaf: ['#3f7846', '#548d4d', '#72a15a'],
    conifer: ['#254f40', '#32634b', '#497956'],
    ginkgo: ['#d1aa36', '#e5c14b', '#a4a345'],
  };
  const leaves = Object.fromEntries(Object.entries(foliage).map(([kind, colors]) =>
    [kind, colors.map(color => material(color, {roughness: 1}))]));
  for (const [index, tree] of (layout.trees || []).entries()) {
    const h = tree.height * 5, r = tree.radius * 3, x = X(tree.x), z = Z(tree.y);
    const mats = leaves[tree.species] || leaves.broadleaf;
    instance('campus_tree_trunks', trunk, bark, x, h * .31, z, 1, h * .62, 1);
    if (tree.species === 'conifer') {
      for (let i = 0; i < 3; i++) instance('campus_tree_conifers', cone, mats[i],
        x, h * (.48 + i * .17), z, r * (1 - i * .25), h * .36, r * (1 - i * .25));
    } else {
      instance('campus_tree_canopies', sphere, mats[index % 3], x, h * .72, z, r * .72, h * .28, r * .72);
      for (const side of [-1, 1]) instance('campus_tree_canopies', sphere, mats[(index + 1) % 3],
        x + side * r * .32, h * .60, z + side * r * .10, r * .62, h * .24, r * .64);
    }
  }

  const box = new THREE.BoxGeometry(1, 1, 1);
  const cylinder = new THREE.CylinderGeometry(.5, .5, 1, 12);
  const roof = new THREE.CylinderGeometry(0, .5, 1, 6);
  const ring = new THREE.TorusGeometry(.36, .065, 12, 48);
  const wheel = new THREE.TorusGeometry(.5, .07, 8, 20);
  const metal = material('#38464a', {metalness: .45, roughness: .45});
  const stone = material('#dad4bf');
  const wood = material('#aa7145');
  const green = material('#538046');
  const glass = material('#6e989c', {metalness: .35, roughness: .25});
  const white = material('#ece7db');
  const blue = material('#286db0');
  const greenBin = material('#3c9369');
  const flower = ['#eccb39', '#cf6392', '#f2e1ca'].map(color => material(color));
  const transform = new THREE.Object3D();

  for (const item of layout.landmarks || []) {
    const group = new THREE.Group();
    group.name = 'campus_feature_' + item.id;
    group.userData.landmarkId = item.id;
    group.userData.cameraOccluder = true;
    group.position.set(X(item.position[0]), 0, Z(item.position[1]));
    group.rotation.y = item.yaw_deg * Math.PI / 180;
    group.scale.set(item.size_m[0], item.size_m[2], item.size_m[1]);
    const accent = material(item.color);
    const batches = new Map();
    // Batch repeated slats, posts and flowers within each object, keeping a
    // separate bounding box per object for the Follow camera and visual oracle.
    function part(geometry, mat, x, y, z, sx, sy, sz, rx = 0, ry = 0, rz = 0) {
      const key = geometry.uuid + mat.uuid;
      if (!batches.has(key)) batches.set(key, {geometry, mat, matrices: []});
      transform.position.set(x, y, z); transform.scale.set(sx, sy, sz);
      transform.rotation.set(rx, ry, rz); transform.updateMatrix();
      batches.get(key).matrices.push(transform.matrix.clone());
    }
    const b = (mat, x,y,z,w,h,d) => part(box,mat,x,y,z,w,h,d);
    const c = (mat, x,y,z,w,h,d) => part(cylinder,mat,x,y,z,w,h,d);
    const post = (x,z,top=.8) => b(metal,x,top/2,z,.035,top,.035);
    const planter = (x,z,w,d) => {
      b(stone,x,.25,z,w,.4,d);
      b(green,x,.49,z,w*.90,.09,d*.88);
      for (let i=0;i<5;i++) for (let j=0;j<2;j++) {
        const px=x+w*(i-2)*.17, pz=z+d*(j-.5)*.45;
        c(green,px,.63,pz,.018,.27,.025);
        part(sphere,flower[(i+j)%3],px,.81,pz,.055,.13,.11);
      }
    };
    switch (item.kind) {
      case 'ring_sculpture':
        b(stone,0,.075,0,.88,.15,.84);
        b(metal,0,.19,0,.22,.12,.25);
        part(ring,accent,0,.57,0,.96,.99,1.25);
        break;
      case 'pavilion':
        c(stone,0,.04,0,.96,.08,.96);
        for (let i=0;i<6;i++) { const a=i*Math.PI/3; post(Math.cos(a)*.34,Math.sin(a)*.34,.72); }
        part(roof,accent,0,.835,0,1,.33,1);
        c(wood,0,.26,0,.38,.04,.38);
        c(metal,0,.16,0,.045,.22,.045);
        break;
      case 'bench':
        for (const x of [-.35,.35]) for (const z of [-.3,.3]) b(metal,x,.24,z,.055,.45,.1);
        for (let i=0;i<4;i++) b(accent,0,.47,-.32+i*.21,.96,.08,.16);
        for (const x of [-.35,.35]) b(metal,x,.72,-.37,.045,.5,.08);
        for (let i=0;i<3;i++) b(accent,0,.66+i*.13,-.36,.96,.09,.13);
        break;
      case 'bins':
        for (const [i,mat] of [blue,greenBin].entries()) {
          const x=(i-.5)*.51;
          b(mat,x,.45,0,.42,.82,.75); b(metal,x,.89,0,.46,.07,.82);
          b(metal,x,.75,.382,.23,.14,.01); b(white,x,.43,.382,.17,.17,.012);
          for (const dx of [-.14,.14]) c(metal,x+dx,.07,0,.055,.1,.65);
        }
        break;
      case 'kiosk':
        b(white,0,.38,0,.85,.72,.80);
        b(glass,0,.49,.405,.68,.33,.012);
        b(wood,0,.30,.44,.8,.05,.1);
        part(roof,accent,0,.85,0,1,.3,1);
        b(accent,0,.66,.416,.73,.04,.02);
        break;
      case 'planter': planter(0,0,.94,.92); break;
      case 'umbrella':
        c(metal,0,.47,0,.025,.9,.025);
        part(roof,accent,0,.88,0,1,.22,1);
        c(wood,0,.32,0,.50,.04,.50);
        for (const x of [-.3,.3]) { c(wood,x,.2,0,.17,.05,.17); c(metal,x,.11,0,.035,.18,.035); }
        break;
      case 'greenhouse':
        b(stone,0,.06,0,.96,.12,.96);
        b(glass,0,.40,0,.88,.57,.88);
        // A pitched opaque glass roof keeps RGB and depth on the same surface.
        for (const side of [-1,1]) part(box,glass,0,.79,side*.22,.88,.035,.53,side*.55);
        for (const x of [-.44,-.22,0,.22,.44]) {
          for (const z of [-.44,.44]) b(white,x,.39,z,.018,.6,.018);
          for (const side of [-1,1]) part(box,white,x,.80,side*.22,.018,.025,.54,side*.55);
        }
        for (const z of [-.44,.44]) b(white,0,.67,z,.90,.025,.025);
        break;
      case 'canopy':
        for (const x of [-.4,.4]) for (const z of [-.37,.37]) post(x,z,.82);
        part(box,accent,0,.87,0,.98,.06,.94,-.12);
        b(wood,0,.21,.25,.8,.04,.18);
        for (const x of [-.32,.32]) post(x,.25,.21);
        break;
      case 'bicycle_rack':
        b(metal,0,.12,0,.96,.045,.88);
        for (let i=0;i<3;i++) {
          const x=(i-1)*.31;
          for (const z of [-.25,.25]) part(wheel,metal,x,.26,z,.40,.44,.40,0,Math.PI/2);
          // Bicycle frames are narrow in x; wheels roll along local z.
          for (const side of [-1,1]) part(box,accent,x,.39,side*.13,.018,.39,.035,side*.72);
          b(accent,x,.40,0,.018,.028,.50);
          part(box,metal,x,.59,-.26,.022,.50,.025,-.15);
          b(metal,x,.83,-.30,.20,.035,.045);
          b(metal,x,.62,.09,.12,.035,.14);
        }
        break;
      default: throw new Error('Unknown campus furnishing: ' + item.kind);
    }
    for (const {geometry, mat, matrices} of batches.values()) {
      const mesh = new THREE.InstancedMesh(geometry,mat,matrices.length);
      matrices.forEach((m,i) => mesh.setMatrixAt(i,m));
      mesh.instanceMatrix.needsUpdate = true;
      mesh.castShadow = mesh.receiveShadow = true;
      mesh.computeBoundingSphere(); group.add(mesh);
    }
    parent.add(group);
  }
}
