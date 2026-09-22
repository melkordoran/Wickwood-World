// Wickwood atlas: a browser approximation of the world, drawn from the published RWX models, textures and terrain.
// Coordinates follow Active Worlds directly: +x is west, +z is north, +y is up (right-handed, like Three.js).
import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (s) => document.querySelector(s);
const statusBox = $('#status');
const setStatus = (text, error = false) => {
  statusBox.textContent = text;
  statusBox.className = error ? 'error' : '';
};

const EYE = 1.8;
const LIGHT_POOL = 10;
const VIEWS = {
  native: { draw: 170 },
  clear: { draw: 800 },
  map: { draw: 2600 },
};

async function fetchData(path, binary = false) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return binary ? response.arrayBuffer() : response.json();
}

const srgb = (r, g, b) => new THREE.Color().setRGB(r / 255, g / 255, b / 255, THREE.SRGBColorSpace);
const toLinear = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
const rotY = (v, yaw) => [v[0] * Math.cos(yaw) + v[2] * Math.sin(yaw), v[1], -v[0] * Math.sin(yaw) + v[2] * Math.cos(yaw)];

function awCoords(x, y, z, yawDeg) {
  const ns = z / 10;
  const we = x / 10;
  const yaw = ((Math.round(yawDeg) % 360) + 360) % 360;
  return `${Math.abs(ns).toFixed(2)}${ns >= 0 ? 'N' : 'S'} ${Math.abs(we).toFixed(2)}${we >= 0 ? 'W' : 'E'} ${(y / 10).toFixed(2)}a ${yaw}`;
}

async function main() {
  const canvas = $('#view');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  } catch (e) {
    throw new Error('This browser could not start WebGL, which the atlas needs.');
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  setStatus('Loading atlas data');
  const [index, modelsBuf, objectsBuf, heightsBuf, texturesBuf] = await Promise.all([
    fetchData('data/index.json'),
    fetchData('data/models.bin', true),
    fetchData('data/objects.bin', true),
    fetchData('data/terrain-h.i16', true),
    fetchData('data/terrain-t.u8', true),
  ]);
  setStatus('Building the forest');

  const A = index.attributes;
  const num = (k, d = 0) => (A[k] === undefined || A[k] === '' ? d : parseFloat(A[k]));
  const SIZE = index.size;
  const MIN = index.minCell;
  const heights = new Int16Array(heightsBuf);
  const cellTextures = new Uint8Array(texturesBuf);
  const maxAniso = Math.min(8, renderer.capabilities.getMaxAnisotropy());

  // ---------------------------------------------------------------- scene, sky, fog, lights
  const scene = new THREE.Scene();
  const fogColor = srgb(num('FogRed'), num('FogGreen'), num('FogBlue'));
  const nativeFog = new THREE.Fog(fogColor, num('FogMinimum'), num('FogMaximum'));
  const hazeFog = new THREE.Fog(fogColor, 250, 1600);
  scene.fog = nativeFog;
  scene.background = fogColor.clone();

  const camera = new THREE.PerspectiveCamera(60, 1, 0.1, 9000);
  const skyTop = srgb(num('SkyTopRed'), num('SkyTopGreen'), num('SkyTopBlue'));
  const sides = ['North', 'East', 'South', 'West'];
  const skyHorizon = new THREE.Color(0, 0, 0);
  for (const s of sides) skyHorizon.add(srgb(num(`Sky${s}Red`), num(`Sky${s}Green`), num(`Sky${s}Blue`)).multiplyScalar(0.25));
  const skyBottom = srgb(num('SkyBottomRed'), num('SkyBottomGreen'), num('SkyBottomBlue'));
  const sky = new THREE.Mesh(
    new THREE.SphereGeometry(8000, 32, 16),
    new THREE.ShaderMaterial({
      uniforms: { top: { value: skyTop }, horizon: { value: skyHorizon }, bottom: { value: skyBottom } },
      vertexShader: 'varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
      fragmentShader: `uniform vec3 top; uniform vec3 horizon; uniform vec3 bottom; varying vec3 vDir;
        void main() { float h = normalize(vDir).y;
          vec3 c = h > 0.0 ? mix(horizon, top, pow(clamp(h, 0.0, 1.0), 0.55)) : mix(horizon, bottom, pow(clamp(-h, 0.0, 1.0), 0.55));
          gl_FragColor = vec4(c, 1.0);
          #include <colorspace_fragment>
        }`,
      side: THREE.BackSide, depthWrite: false, fog: false,
    }),
  );
  sky.renderOrder = -10;
  scene.add(sky);

  const ambient = new THREE.AmbientLight(srgb(num('AmbientLightRed'), num('AmbientLightGreen'), num('AmbientLightBlue')), Math.PI);
  scene.add(ambient);
  const moon = new THREE.DirectionalLight(srgb(num('LightRed'), num('LightGreen'), num('LightBlue')), Math.PI);
  const lightDir = new THREE.Vector3(num('LightX'), num('LightY', -1), num('LightZ')).normalize();
  moon.position.copy(lightDir).multiplyScalar(-2000);
  scene.add(moon, moon.target);

  // ---------------------------------------------------------------- textures
  const loader = new THREE.TextureLoader();
  const textureCache = new Map();
  function texture(name, flipY = false) {
    const key = `${name}|${flipY}`;
    if (!textureCache.has(key)) {
      const t = loader.load(`../textures/${name}.jpg`);
      t.colorSpace = THREE.SRGBColorSpace;
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.anisotropy = maxAniso;
      t.flipY = flipY;
      textureCache.set(key, t);
    }
    return textureCache.get(key);
  }

  // ---------------------------------------------------------------- terrain (one texture per 10 m cell, as in the world)
  const heightAt = (ix, iz) => heights[Math.min(SIZE - 1, Math.max(0, iz)) * SIZE + Math.min(SIZE - 1, Math.max(0, ix))] / 100;
  function groundAt(x, z) {
    const fx = x / 10 - MIN;
    const fz = z / 10 - MIN;
    const ix = Math.floor(fx);
    const iz = Math.floor(fz);
    const tx = fx - ix;
    const tz = fz - iz;
    const h00 = heightAt(ix, iz), h10 = heightAt(ix + 1, iz), h01 = heightAt(ix, iz + 1), h11 = heightAt(ix + 1, iz + 1);
    return (h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz;
  }
  const terrainMaterials = [0, 1, 2, 3, 4].map((i) => new THREE.MeshLambertMaterial({ map: texture(`terrain${i}`, true) }));
  const CHUNK = 128;
  for (let cz = 0; cz < SIZE - 1; cz += CHUNK) {
    for (let cx = 0; cx < SIZE - 1; cx += CHUNK) {
      const x1 = Math.min(cx + CHUNK, SIZE - 1);
      const z1 = Math.min(cz + CHUNK, SIZE - 1);
      const w = x1 - cx + 1;
      const h = z1 - cz + 1;
      const pos = new Float32Array(w * h * 3);
      const nor = new Float32Array(w * h * 3);
      const uv = new Float32Array(w * h * 2);
      for (let j = 0; j < h; j++) {
        for (let i = 0; i < w; i++) {
          const ix = cx + i, iz = cz + j, v = j * w + i;
          const x = (MIN + ix) * 10, z = (MIN + iz) * 10;
          pos.set([x, heightAt(ix, iz), z], v * 3);
          const n = new THREE.Vector3(heightAt(ix - 1, iz) - heightAt(ix + 1, iz), 20, heightAt(ix, iz - 1) - heightAt(ix, iz + 1)).normalize();
          nor.set([n.x, n.y, n.z], v * 3);
          uv.set([x / 10, z / 10], v * 2);
        }
      }
      const buckets = [[], [], [], [], []];
      for (let j = 0; j < h - 1; j++) {
        for (let i = 0; i < w - 1; i++) {
          const a = j * w + i, b = a + 1, d = a + w, c = d + 1;
          const t = Math.min(4, cellTextures[(cz + j) * SIZE + cx + i]);
          buckets[t].push(a, c, b, a, d, c);
        }
      }
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geometry.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
      geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
      const indices = [];
      buckets.forEach((list, t) => {
        if (list.length) {
          geometry.addGroup(indices.length, list.length, t);
          for (const k of list) indices.push(k);
        }
      });
      geometry.setIndex(indices);
      geometry.computeBoundingSphere();
      scene.add(new THREE.Mesh(geometry, terrainMaterials));
    }
  }

  // ---------------------------------------------------------------- water plane (drawn under terrain, as WaterUnderTerrain)
  if ((A.WaterEnabled || 'Y') === 'Y') {
    const waterMap = loader.load(`../textures/${A.WaterTexture || 'wk_water'}.jpg`);
    waterMap.colorSpace = THREE.SRGBColorSpace;
    waterMap.wrapS = waterMap.wrapT = THREE.RepeatWrapping;
    waterMap.anisotropy = maxAniso;
    waterMap.repeat.set(640, 640);
    const water = new THREE.Mesh(
      new THREE.PlaneGeometry(SIZE * 10, SIZE * 10),
      new THREE.MeshLambertMaterial({
        color: srgb(num('WaterRed'), num('WaterGreen'), num('WaterBlue')).lerp(new THREE.Color(1, 1, 1), 0.35),
        map: waterMap, transparent: true, opacity: num('WaterOpacity', 255) / 255, depthWrite: false,
      }),
    );
    water.rotation.x = -Math.PI / 2;
    water.position.y = num('WaterLevel');
    water.renderOrder = 1;
    scene.add(water);
  }

  // ---------------------------------------------------------------- models
  const materials = index.materials.map((d) => {
    const color = new THREE.Color().setRGB(d.color[0], d.color[1], d.color[2], THREE.SRGBColorSpace);
    const transparent = d.opacity < 1;
    if (d.glow) return new THREE.MeshBasicMaterial({ vertexColors: true, transparent, opacity: d.opacity, depthWrite: !transparent });
    if (d.sign) return new THREE.MeshBasicMaterial({ color: 0x22170f });
    const m = new THREE.MeshLambertMaterial({ color, transparent, opacity: d.opacity, depthWrite: !transparent });
    if (d.texture) m.map = texture(d.texture);
    return m;
  });
  const models = index.models.map((m) => {
    const parts = m.parts.map((p) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(modelsBuf, p.pos, p.count * 3), 3));
      g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(modelsBuf, p.uv, p.count * 2), 2));
      if (p.prelight !== undefined) {
        const c = new Float32Array(modelsBuf, p.prelight, p.count * 3).map(toLinear);
        g.setAttribute('color', new THREE.BufferAttribute(c, 3));
      }
      g.computeVertexNormals();
      g.computeBoundingSphere();
      return { geometry: g, material: materials[p.material], sign: p.signBounds !== undefined, signBounds: p.signBounds };
    });
    const radius = Math.hypot(...m.bounds.size) / 2;
    return { ...m, parts, radius, objects: [] };
  });

  const N = index.objects;
  const dv = new DataView(objectsBuf);
  const objModel = new Uint16Array(N);
  const objPos = new Float32Array(N * 3);
  const objYaw = new Float32Array(N);
  for (let k = 0; k < N; k++) {
    const o = k * index.objectRecordBytes;
    objModel[k] = dv.getUint16(o, true);
    objPos[k * 3] = dv.getFloat32(o + 4, true);
    objPos[k * 3 + 1] = dv.getFloat32(o + 8, true);
    objPos[k * 3 + 2] = dv.getFloat32(o + 12, true);
    objYaw[k] = dv.getFloat32(o + 16, true);
    models[objModel[k]].objects.push(k);
  }
  const specialsByObject = new Map(index.specials.map((s) => [s.object, s]));

  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const one = new THREE.Vector3(1, 1, 1);
  const p3 = new THREE.Vector3();
  const m4 = new THREE.Matrix4();
  const placeMatrix = (k) => m4.compose(p3.set(objPos[k * 3], objPos[k * 3 + 1], objPos[k * 3 + 2]), q.setFromAxisAngle(up, objYaw[k]), one);

  for (const m of models) {
    const matrices = new THREE.InstancedBufferAttribute(new Float32Array(Math.max(1, m.objects.length) * 16), 16);
    matrices.setUsage(THREE.DynamicDrawUsage);
    m.matrices = matrices;
    m.meshes = [];
    for (const part of m.parts) {
      if (part.sign) continue; // sign faces are drawn per object with their own text
      const mesh = new THREE.InstancedMesh(part.geometry, part.material, Math.max(1, m.objects.length));
      mesh.instanceMatrix = matrices;
      mesh.count = 0;
      mesh.frustumCulled = false;
      scene.add(mesh);
      m.meshes.push(mesh);
    }
  }

  // ---------------------------------------------------------------- signs with their text
  function signTexture(text, fg, bg, aspect) {
    const W = 512;
    const H = Math.max(64, Math.min(1024, Math.round(W / Math.max(0.25, aspect))));
    const c = document.createElement('canvas');
    c.width = W;
    c.height = H;
    const g = c.getContext('2d');
    g.fillStyle = `#${bg}`;
    g.fillRect(0, 0, W, H);
    g.fillStyle = `#${fg}`;
    g.textAlign = 'center';
    g.textBaseline = 'middle';
    const words = text.split(/\s+/);
    for (let size = Math.floor(H * 0.42); size >= 12; size -= 2) {
      g.font = `600 ${size}px Georgia, "Times New Roman", serif`;
      const lines = [];
      let line = '';
      for (const w of words) {
        const trial = line ? `${line} ${w}` : w;
        if (g.measureText(trial).width > W * 0.88 && line) {
          lines.push(line);
          line = w;
        } else {
          line = trial;
        }
      }
      lines.push(line);
      const lh = size * 1.15;
      if (lines.length * lh <= H * 0.86 && lines.every((l) => g.measureText(l).width <= W * 0.9)) {
        lines.forEach((l, i) => g.fillText(l, W / 2, H / 2 + (i - (lines.length - 1) / 2) * lh));
        break;
      }
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.flipY = false;
    t.anisotropy = maxAniso;
    return t;
  }
  const signMeshes = [];
  for (const m of models) {
    const signPart = m.parts.find((p) => p.sign);
    if (!signPart) continue;
    const size = signPart.signBounds.size;
    for (const k of m.objects) {
      const s = specialsByObject.get(k)?.sign;
      const material = s
        ? new THREE.MeshBasicMaterial({ map: signTexture(s.text, s.color, s.bcolor, size[0] / Math.max(0.01, size[1])) })
        : signPart.material;
      const mesh = new THREE.Mesh(signPart.geometry, material);
      placeMatrix(k).decompose(mesh.position, mesh.quaternion, mesh.scale);
      mesh.userData.object = k;
      scene.add(mesh);
      signMeshes.push(mesh);
    }
  }

  // ---------------------------------------------------------------- lanterns: glow sprites for every light, real lights for the nearest
  const lights = index.specials.filter((s) => s.light).map((s) => {
    const center = models[s.model].glowCenter || [0, 1, 0];
    const off = rotY(center, s.yaw);
    return { ...s.light, x: s.x + off[0], y: s.y + off[1], z: s.z + off[2], color: new THREE.Color(`#${s.light.color}`) };
  });
  const glowCanvas = document.createElement('canvas');
  glowCanvas.width = glowCanvas.height = 64;
  {
    const g = glowCanvas.getContext('2d');
    const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)');
    grad.addColorStop(0.25, 'rgba(255,255,255,0.55)');
    grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad;
    g.fillRect(0, 0, 64, 64);
  }
  const glowPos = new Float32Array(lights.length * 3);
  const glowCol = new Float32Array(lights.length * 3);
  lights.forEach((l, i) => {
    glowPos.set([l.x, l.y, l.z], i * 3);
    glowCol.set([l.color.r * l.brightness, l.color.g * l.brightness, l.color.b * l.brightness], i * 3);
  });
  const glowGeometry = new THREE.BufferGeometry();
  glowGeometry.setAttribute('position', new THREE.BufferAttribute(glowPos, 3));
  glowGeometry.setAttribute('color', new THREE.BufferAttribute(glowCol, 3));
  const glowSprites = new THREE.Points(glowGeometry, new THREE.PointsMaterial({
    size: 2.4, map: new THREE.CanvasTexture(glowCanvas), vertexColors: true, transparent: true, depthWrite: false,
    blending: THREE.AdditiveBlending, sizeAttenuation: true,
  }));
  scene.add(glowSprites);
  const pool = Array.from({ length: LIGHT_POOL }, () => {
    const l = new THREE.PointLight(0xffffff, 0, 20, 2);
    scene.add(l);
    return l;
  });
  let poolAssigned = [];

  // ---------------------------------------------------------------- trails overlay
  const trailGroup = new THREE.Group();
  for (const r of index.routes) {
    const pts = r.points.map(([x, z, y]) => new THREE.Vector3(x, (y ?? groundAt(x, z)) + 0.5, z));
    trailGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0xffc766, fog: true })));
  }
  trailGroup.visible = false;
  scene.add(trailGroup);

  // ---------------------------------------------------------------- places panel and labels
  const labels = $('#labels');
  const places = index.destinations.map((d) => {
    const el = document.createElement('div');
    el.className = 'label';
    el.textContent = d.name;
    labels.appendChild(el);
    return { ...d, el };
  });
  const list = $('#places');
  for (const d of places) {
    const li = document.createElement('li');
    const b = document.createElement('button');
    b.type = 'button';
    b.innerHTML = '<span class="name"></span><span class="tp"></span><span class="desc"></span>';
    b.querySelector('.name').textContent = d.name;
    b.querySelector('.tp').textContent = d.teleport;
    b.querySelector('.desc').textContent = d.description || '';
    b.addEventListener('click', () => goTo(d));
    li.appendChild(b);
    list.appendChild(li);
  }

  // ---------------------------------------------------------------- controls, views, culling
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.12;
  controls.maxDistance = 6000;
  controls.minDistance = 0.5;
  controls.maxPolarAngle = Math.PI * 0.94;
  let view = 'native';
  let drawDistance = VIEWS.native.draw;
  let lastCull = new THREE.Vector3(1e9, 0, 0);
  let drawn = 0;
  let trianglesDrawn = 0;

  function cull(force = false) {
    if (!force && lastCull.distanceToSquared(camera.position) < 64) return;
    lastCull.copy(camera.position);
    const cx = camera.position.x;
    const cz = camera.position.z;
    drawn = 0;
    trianglesDrawn = 0;
    for (const m of models) {
      const arr = m.matrices.array;
      const reach = (drawDistance + m.radius) ** 2;
      let c = 0;
      for (const k of m.objects) {
        const dx = objPos[k * 3] - cx;
        const dz = objPos[k * 3 + 2] - cz;
        if (dx * dx + dz * dz > reach) continue;
        placeMatrix(k).toArray(arr, c * 16);
        c++;
      }
      m.matrices.needsUpdate = true;
      for (const mesh of m.meshes) mesh.count = c;
      drawn += c;
      trianglesDrawn += c * m.triangles;
    }
    for (const s of signMeshes) s.visible = s.position.distanceTo(camera.position) < drawDistance + 20;
    // nearest lanterns become real point lights
    const lit = $('#lights').checked;
    const near = lit ? lights.map((l, i) => [i, (l.x - cx) ** 2 + (l.z - cz) ** 2]).filter(([, d]) => d < (drawDistance + 30) ** 2)
      .sort((a, b) => a[1] - b[1]).slice(0, LIGHT_POOL) : [];
    poolAssigned = near.map(([i]) => lights[i]);
    pool.forEach((pl, i) => {
      const l = poolAssigned[i];
      if (!l) {
        pl.intensity = 0;
        return;
      }
      pl.position.set(l.x, l.y, l.z);
      pl.color.copy(l.color);
      pl.distance = l.radius * 1.4;
      pl.userData.base = l.brightness * 42;
      pl.userData.flicker = l.flicker;
      pl.intensity = pl.userData.base;
    });
    glowSprites.visible = lit;
  }

  function setView(name, recentre = true) {
    view = name;
    for (const b of document.querySelectorAll('.seg button')) b.setAttribute('aria-pressed', String(b.dataset.view === name));
    scene.fog = name === 'native' ? nativeFog : name === 'clear' ? hazeFog : null;
    // The map view brightens the world's night lighting so terrain and paths stay legible from above.
    ambient.intensity = name === 'map' ? Math.PI * 4 : Math.PI;
    moon.intensity = name === 'map' ? Math.PI * 2.5 : Math.PI;
    $('#mapNote').hidden = name !== 'map';
    if (name === 'map' && !$('#trails').checked) {
      $('#trails').checked = true;
      trailGroup.visible = true;
    }
    drawDistance = VIEWS[name].draw;
    $('#draw').value = String(drawDistance);
    $('#drawOut').textContent = `${drawDistance} m`;
    if (recentre && name === 'map') {
      camera.position.set(0, 3200, -2200);
      controls.target.set(0, 0, 150);
    } else if (recentre) {
      const t = controls.target;
      goTo({ x: t.x, z: t.z, y: groundAt(t.x, t.z), yaw: currentYaw() });
    }
    controls.update();
    cull(true);
  }

  function currentYaw() {
    const dx = controls.target.x - camera.position.x;
    const dz = controls.target.z - camera.position.z;
    return THREE.MathUtils.radToDeg(Math.atan2(dx, dz));
  }

  function goTo(d) {
    const yaw = THREE.MathUtils.degToRad(d.yaw || 0);
    const dir = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
    const ground = Math.max(d.y ?? -1e9, groundAt(d.x, d.z));
    if (view === 'native') {
      camera.position.set(d.x, ground + EYE, d.z);
      controls.target.set(d.x + dir.x * 12, groundAt(d.x + dir.x * 12, d.z + dir.z * 12) + EYE * 0.8, d.z + dir.z * 12);
    } else if (view === 'clear') {
      camera.position.set(d.x - dir.x * 55, ground + 32, d.z - dir.z * 55);
      controls.target.set(d.x + dir.x * 15, ground + 3, d.z + dir.z * 15);
    } else {
      camera.position.set(d.x, ground + 700, d.z - 520);
      controls.target.set(d.x, ground, d.z);
    }
    controls.update();
    cull(true);
  }

  for (const b of document.querySelectorAll('.seg button')) b.addEventListener('click', () => setView(b.dataset.view));
  $('#draw').addEventListener('input', (e) => {
    drawDistance = Number(e.target.value);
    $('#drawOut').textContent = `${drawDistance} m`;
    cull(true);
  });
  $('#lights').addEventListener('change', () => cull(true));
  $('#trails').addEventListener('change', (e) => { trailGroup.visible = e.target.checked; });
  $('#labelsToggle').addEventListener('change', (e) => { labels.style.display = e.target.checked ? '' : 'none'; });
  const panel = $('#panel');
  $('#panelToggle').addEventListener('click', (e) => {
    const collapsed = panel.classList.toggle('collapsed');
    e.target.setAttribute('aria-expanded', String(!collapsed));
  });

  const keys = new Set();
  window.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLInputElement) return;
    keys.add(e.key.toLowerCase());
  });
  window.addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
  window.addEventListener('blur', () => keys.clear());

  function move(dt) {
    const f = (keys.has('w') || keys.has('arrowup') ? 1 : 0) - (keys.has('s') || keys.has('arrowdown') ? 1 : 0);
    const r = (keys.has('d') || keys.has('arrowright') ? 1 : 0) - (keys.has('a') || keys.has('arrowleft') ? 1 : 0);
    const v = (keys.has('r') ? 1 : 0) - (keys.has('f') ? 1 : 0);
    if (!f && !r && !v) return;
    const speed = (keys.has('shift') ? 28 : 7) * (view === 'map' ? 12 : 1) * dt;
    const fwd = new THREE.Vector3().subVectors(controls.target, camera.position).setY(0);
    if (fwd.lengthSq() < 1e-6) fwd.set(0, 0, 1);
    fwd.normalize();
    const right = new THREE.Vector3(-fwd.z, 0, fwd.x);
    const delta = fwd.multiplyScalar(f * speed).add(right.multiplyScalar(r * speed));
    delta.y = v * speed;
    camera.position.add(delta);
    controls.target.add(delta);
    const floor = groundAt(camera.position.x, camera.position.z) + (view === 'native' && !v ? EYE : 0.4);
    if (view === 'native' && !v) {
      const dy = floor - camera.position.y;
      camera.position.y += dy;
      controls.target.y += dy;
    } else if (camera.position.y < floor) {
      const dy = floor - camera.position.y;
      camera.position.y += dy;
      controls.target.y += dy;
    }
  }

  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resize);
  resize();

  const entry = places.find((d) => d.teleport === A.EntryPoint) || places[0];
  goTo(entry);

  const clock = new THREE.Timer();
  let fps = 60;
  const coords = $('#coords');
  const stats = $('#stats');
  const tmp = new THREE.Vector3();
  let statTimer = 1;
  function frame() {
    clock.update();
    const dt = Math.min(0.1, clock.getDelta());
    move(dt);
    controls.update();
    cull();
    sky.position.copy(camera.position);
    const t = clock.getElapsed();
    pool.forEach((pl, i) => {
      if (pl.userData.flicker) pl.intensity = pl.userData.base * (0.82 + 0.18 * Math.sin(t * 13.1 + i * 1.7) * Math.sin(t * 7.3 + i));
    });
    renderer.render(scene, camera);
    fps = fps * 0.95 + (dt > 0 ? 1 / dt : 60) * 0.05;
    const w = window.innerWidth;
    const h = window.innerHeight;
    const showLabels = $('#labelsToggle').checked;
    for (const d of places) {
      tmp.set(d.x, groundAt(d.x, d.z) + 4, d.z);
      const dist = tmp.distanceTo(camera.position);
      tmp.project(camera);
      const visible = showLabels && tmp.z < 1 && Math.abs(tmp.x) < 1.1 && Math.abs(tmp.y) < 1.1 && (view !== 'native' || dist < 400);
      d.el.style.display = visible ? '' : 'none';
      if (visible) d.el.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * w}px, ${(-tmp.y * 0.5 + 0.5) * h}px) translate(-50%, -100%)`;
    }
    statTimer += dt;
    if (statTimer > 0.25) {
      statTimer = 0;
      const ground = groundAt(camera.position.x, camera.position.z);
      coords.textContent = awCoords(camera.position.x, camera.position.y, camera.position.z, currentYaw());
      stats.textContent = `${drawn.toLocaleString()} of ${N.toLocaleString()} objects drawn, ${(trianglesDrawn / 1000).toFixed(0)}k triangles, `
        + `${poolAssigned.length} lanterns lighting, ${Math.round(fps)} fps. Eye ${(camera.position.y - ground).toFixed(1)} m above ground.`;
    }
    requestAnimationFrame(frame);
  }
  statusBox.classList.add('done');
  window.__atlas = { index, drawnObjects: () => drawn, triangles: () => trianglesDrawn, view: () => view, goTo, setView, camera, controls, scene, renderer, groundAt };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  console.error(e);
  setStatus(`The atlas could not start: ${e.message}`, true);
});
