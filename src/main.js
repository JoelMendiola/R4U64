import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import { StereoEffect } from 'three/addons/effects/StereoEffect.js';
import './style.css';

const sceneHost = document.querySelector('#scene');
const cameraVideo = document.querySelector('#camera');
const landmarkCanvas = document.querySelector('#landmarks');
const cameraToggle = document.querySelector('#camera-toggle');
const cameraSwitch = document.querySelector('#camera-switch');
const cardboardToggle = document.querySelector('#cardboard-toggle');
const startScreen = document.querySelector('#start-screen');
const landmarkContext = landmarkCanvas.getContext('2d');
const objects = [];
const hands = [];
const grabbedBy = new Map();
const initialTransforms = [];
let handLandmarker;
let cameraStream;
let cameraFacingMode = 'user';
let cameraMirror = true;
let lastVideoTime = -1;
const cameraPinches = new Map();
const cameraHandVisuals = [];
const CAMERA_HAND_REFERENCE_SIZE = 0.2;
const CAMERA_HAND_DEPTH_SCALE = 3;
const CAMERA_HAND_DEPTH_LIMIT = 0.65;
const CAMERA_LANDMARK_DEPTH_SCALE = 1.5;
const MEDIAPIPE_VERSION = '0.10.35';
const handConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];
const pinchPoint = new THREE.Vector3();
const handWorldPosition = new THREE.Vector3();

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
camera.position.set(0, 1.55, 3.8);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(sceneHost.clientWidth, sceneHost.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.xr.enabled = true;
sceneHost.appendChild(renderer.domElement);
const stereoEffect = new StereoEffect(renderer);
let cardboardMode = false;

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    return;
  }
  if (!window.isSecureContext) {
    return;
  }
  cameraToggle.disabled = true;
  cameraSwitch.disabled = true;
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: cameraFacingMode }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
      audio: false,
    });
    cameraFacingMode = cameraStream.getVideoTracks()[0]?.getSettings().facingMode || cameraFacingMode;
    cameraMirror = cameraFacingMode === 'user';
    cameraVideo.srcObject = cameraStream;
    await cameraVideo.play();
    cameraVideo.style.transform = `scaleX(${cameraMirror ? -1 : 1})`;
    if (!handLandmarker) {
      const vision = await FilesetResolver.forVisionTasks(
        `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`,
      );
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.5,
      });
    }
    cameraVideo.classList.add('visible');
    landmarkCanvas.classList.add('visible');
    startScreen.classList.add('hidden');
    cameraSwitch.disabled = false;
    cameraSwitch.textContent = cameraFacingMode === 'user'
      ? 'Cambiar a cámara trasera'
      : 'Cambiar a cámara frontal';
    cameraToggle.textContent = 'Cámara activa';
  } catch (error) {
    cameraStream?.getTracks().forEach((track) => track.stop());
    cameraStream = undefined;
    cameraSwitch.disabled = true;
    cameraToggle.disabled = false;
    cameraToggle.textContent = 'Activar cámara';
    const cameraErrors = {
      NotAllowedError: 'Permiso de cámara denegado',
      NotFoundError: 'No se encontró una cámara',
      NotReadableError: 'La cámara está siendo usada por otra app',
      SecurityError: 'El navegador bloqueó el acceso a la cámara',
    };
    cameraToggle.textContent = cameraErrors[error.name] || 'Activar cámara';
    console.error(error);
  }
}

function setCardboardMode(enabled) {
  cardboardMode = enabled;
  cardboardToggle.textContent = enabled ? 'Salir de Cardboard' : 'Dividir pantalla';
  document.body.classList.toggle('cardboard-mode', enabled);
  stereoEffect.setSize(sceneHost.clientWidth, sceneHost.clientHeight);
}

async function switchToRearCamera() {
  cameraFacingMode = 'environment';
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  cameraVideo.srcObject = null;
  lastVideoTime = -1;
  await startCamera();
}

async function switchCamera() {
  if (cameraFacingMode === 'user') {
    await switchToRearCamera();
    return;
  }
  cameraFacingMode = 'user';
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = undefined;
  cameraVideo.srcObject = null;
  lastVideoTime = -1;
  await startCamera();
}

function cameraPoint(landmark, handedness = 'Right', handDepth = 0) {
  const normalizedX = cameraMirror ? 0.5 - landmark.x : landmark.x - 0.5;
  const landmarkDepth = -(landmark.z || 0) * CAMERA_LANDMARK_DEPTH_SCALE;
  return new THREE.Vector3(
    normalizedX * 3.1,
    2.25 - landmark.y * 2.1,
    0.05 + handDepth + landmarkDepth,
  );
}

function cameraHandDepth(landmarks) {
  const handSize = Math.hypot(
    landmarks[0].x - landmarks[12].x,
    landmarks[0].y - landmarks[12].y,
  );
  return THREE.MathUtils.clamp(
    (handSize - CAMERA_HAND_REFERENCE_SIZE) * CAMERA_HAND_DEPTH_SCALE,
    -CAMERA_HAND_DEPTH_LIMIT,
    CAMERA_HAND_DEPTH_LIMIT,
  );
}

function createCameraHandVisual() {
  const group = new THREE.Group();
   const jointMaterial = new THREE.MeshStandardMaterial({
     color: 0x72f8dd,
     roughness: 0.42,
     metalness: 0.04,
      emissive: 0x202020,
      emissiveIntensity: 0.18,
      depthTest: false,
    });
   const boneMaterial = new THREE.LineBasicMaterial({ color: 0x72f8dd, transparent: true, opacity: 0.9, depthTest: false });
   const palmMaterial = new THREE.MeshStandardMaterial({ color: 0x72f8dd, roughness: 0.35, metalness: 0.05 });
   group.userData.joints = Array.from({ length: 21 }, () => {
      const joint = new THREE.Mesh(new THREE.SphereGeometry(0.02, 10, 8), jointMaterial);
    group.add(joint);
     return joint;
   });
   group.userData.labels = Array.from({ length: 21 }, (_, index) => {
     const canvas = document.createElement('canvas');
      canvas.width = 64;
      canvas.height = 64;
     const context = canvas.getContext('2d');
     context.fillStyle = '#09131d';
     context.beginPath();
     context.arc(32, 32, 25, 0, Math.PI * 2);
     context.fill();
     context.fillStyle = '#72f8dd';
     context.font = 'bold 28px monospace';
     context.textAlign = 'center';
     context.textBaseline = 'middle';
      context.fillText(index, 32, 32);
     const label = new THREE.Sprite(new THREE.SpriteMaterial({
       map: new THREE.CanvasTexture(canvas),
       transparent: true,
       depthTest: false,
     }));
     label.scale.setScalar(0.09);
     group.add(label);
     return label;
   });
   group.userData.bones = handConnections.map(([start, end]) => {
     const bone = new THREE.Line(new THREE.BufferGeometry(), boneMaterial);
     bone.userData.connection = [start, end];
     group.add(bone);
       return bone;
     });
   group.userData.palm = new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), palmMaterial);
    group.add(group.userData.palm);
   group.visible = false;
  scene.add(group);
  cameraHandVisuals.push(group);
  return group;
}

function updateCameraHandVisual(group, landmarks) {
    const handDepth = cameraHandDepth(landmarks);
    const points = landmarks.map((landmark) => cameraPoint(landmark, 'Right', handDepth));
   const palm = new THREE.Vector3()
     .add(points[0]).add(points[5]).add(points[9]).add(points[13]).add(points[17])
     .multiplyScalar(0.2);
   group.userData.joints.forEach((joint, index) => joint.position.copy(points[index]));
   group.userData.labels.forEach((label, index) => {
     label.position.copy(points[index]);
     label.position.x += 0.055;
     label.position.y += 0.055;
   });
   group.userData.bones.forEach((bone) => {
     const [start, end] = bone.userData.connection;
     bone.geometry.setFromPoints([points[start], points[end]]);
   });
    const palmWidth = points[5].distanceTo(points[17]);
   group.userData.palm.position.copy(palm);
   group.userData.palm.scale.set(palmWidth * 0.45, palmWidth * 0.3, palmWidth * 0.1);
   group.visible = true;
 }

function drawLandmarks(landmarks) {
  landmarkCanvas.width = cameraVideo.videoWidth;
  landmarkCanvas.height = cameraVideo.videoHeight;
  landmarkContext.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
  landmarkContext.fillStyle = '#72f8dd';
  for (const hand of landmarks) {
    for (const point of [hand[4], hand[8]]) {
      landmarkContext.beginPath();
       const x = (cameraMirror ? 1 - point.x : point.x) * landmarkCanvas.width;
       landmarkContext.arc(x, point.y * landmarkCanvas.height, 11, 0, Math.PI * 2);
      landmarkContext.fill();
    }
  }
}

function updateCameraHands() {
  if (!handLandmarker || cameraVideo.readyState < 2 || cameraVideo.currentTime === lastVideoTime) return;
  lastVideoTime = cameraVideo.currentTime;
   let result;
   try {
     result = handLandmarker.detectForVideo(cameraVideo, performance.now());
   } catch (error) {
     console.error('Error detectando manos:', error);
     return;
   }
   const seen = new Set();
   drawLandmarks(result.landmarks || []);
    (result.landmarks || []).forEach((landmarks, index) => {
      const source = `camera-${index}`;
      seen.add(source);
       const handedness = result.handednesses?.[index]?.[0]?.categoryName || 'Right';
       const pinching = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y) < 0.075;
      const handDepth = cameraHandDepth(landmarks);
      const handVisual = cameraHandVisuals[index] || createCameraHandVisual();
      updateCameraHandVisual(handVisual, landmarks);
     const point = cameraPoint({
        x: (landmarks[4].x + landmarks[8].x) / 2,
        y: (landmarks[4].y + landmarks[8].y) / 2,
        z: (landmarks[4].z + landmarks[8].z) / 2,
      }, handedness, handDepth);
    if (pinching && !cameraPinches.get(source)) startGrab(source, point);
    if (!pinching && cameraPinches.get(source)) endGrab(source);
     cameraPinches.set(source, pinching);
     const grab = grabbedBy.get(source);
     if (grab) {
       const now = performance.now();
       const elapsed = Math.max((now - grab.lastTime) / 1000, 0.001);
       grab.velocity.copy(point).sub(grab.lastPoint).multiplyScalar(1 / elapsed);
       grab.object.position.copy(point).add(grab.offset);
       grab.lastPoint.copy(point);
       grab.lastTime = now;
     }
  });
  for (const source of cameraPinches.keys()) {
    if (!seen.has(source)) {
      endGrab(source);
      cameraPinches.delete(source);
    }
  }
  cameraHandVisuals.forEach((visual, index) => {
    if (!(result.landmarks || [])[index]) visual.visible = false;
  });
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.2, 0);
controls.enableDamping = true;
controls.enablePan = false;
controls.minDistance = 2.4;
controls.maxDistance = 6;

scene.add(new THREE.HemisphereLight(0xb9d5ff, 0x151528, 2.1));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.2);
keyLight.position.set(-2, 4, 3);
keyLight.castShadow = true;
scene.add(keyLight);
const rimLight = new THREE.PointLight(0x8f68ff, 8, 8);
rimLight.position.set(2, 1.8, -1);
scene.add(rimLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(3.2, 64),
  new THREE.MeshStandardMaterial({ color: 0x10172a, roughness: 0.72, metalness: 0.1 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = 0;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(6, 24, 0x384065, 0x1c2540);
grid.position.y = 0.006;
grid.material.transparent = true;
grid.material.opacity = 0.42;
scene.add(grid);

function createObject(geometry, material, position, rotation = [0, 0, 0]) {
  const mesh = new THREE.Mesh(geometry, material);
  mesh.position.set(...position);
  mesh.rotation.set(...rotation);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  mesh.userData.baseMaterial = material;
  geometry.computeBoundingSphere();
  mesh.userData.radius = geometry.boundingSphere?.radius || 0.35;
  mesh.userData.velocity = new THREE.Vector3();
  mesh.userData.physicsActive = false;
  mesh.userData.home = { position: mesh.position.clone(), quaternion: mesh.quaternion.clone() };
  scene.add(mesh);
  objects.push(mesh);
  initialTransforms.push(mesh.userData.home);
  return mesh;
}

createObject(new THREE.IcosahedronGeometry(0.38, 2), new THREE.MeshStandardMaterial({ color: 0xff745c, roughness: 0.28, metalness: 0.2 }), [-0.9, 1.45, 0]);
createObject(new THREE.BoxGeometry(0.62, 0.62, 0.62), new THREE.MeshStandardMaterial({ color: 0x58d6c3, roughness: 0.24, metalness: 0.35 }), [0, 1.15, -0.18], [0.2, 0.5, 0.15]);
createObject(new THREE.TorusKnotGeometry(0.28, 0.095, 96, 16), new THREE.MeshStandardMaterial({ color: 0xb084ff, roughness: 0.2, metalness: 0.42 }), [0.9, 1.52, 0.05], [0.2, 0, 0.3]);

const handFactory = new XRHandModelFactory();
const controllerFactory = new XRControllerModelFactory();
const rayGeometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, -1)]);
const rayMaterial = new THREE.LineBasicMaterial({ color: 0x8fffea, transparent: true, opacity: 0.72 });

function setupInput(index) {
  const controller = renderer.xr.getController(index);
  controller.add(new THREE.Line(rayGeometry, rayMaterial));
  controller.addEventListener('selectstart', (event) => startGrab(event.target, event.target.getWorldPosition(pinchPoint)));
  controller.addEventListener('selectend', (event) => endGrab(event.target));
  scene.add(controller);

  const grip = renderer.xr.getControllerGrip(index);
  grip.add(controllerFactory.createControllerModel(grip));
  scene.add(grip);

  const hand = renderer.xr.getHand(index);
  hand.userData.handedness = index === 0 ? 'left' : 'right';
  hand.add(handFactory.createHandModel(hand, 'mesh'));
  hand.addEventListener('pinchstart', (event) => startGrab(event.target, getPinchPosition(event.target)));
  hand.addEventListener('pinchend', (event) => endGrab(event.target));
  hands.push(hand);
  scene.add(hand);
}

function getPinchPosition(hand) {
  const indexTip = hand.joints?.['index-finger-tip'];
  const thumbTip = hand.joints?.['thumb-tip'];
  if (!indexTip || !thumbTip) return hand.getWorldPosition(pinchPoint);
  indexTip.getWorldPosition(handWorldPosition);
  thumbTip.getWorldPosition(pinchPoint);
  return pinchPoint.lerp(handWorldPosition, 0.5);
}

function startGrab(source, point) {
  let nearest = null;
  let nearestDistance = 0.28;
  for (const object of objects) {
    const distance = object.getWorldPosition(handWorldPosition).distanceTo(point);
    if (distance < nearestDistance) {
      nearest = object;
      nearestDistance = distance;
    }
  }
  if (!nearest) return;
   grabbedBy.set(source, {
     object: nearest,
     offset: nearest.position.clone().sub(point),
     lastPoint: point.clone(),
     lastTime: performance.now(),
     velocity: new THREE.Vector3(),
   });
   nearest.userData.physicsActive = false;
   nearest.userData.velocity.set(0, 0, 0);
  nearest.userData.grabbed = true;
  nearest.userData.baseMaterial.emissive.set(0x39245f);
  nearest.userData.baseMaterial.emissiveIntensity = 0.8;
}

function endGrab(source) {
  const grab = grabbedBy.get(source);
  if (!grab) return;
  grab.object.userData.grabbed = false;
   grab.object.userData.baseMaterial.emissiveIntensity = 0;
  grab.object.userData.velocity.copy(grab.velocity);
  grab.object.userData.physicsActive = true;
   grabbedBy.delete(source);
}

function updateHands() {
  for (const hand of hands) {
    const grab = grabbedBy.get(hand);
    if (!grab) continue;
    const point = getPinchPosition(hand);
     const now = performance.now();
     const elapsed = Math.max((now - grab.lastTime) / 1000, 0.001);
     grab.velocity.copy(point).sub(grab.lastPoint).multiplyScalar(1 / elapsed);
     grab.object.position.copy(point).add(grab.offset);
     grab.lastPoint.copy(point);
     grab.lastTime = now;
     grab.object.rotation.y += 0.012;
   }
}

function updatePhysics(delta) {
  for (const object of objects) {
    if (!object.userData.physicsActive || [...grabbedBy.values()].some((grab) => grab.object === object)) continue;
    const velocity = object.userData.velocity;
    velocity.y -= 9.8 * delta;
    object.position.addScaledVector(velocity, delta);
    if (object.position.y < object.userData.radius) {
      object.position.y = object.userData.radius;
      if (velocity.y < 0) velocity.y = -velocity.y * 0.72;
      velocity.x *= 0.86;
      velocity.z *= 0.86;
      if (Math.abs(velocity.y) < 0.08) velocity.y = 0;
    }
    object.rotation.x += velocity.z * delta;
    object.rotation.z -= velocity.x * delta;
  }
}

renderer.xr.addEventListener('sessionstart', () => {
  controls.enabled = false;
});
renderer.xr.addEventListener('sessionend', () => {
  controls.enabled = true;
  grabbedBy.clear();
});

setupInput(0);
setupInput(1);
document.body.appendChild(VRButton.createButton(renderer, { optionalFeatures: ['hand-tracking'] }));
cameraToggle.addEventListener('click', startCamera);
cameraSwitch.addEventListener('click', switchCamera);
cardboardToggle.addEventListener('click', () => setCardboardMode(!cardboardMode));

document.querySelector('#reset').addEventListener('click', () => {
  objects.forEach((object, index) => {
    object.position.copy(initialTransforms[index].position);
    object.quaternion.copy(initialTransforms[index].quaternion);
     object.userData.baseMaterial.emissiveIntensity = 0;
     object.userData.velocity.set(0, 0, 0);
     object.userData.physicsActive = false;
  });
});

window.addEventListener('resize', () => {
  const { clientWidth, clientHeight } = sceneHost;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight);
  stereoEffect.setSize(clientWidth, clientHeight);
});

const handClock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  controls.update();
  updateCameraHands();
   const delta = handClock.getDelta();
    updatePhysics(delta);
  updateHands();
  for (const object of objects) {
    if (!object.userData.grabbed) object.rotation.y += 0.0025;
  }
   if (cardboardMode) stereoEffect.render(scene, camera);
   else renderer.render(scene, camera);
});
