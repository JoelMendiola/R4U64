import * as THREE from 'three';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';
import { Hand } from 'kalidokit';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { ARButton } from 'three/addons/webxr/ARButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';
import { XRHandModelFactory } from 'three/addons/webxr/XRHandModelFactory.js';
import { StereoEffect } from 'three/addons/effects/StereoEffect.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import * as SkeletonUtils from 'three/addons/utils/SkeletonUtils.js';
import rightHandModelUrl from '../hand-rigged-r.glb';
import leftHandModelUrl from '../hand-rigged-l.glb';
import './style.css';

const sceneHost = document.querySelector('#scene');
const cameraVideo = document.querySelector('#camera');
const landmarkCanvas = document.querySelector('#landmarks');
const cameraToggle = document.querySelector('#camera-toggle');
const cameraSwitch = document.querySelector('#camera-switch');
const cardboardToggle = document.querySelector('#cardboard-toggle');
const controlBar = document.querySelector('.controls');
const startScreen = document.querySelector('#start-screen');
const calibration = document.querySelector('#calibration');
const rotationInputs = {
  Left: ['x', 'y', 'z'].map((axis) => document.querySelector(`#rotation-left-${axis}`)),
  Right: ['x', 'y', 'z'].map((axis) => document.querySelector(`#rotation-right-${axis}`)),
};
const rotationValues = {
  Left: ['x', 'y', 'z'].map((axis) => document.querySelector(`#rotation-left-${axis}-value`)),
  Right: ['x', 'y', 'z'].map((axis) => document.querySelector(`#rotation-right-${axis}-value`)),
};
const rotationCoordinates = {
  Left: document.querySelector('#rotation-left-coordinates'),
  Right: document.querySelector('#rotation-right-coordinates'),
};
const positionInputs = {
  Left: document.querySelector('#position-left'),
  Right: document.querySelector('#position-right'),
};
const positionValues = {
  Left: document.querySelector('#position-left-value'),
  Right: document.querySelector('#position-right-value'),
};
const landmarkContext = landmarkCanvas.getContext('2d');
const objects = [];
const hands = [];
const grabbedBy = new Map();
const initialTransforms = [];
let handLandmarker;
let cameraStream;
let cameraStarting = false;
let cameraFacingMode = 'user';
let cameraMirror = true;
let lastVideoTime = -1;
let xrSessionActive = false;
const cameraPinches = new Map();
const cameraGrabPoints = new Map();
const cameraHandVisuals = [];
const handModelTemplates = { Left: null, Right: null };
const modelAnimations = new WeakMap();
const handMixers = [];
let calibrationModel;
let leftCalibrationModel;
const HAND_MODEL_SCALE = 1.15;
const HAND_MODEL_TILT = 0.50;
const HAND_MODEL_X_OFFSET = 0;
const HAND_MODEL_Y_OFFSET = 0;
const HAND_MODEL_Z_OFFSET = 0;
const HAND_MODEL_ROTATIONS = {
  Left: [9, 34, 0],
  Right: [-16, -16, 0],
};
const LANDMARK_Y_ROTATION_LIMIT = THREE.MathUtils.degToRad(10);
const LANDMARK_RIG_MAP = {
  0: 'radius_ulna', 1: 'thumb_trapez', 2: 'thumb_meta', 3: 'thumb_prox', 4: 'thumb_dist',
  5: 'index_meta', 6: 'index_prox', 7: 'index_midd', 8: 'index_dist',
  9: 'midd_meta', 10: 'midd_prox', 11: 'midd_midd', 12: 'midd_dist',
  13: 'ring_meta', 14: 'ring_prox', 15: 'ring_midd', 16: 'ring_dist',
  17: 'pinky_prox', 18: 'pinky_prox', 19: 'pinky_midd', 20: 'pinky_dist',
};
let interactiveHand;
let handGrabbed = false;
const handGrabOffset = new THREE.Vector3();
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

function attachHandModel(group, source) {
  group.userData.model = SkeletonUtils.clone(source);
  group.userData.model.frustumCulled = false;
  group.userData.model.traverse((part) => {
    if (part.isMesh || part.isSkinnedMesh) part.frustumCulled = false;
  });
  group.userData.rigBones = [];
  const thumbOffsets = { trapez: 0, meta: 1, prox: 2, dist: 3 };
  const fingerOffsets = { meta: 0, prox: 1, midd: 2, dist: 3 };
  group.userData.model.traverse((part) => {
    if (!part.isBone) return;
    const name = part.name.toLowerCase().replace(/[^a-z0-9]/g, '');
    const target = name.match(/^(thumb|index|midd|middle|ring|pinky|little)(trapez|meta|prox|midd|dist)$/);
    if (!target) return;
    const finger = ['thumb', 'index', 'midd', 'middle', 'ring', 'pinky', 'little'].indexOf(target[1]);
    if (finger < 0) return;
    const offset = (finger === 0 ? thumbOffsets : fingerOffsets)[target[2]];
    if (offset === undefined) return;
    const base = finger === 0 ? 0 : 5 + (finger - 1) * 4;
    group.userData.rigBones.push({ bone: part, start: base + offset, end: base + offset + 1 });
  });
  group.add(group.userData.model);
  const animations = modelAnimations.get(source) || [];
  group.userData.mixer = new THREE.AnimationMixer(group.userData.model);
  group.userData.poseAction = animations.length
    ? group.userData.mixer.clipAction(animations.find((clip) => clip.name === 'Pose_OK') || animations[0])
    : null;
  if (group.userData.poseAction) {
    group.userData.poseAction.setLoop(THREE.LoopOnce, 1);
    group.userData.poseAction.clampWhenFinished = true;
  }
  handMixers.push(group.userData.mixer);
  group.userData.model.updateMatrixWorld(true);
  group.userData.rigBones.forEach((entry) => {
    entry.restLocalQuaternion = entry.bone.quaternion.clone();
    entry.restQuaternion = entry.bone.getWorldQuaternion(new THREE.Quaternion());
    const child = entry.bone.children.find((part) => part.isBone);
    entry.restDirection = child
      ? child.getWorldPosition(new THREE.Vector3()).sub(entry.bone.getWorldPosition(new THREE.Vector3()))
      : new THREE.Vector3();
  });
  const rootBone = group.userData.model.getObjectByName('radius_ulna');
  group.userData.rootRestPosition = rootBone?.getWorldPosition(new THREE.Vector3()) || new THREE.Vector3();
  group.userData.joints.forEach((joint) => { joint.visible = true; });
  group.userData.bones.forEach((bone) => { bone.visible = true; });
}

function prepareHandTemplate(gltfScene, animations) {
  let hasMesh = false;
  gltfScene.traverse((part) => { if (part.isMesh || part.isSkinnedMesh) hasMesh = true; });
  if (!hasMesh) {
    return null;
  }
  // Normalizamos una sola vez: contenido escalado a ~1 unidad y centrado en el origen.
  const bounds = new THREE.Box3().setFromObject(gltfScene);
  const size = bounds.getSize(new THREE.Vector3());
  const center = bounds.getCenter(new THREE.Vector3());
  const innerScale = 1 / Math.max(size.x, size.y, size.z);
  // world = position + scale * v  →  para centrar: position = -scale * center
  gltfScene.scale.setScalar(innerScale);
  gltfScene.position.copy(center).multiplyScalar(-innerScale);

  const wrapper = new THREE.Group();
  wrapper.add(gltfScene);
  modelAnimations.set(wrapper, animations);
  wrapper.add(new THREE.AxesHelper(0.7));
  wrapper.userData.normalizedSize = 1;
  wrapper.traverse((part) => {
    if (!part.isMesh) return;
    part.material = new THREE.MeshStandardMaterial({
      color: 0x72f8dd,
      roughness: 0.38,
      metalness: 0.03,
      emissive: 0x16483f,
      emissiveIntensity: 0.45,
      side: THREE.DoubleSide,
      depthTest: false,
      depthWrite: false,
    });
    part.castShadow = true;
    part.receiveShadow = true;
    part.renderOrder = 4;
  });
  return wrapper;
}

new GLTFLoader().load(rightHandModelUrl, ({ scene: gltfScene, animations }) => {
  handModelTemplates.Right = prepareHandTemplate(gltfScene, animations);
});
new GLTFLoader().load(leftHandModelUrl, ({ scene: gltfScene, animations }) => {
  handModelTemplates.Left = prepareHandTemplate(gltfScene, animations);
});

const scene = new THREE.Scene();
const simulationRoot = new THREE.Group();
simulationRoot.visible = true;
scene.add(simulationRoot);
let simulationStarted = true;
let hitTestSource;
let hitTestSourceRequested = false;
let xrSessionMode;
const surfaceMatrix = new THREE.Matrix4();
const reticle = new THREE.Mesh(
  new THREE.RingGeometry(0.12, 0.16, 32).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x72f8dd, transparent: true, opacity: 0.9 }),
);
reticle.matrixAutoUpdate = false;
reticle.visible = false;
scene.add(reticle);

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
let cardboardOrientationEnabled = false;
let deviceOrientation;

function handleDeviceOrientation(event) {
  deviceOrientation = event;
}

async function enableCardboardOrientation() {
  if (cardboardOrientationEnabled) return;
  if (typeof DeviceOrientationEvent === 'undefined') return;
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    const permission = await DeviceOrientationEvent.requestPermission();
    if (permission !== 'granted') return;
  }
  window.addEventListener('deviceorientation', handleDeviceOrientation, true);
  cardboardOrientationEnabled = true;
}

function updateCardboardOrientation() {
  if (!cardboardMode || !deviceOrientation) return;
  const alpha = THREE.MathUtils.degToRad(deviceOrientation.alpha || 0);
  const beta = THREE.MathUtils.degToRad(deviceOrientation.beta || 0);
  const gamma = THREE.MathUtils.degToRad(deviceOrientation.gamma || 0);
  const screenAngle = THREE.MathUtils.degToRad(window.screen.orientation?.angle || 0);
  const euler = new THREE.Euler(beta, alpha, -gamma, 'YXZ');
  const screenQuaternion = new THREE.Quaternion().setFromAxisAngle(
    new THREE.Vector3(0, 0, 1),
    -screenAngle,
  );
  camera.quaternion.setFromEuler(euler);
  camera.quaternion.multiply(new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5)));
  camera.quaternion.multiply(screenQuaternion);
}

function updateCalibrationModel() {
  for (const side of ['Left', 'Right']) {
    const degrees = rotationInputs[side].map((input) => Number(input.value));
    const rotation = degrees.map((value) => THREE.MathUtils.degToRad(value));
    const model = side === 'Left' ? leftCalibrationModel : calibrationModel;
    if (model) model.rotation.set(...rotation);
    rotationValues[side].forEach((output, index) => { output.value = `${degrees[index]}°`; });
    rotationCoordinates[side].textContent = `${side === 'Left' ? 'L' : 'R'} X: ${degrees[0]}°  Y: ${degrees[1]}°  Z: ${degrees[2]}°`;
  }
  if (calibrationModel) calibrationModel.position.x = THREE.MathUtils.clamp(Number(positionInputs.Right.value), 0.1, 2);
  if (leftCalibrationModel) leftCalibrationModel.position.x = THREE.MathUtils.clamp(Number(positionInputs.Left.value), -2, -0.1);
  positionValues.Left.value = positionInputs.Left.value;
  positionValues.Right.value = positionInputs.Right.value;
}

Object.values(rotationInputs).flat().forEach((input) => input.addEventListener('input', updateCalibrationModel));
Object.values(positionInputs).forEach((input) => input.addEventListener('input', updateCalibrationModel));

async function startCamera() {
  if (cameraStarting || (xrSessionActive && xrSessionMode === 'ar')) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    return;
  }
  if (!window.isSecureContext) {
    return;
  }
  cameraStarting = true;
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
    cameraStarting = false;
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
    cameraStarting = false;
    console.error(error);
  }
}

function setCardboardMode(enabled) {
  cardboardMode = enabled;
  cardboardToggle.textContent = enabled ? 'Salir de pantalla dividida' : 'Dividir pantalla';
  document.body.classList.toggle('cardboard-mode', enabled);
  stereoEffect.setSize(sceneHost.clientWidth, sceneHost.clientHeight);
  controls.enabled = !enabled && !xrSessionActive;
  if (enabled) enableCardboardOrientation().catch((error) => console.error('No se pudo activar la orientación:', error));
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

function cameraPoint(landmark, handedness = 'Right') {
  const normalizedX = cameraMirror ? 0.5 - landmark.x : landmark.x - 0.5;
  const point = new THREE.Vector3(normalizedX * 3.1, 2.25 - landmark.y * 2.1, 0.05);
  if (cardboardMode && !xrSessionActive) {
    point.sub(camera.position).applyQuaternion(camera.quaternion).add(camera.position);
  } else if (xrSessionActive && simulationStarted) {
    point.applyMatrix4(simulationRoot.matrix);
  }
  return point;
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
    const palmMaterial = new THREE.MeshStandardMaterial({
      color: 0x72f8dd,
      roughness: 0.35,
      metalness: 0.05,
      depthTest: false,
      depthWrite: false,
    });
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
    group.renderOrder = 10;
    group.visible = false;
  scene.add(group);
  cameraHandVisuals.push(group);
  return group;
}

function applyKalidokitHand(group, worldLandmarks, handedness, points, pinching) {
  if (!group.userData.model || !worldLandmarks?.length) return;
  const solved = Hand.solve(worldLandmarks, handedness);
  if (!solved) return;
  const side = handedness === 'Left' ? 'Left' : 'Right';
  const fingers = {
    thumb: ['Proximal', 'Intermediate', 'Distal'],
    index: ['Proximal', 'Intermediate', 'Distal'],
    midd: ['Proximal', 'Intermediate', 'Distal'],
    ring: ['Proximal', 'Intermediate', 'Distal'],
    pinky: ['Proximal', 'Intermediate', 'Distal'],
  };
  const segments = {
    thumb: ['trapez', 'meta', 'prox'],
    index: ['meta', 'prox', 'midd'],
    midd: ['meta', 'prox', 'midd'],
    ring: ['meta', 'prox', 'midd'],
    pinky: ['meta', 'prox', 'midd'],
  };
  for (const [finger, kalidoSegments] of Object.entries(fingers)) {
    kalidoSegments.forEach((segment, index) => {
      const rotation = solved[`${side}${finger === 'midd' ? 'Middle' : finger[0].toUpperCase() + finger.slice(1)}${segment}`];
      const bone = group.userData.model.getObjectByName(`${finger}_${segments[finger][index]}`);
      if (rotation && bone) {
        const rest = group.userData.rigBones.find((entry) => entry.bone === bone)?.restLocalQuaternion;
        const pose = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(-rotation.x, rotation.y, handedness === 'Right' ? -rotation.z : rotation.z),
        );
        bone.quaternion.copy(rest || new THREE.Quaternion()).multiply(pose);
      }
    });
  }
  if (pinching) {
    const curl = handedness === 'Right' ? 0.9 : -0.9;
    const curlQuaternion = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, curl));
    for (const entry of group.userData.rigBones || []) {
      if (!entry.restLocalQuaternion) continue;
      const target = entry.restLocalQuaternion.clone().multiply(curlQuaternion);
      entry.bone.quaternion.slerp(target, 0.35);
    }
  }
}

function updatePoseAnimation(group, pinching) {
  const action = group.userData.poseAction;
  if (!action) return;
  if (pinching && !group.userData.poseActive) {
    action.reset().play();
    group.userData.poseActive = true;
  } else if (!pinching && group.userData.poseActive) {
    action.stop().reset();
    group.userData.poseActive = false;
  }
}

function updateCameraHandVisual(group, landmarks, worldLandmarks, handedness, pinching) {
   const points = landmarks.map((landmark) => cameraPoint(landmark, handedness));
   const palm = new THREE.Vector3()
     .add(points[0]).add(points[5]).add(points[9]).add(points[13]).add(points[17])
     .multiplyScalar(0.2);
    if (group.userData.modelSide !== handedness && group.userData.model) {
      group.remove(group.userData.model);
      group.userData.model = null;
    }
    if (!group.userData.model && handModelTemplates[handedness]) {
      attachHandModel(group, handModelTemplates[handedness]);
      group.userData.modelSide = handedness;
    }
   if (group.userData.model) {
     const palm = new THREE.Vector3()
       .add(points[0]).add(points[5]).add(points[9]).add(points[13]).add(points[17])
       .multiplyScalar(0.2);
     const handLength = points[0].distanceTo(points[12]);
     group.userData.model.scale.setScalar(handLength * HAND_MODEL_SCALE);
       const baseRotation = HAND_MODEL_ROTATIONS[handedness].map((value) => THREE.MathUtils.degToRad(value));
       const palmDirection = worldLandmarks?.[17] && worldLandmarks?.[5]
         ? {
             x: worldLandmarks[17].x - worldLandmarks[5].x,
             z: worldLandmarks[17].z - worldLandmarks[5].z,
           }
         : null;
       const landmarkYaw = palmDirection
         ? THREE.MathUtils.clamp(Math.atan2(palmDirection.z, -palmDirection.x), -LANDMARK_Y_ROTATION_LIMIT, LANDMARK_Y_ROTATION_LIMIT)
         : 0;
       const rotation = [baseRotation[0], baseRotation[1] + landmarkYaw, baseRotation[2]];
      group.userData.model.rotation.set(...rotation);
     group.userData.model.position.copy(palm);
     group.userData.model.position.x += HAND_MODEL_X_OFFSET;
     group.userData.model.position.y += HAND_MODEL_Y_OFFSET;
     group.userData.model.position.z += HAND_MODEL_Z_OFFSET;
   }
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
   try {
     updatePoseAnimation(group, pinching);
   } catch (error) {
     console.error('No se pudo animar el modelo de mano:', error);
   }
}

function updateInteractiveHand(points) {
  if (!interactiveHand || !points.length) return;
  const palm = new THREE.Vector3()
    .add(points[0])
    .add(points[5])
    .add(points[9])
    .add(points[13])
    .add(points[17])
    .multiplyScalar(0.2);
  const touchesModel = points.some((point) => point.distanceTo(interactiveHand.position) < 0.55);
  if (touchesModel && !handGrabbed) {
    handGrabbed = true;
    handGrabOffset.copy(interactiveHand.position).sub(palm);
  } else if (!touchesModel && handGrabbed) {
    handGrabbed = false;
  }
  if (handGrabbed) interactiveHand.position.copy(palm).add(handGrabOffset);
  if (!interactiveHand.userData.model) return;
  interactiveHand.userData.model.updateMatrixWorld(true);
  for (const entry of interactiveHand.userData.rigBones || []) {
    const targetDirection = points[entry.end].clone().sub(points[entry.start]);
    if (targetDirection.lengthSq() < 0.000001 || entry.restDirection.lengthSq() < 0.000001) continue;
    const delta = new THREE.Quaternion().setFromUnitVectors(
      entry.restDirection.clone().normalize(),
      targetDirection.normalize(),
    );
    const targetWorldQuaternion = delta.multiply(entry.restQuaternion);
    if (entry.bone.parent) {
      const parentWorldQuaternion = entry.bone.parent.getWorldQuaternion(new THREE.Quaternion());
      entry.bone.quaternion.copy(parentWorldQuaternion.invert().multiply(targetWorldQuaternion));
    } else {
      entry.bone.quaternion.copy(targetWorldQuaternion);
    }
    entry.bone.updateMatrixWorld(true);
  }
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
   if (!(result.landmarks || []).length) handGrabbed = false;
   (result.landmarks || []).forEach((landmarks, index) => {
      const source = `camera-${index}`;
      seen.add(source);
      const handedness = result.handednesses?.[index]?.[0]?.categoryName || 'Right';
      const pinching = Math.hypot(landmarks[4].x - landmarks[8].x, landmarks[4].y - landmarks[8].y) < 0.075;
     const handVisual = cameraHandVisuals[index] || createCameraHandVisual();
     updateCameraHandVisual(
       handVisual,
        landmarks,
        result.worldLandmarks?.[index],
         handedness,
        pinching,
      );
      updateInteractiveHand(landmarks.map((landmark) => cameraPoint(landmark, handedness)));
     const point = cameraPoint({
       x: (landmarks[4].x + landmarks[8].x) / 2,
       y: (landmarks[4].y + landmarks[8].y) / 2,
     }, handedness);
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
simulationRoot.add(floor);

const grid = new THREE.GridHelper(6, 24, 0x384065, 0x1c2540);
grid.position.y = 0.006;
grid.material.transparent = true;
grid.material.opacity = 0.42;
simulationRoot.add(grid);

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
  simulationRoot.add(mesh);
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
  hand.traverse((part) => {
    if (!part.isMesh && !part.isSkinnedMesh) return;
    part.renderOrder = 10;
    const materials = Array.isArray(part.material) ? part.material : [part.material];
    materials.forEach((material) => {
      if (!material) return;
      material.depthTest = false;
      material.depthWrite = false;
      material.needsUpdate = true;
    });
  });
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
  if (!simulationStarted) return;
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

async function requestXRHitTestSource() {
  const session = renderer.xr.getSession();
  if (!session || hitTestSourceRequested) return;
  hitTestSourceRequested = true;
  try {
    const viewerSpace = await session.requestReferenceSpace('viewer');
    hitTestSource = await session.requestHitTestSource({ space: viewerSpace });
  } catch (error) {
    hitTestSourceRequested = false;
    console.error('No se pudo iniciar la detección de superficies:', error);
  }
}

function updateXRSurface() {
  if (xrSessionMode !== 'ar' || !xrSessionActive || !hitTestSource || simulationStarted) return;
  const frame = renderer.xr.getFrame();
  const referenceSpace = renderer.xr.getReferenceSpace();
  if (!frame || !referenceSpace) return;

  const hit = frame.getHitTestResults(hitTestSource).map((result) => {
    const pose = result.getPose(referenceSpace);
    if (!pose) return null;
    const matrix = pose.transform.matrix;
    const normalY = Math.abs(matrix[5]);
    if (normalY > 0.85) return { result, pose };
    return null;
  }).filter(Boolean)[0];
  if (!hit) return;

  const { pose } = hit;
  surfaceMatrix.fromArray(pose.transform.matrix);
  reticle.matrix.copy(surfaceMatrix);
  reticle.visible = true;

  simulationRoot.matrix.copy(surfaceMatrix);
  simulationRoot.matrix.decompose(simulationRoot.position, simulationRoot.quaternion, simulationRoot.scale);
  simulationRoot.visible = true;
  simulationStarted = true;
}

renderer.xr.addEventListener('sessionstart', () => {
  xrSessionActive = true;
  if (xrSessionMode === 'ar') {
    simulationStarted = false;
    simulationRoot.visible = false;
    reticle.visible = false;
    hitTestSource = undefined;
    hitTestSourceRequested = false;
    requestXRHitTestSource();
  } else {
    simulationStarted = true;
    simulationRoot.visible = true;
  }
  controls.enabled = false;
  cameraPinches.clear();
  grabbedBy.clear();
});
renderer.xr.addEventListener('sessionend', () => {
  xrSessionActive = false;
  hitTestSource?.cancel();
  hitTestSource = undefined;
  hitTestSourceRequested = false;
  reticle.visible = false;
  simulationRoot.visible = true;
  simulationStarted = true;
  controls.enabled = !cardboardMode;
  grabbedBy.clear();
  xrSessionMode = undefined;
});

setupInput(0);
setupInput(1);
const vrButton = VRButton.createButton(renderer, { optionalFeatures: ['hand-tracking'] });
vrButton.classList.add('VRButton');
vrButton.textContent = 'Entrar en VR';
vrButton.addEventListener('click', () => { xrSessionMode = 'vr'; });
controlBar.appendChild(vrButton);

const arButton = ARButton.createButton(renderer, {
  requiredFeatures: ['hit-test', 'local-floor'],
  optionalFeatures: ['plane-detection', 'hand-tracking'],
});
arButton.classList.add('ARButton');
arButton.textContent = 'Entrar en AR';
arButton.addEventListener('click', () => {
  xrSessionMode = 'ar';
});
controlBar.appendChild(arButton);
cameraToggle.addEventListener('click', startCamera);
cameraSwitch.addEventListener('click', switchCamera);
cardboardToggle.addEventListener('click', () => {
  if (!cardboardMode && !cameraStream) startCamera();
  setCardboardMode(!cardboardMode);
});

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
  if (!xrSessionActive && !cardboardMode) controls.update();
  updateCardboardOrientation();
  updateCameraHands();
  updateXRSurface();
   const delta = handClock.getDelta();
   handMixers.forEach((mixer) => mixer.update(delta));
   updatePhysics(delta);
  updateHands();
   for (const object of objects) {
     if (simulationStarted && !object.userData.grabbed) object.rotation.y += 0.0025;
  }
    if (cardboardMode && !xrSessionActive) stereoEffect.render(scene, camera);
   else renderer.render(scene, camera);
});
