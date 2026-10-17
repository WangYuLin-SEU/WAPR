// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

/* LM-O frame: detector bboxes, projected pose, and the same 3D scene as the dataset viewer.
   LM-O 的一帧：检测包围盒、投影后的位姿，以及与数据集查看器相同的 3D 场景。
   Point positions in cloud.bin are OpenCV camera meters. The view uses y-up: (x, -y, -z).
   cloud.bin 里的点是 OpenCV 相机系、米。视图用 y 朝上：(x, -y, -z)。 */
(function () {
  var BOX = [];
  for (var corner = 0; corner < 8; corner++) {
    for (var bit = 0; bit < 3; bit++) {
      var other = corner ^ (1 << bit);
      if (other > corner) BOX.push([corner, other]);
    }
  }

  var scene = null;
  var rgbImage = null;
  var contourImage = null;
  var view = null;
  var selected = null;

  function $(id) { return document.getElementById(id); }

  function flatCorners(corners) {
    if (!corners || !corners.length || typeof corners[0] === "number") return corners;
    var flat = [];
    for (var i = 0; i < corners.length; i++) {
      flat.push(corners[i][0], corners[i][1], corners[i][2]);
    }
    return flat;
  }

  function project(pose, corners, index, K) {
    var x = corners[index * 3];
    var y = corners[index * 3 + 1];
    var z = corners[index * 3 + 2];
    var X = pose[0] * x + pose[1] * y + pose[2] * z + pose[3];
    var Y = pose[4] * x + pose[5] * y + pose[6] * z + pose[7];
    var Z = pose[8] * x + pose[9] * y + pose[10] * z + pose[11];
    if (Z <= 1.0e-4) return null;
    return [
      (K[0] * X + K[1] * Y + K[2] * Z) / Z,
      (K[3] * X + K[4] * Y + K[5] * Z) / Z
    ];
  }

  function ink(color) {
    var luma = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2];
    return luma > 170 ? "#1c1c1c" : "#ffffff";
  }

  function tag(ctx, text, x, y, color, taken) {
    ctx.font = "600 14px sans-serif";
    var padX = 4;
    var width = Math.ceil(ctx.measureText(text).width) + padX * 2;
    var height = 18;
    var left = Math.round(Math.max(0, Math.min(x, scene.width - width)));
    var top = Math.round(y);
    if (top < 0) top = 0;
    for (var step = 0; step < 16; step++) {
      var hit = false;
      for (var i = 0; i < taken.length; i++) {
        var box = taken[i];
        if (left < box.x + box.w && left + width > box.x && top < box.y + box.h && top + height > box.y) {
          hit = true;
        }
      }
      if (!hit) break;
      top += height + 1;
      if (top + height > scene.height) top = 0;
    }
    taken.push({ x: left, y: top, w: width, h: height });
    ctx.fillStyle = "rgb(" + color[0] + "," + color[1] + "," + color[2] + ")";
    ctx.fillRect(left, top, width, height);
    ctx.fillStyle = ink(color);
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + padX, top + height / 2);
  }

  function secondsText(seconds) {
    return seconds.toFixed(seconds < 1 ? 3 : 2) + " s";
  }

  function pageLang() {
    return document.documentElement.getAttribute("data-lang") === "zh" ? "zh" : "en";
  }

  function stampBox(ctx, text) {
    ctx.font = "600 15px sans-serif";
    var padX = 8;
    var width = Math.ceil(ctx.measureText(text).width) + padX * 2;
    return { text: text, x: 8, y: 8, w: width, h: 26, padX: padX };
  }

  function stamp(ctx, box) {
    ctx.font = "600 15px sans-serif";
    ctx.fillStyle = "rgba(20, 20, 20, 0.82)";
    ctx.fillRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle = "#ffffff";
    ctx.textBaseline = "middle";
    ctx.fillText(box.text, box.x + box.padX, box.y + box.h / 2);
  }

  function drawBoxes(canvas, rows, timeText) {
    if (!canvas || !scene || !rgbImage || !rgbImage.complete) return;
    canvas.width = scene.width;
    canvas.height = scene.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(rgbImage, 0, 0);
    var taken = [];
    if (timeText) {
      var timeBox = stampBox(ctx, timeText);
      taken.push(timeBox);
    }
    var objects = rows.slice().sort(function (a, b) { return a.score_2d - b.score_2d; });
    for (var i = 0; i < objects.length; i++) {
      var obj = objects[i];
      var on = obj.id && obj.id === selected;
      var box = obj.bbox_xywh;
      ctx.lineWidth = on ? Math.max(3, scene.width / 180) : Math.max(1.5, scene.width / 360);
      ctx.strokeStyle = "rgb(" + obj.color[0] + "," + obj.color[1] + "," + obj.color[2] + ")";
      ctx.strokeRect(box[0], box[1], box[2], box[3]);
      tag(ctx, obj.name + " 2D " + obj.score_2d.toFixed(2), box[0], box[1] - 18, obj.color, taken);
    }
    if (timeText) stamp(ctx, taken[0]);
  }

  function drawAll() {
    drawBoxes($("lmo-all-canvas"), scene.detections || [], "wapr.det2d " + secondsText(scene.det_s) + " · " + scene.gpu);
  }

  function drawFiltered() {
    drawBoxes($("lmo-det-canvas"), scene.objects, "");
  }

  function drawPoses() {
    var canvas = $("lmo-pose-canvas");
    if (!canvas || !scene || !rgbImage || !rgbImage.complete) return;
    canvas.width = scene.width;
    canvas.height = scene.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(rgbImage, 0, 0);
    if (contourImage && contourImage.complete && contourImage.naturalWidth) {
      ctx.drawImage(contourImage, 0, 0);
    }
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.lineWidth = Math.max(1.5, scene.width / 360);
    var objects = scene.objects.slice().sort(function (a, b) {
      if ((a.id === selected) !== (b.id === selected)) return a.id === selected ? 1 : -1;
      return a.score_6d - b.score_6d;
    });
    var poseLabel = pageLang() === "zh" ? "6D 位姿 " : "6D pose ";
    var timeBox = stampBox(ctx, poseLabel + secondsText(scene.pose_s) + " · " + scene.gpu);
    var taken = [timeBox];
    for (var i = 0; i < objects.length; i++) {
      var obj = objects[i];
      var on = obj.id === selected;
      ctx.lineWidth = on ? Math.max(3, scene.width / 160) : Math.max(1.5, scene.width / 360);
      ctx.globalAlpha = on || selected === null ? 1 : 0.28;
      ctx.strokeStyle = "rgb(" + obj.color[0] + "," + obj.color[1] + "," + obj.color[2] + ")";
      ctx.beginPath();
      var top = null;
      for (var e = 0; e < BOX.length; e++) {
        var a = project(obj.box_pose, obj.corners, BOX[e][0], scene.K);
        var b = project(obj.box_pose, obj.corners, BOX[e][1], scene.K);
        if (!a || !b) continue;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        if (!top || a[1] < top[1]) top = a;
        if (!top || b[1] < top[1]) top = b;
      }
      ctx.stroke();
      if (!top) continue;
      tag(
        ctx,
        obj.name + " 6D " + obj.score_6d.toFixed(3),
        top[0],
        top[1] - 18,
        obj.color,
        taken
      );
    }
    ctx.globalAlpha = 1;
    stamp(ctx, timeBox);
  }

  function poseMatrix(pose) {
    var m = new THREE.Matrix4();
    m.set(
      pose[0], pose[1], pose[2], pose[3],
      -pose[4], -pose[5], -pose[6], -pose[7],
      -pose[8], -pose[9], -pose[10], -pose[11],
      0, 0, 0, 1
    );
    return m;
  }

  function srgbByteToLinear(byte) {
    var c = byte / 255;
    if (c <= 0.04045) return c / 12.92;
    return Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function meshKey(objId) {
    var id = String(objId);
    while (id.length < 6) id = "0" + id;
    return "lmo/obj_" + id;
  }

  function decodeMesh(meta, buffer, key) {
    var n = meta.vertices;
    var f = meta.faces;
    var positions = new Float32Array(buffer, 0, n * 3);
    var indices = new Uint32Array(buffer, n * 12, f * 3);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    var colorByte = n * 12 + f * 12;
    if (meta.mode === "uv") {
      var uv = new Float32Array(buffer, colorByte, n * 2);
      geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
      return new Promise(function (resolve, reject) {
        var loader = new THREE.TextureLoader();
        loader.load("demo/meshes/" + key + ".jpg", function (tex) {
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.flipY = true;
          tex.wrapS = THREE.ClampToEdgeWrapping;
          tex.wrapT = THREE.ClampToEdgeWrapping;
          tex.needsUpdate = true;
          resolve({ geometry: geo, texture: tex, mode: "uv" });
        }, undefined, reject);
      });
    }
    var rgb = new Uint8Array(buffer, colorByte, n * 3);
    var colors = new Float32Array(n * 3);
    for (var i = 0; i < n * 3; i++) colors[i] = srgbByteToLinear(rgb[i]);
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return { geometry: geo, texture: null, mode: "vertex" };
  }

  var meshCache = {};

  function loadMesh(objId) {
    var key = meshKey(objId);
    if (meshCache[key]) return meshCache[key];
    meshCache[key] = WAPRAssets.fetch("demo/meshes/" + key + ".json").then(function (res) {
      if (!res.ok) throw new Error(key);
      return res.json();
    }).then(function (meta) {
      return WAPRAssets.fetch("demo/meshes/" + key + ".bin").then(function (res) {
        if (!res.ok) throw new Error(key);
        return res.arrayBuffer();
      }).then(function (buffer) {
        return decodeMesh(meta, buffer, key);
      });
    });
    return meshCache[key];
  }

  function placeCamera() {
    var offset = new THREE.Vector3().setFromSpherical(view.spherical);
    view.camera.position.copy(view.target).add(offset);
    view.camera.lookAt(view.target);
  }

  /* After a click, orbit this instance. Copies that share one model stay
     separate: the pivot is this mesh, and the camera keeps its distance.
     点中之后绕这个实例转。同一个模型的多份是分开的：转轴是这一份网格，距离保持不变。 */
  function focusInstance(id) {
    if (!view) return;
    view.orbitId = id;
    if (!view.records) return;
    var mesh = null;
    for (var i = 0; i < view.records.length; i++) {
      if (view.records[i].obj.id === id) {
        mesh = view.records[i].mesh;
        break;
      }
    }
    if (!mesh || !mesh.geometry) return;
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    mesh.updateWorldMatrix(true, false);
    var center = mesh.geometry.boundingBox.clone().applyMatrix4(mesh.matrixWorld).getCenter(new THREE.Vector3());
    var offset = view.camera.position.clone().sub(center);
    if (offset.lengthSq() < 1.0e-8) offset.set(0, 0, 0.35);
    view.target.copy(center);
    view.spherical.setFromVector3(offset);
    placeCamera();
  }

  function meshAt(event) {
    if (!view || !view.raycaster || !view.lines) return 0;
    var canvas = view.renderer.domElement;
    var rect = canvas.getBoundingClientRect();
    var x = event.clientX - rect.left;
    var y = event.clientY - rect.top;
    var cssH = view.cssH || rect.height;
    var left = view.vpX || 0;
    var top = view.vpH ? cssH - view.vpY - view.vpH : 0;
    var vw = view.vpW || rect.width;
    var vh = view.vpH || rect.height;
    if (x < left || y < top || x > left + vw || y > top + vh) return 0;
    var ndc = new THREE.Vector2(
      ((x - left) / vw) * 2 - 1,
      -((y - top) / vh) * 2 + 1
    );
    view.lines.updateMatrixWorld(true);
    view.raycaster.setFromCamera(ndc, view.camera);
    var hits = view.raycaster.intersectObjects(view.lines.children, false);
    for (var i = 0; i < hits.length; i++) {
      if (hits[i].object.userData.id) return hits[i].object.userData.id;
    }
    return 0;
  }

  function frameView() {
    if (!view || !scene || !scene.K || !view.points) return;
    view.points.geometry.computeBoundingBox();
    var focus = view.points.geometry.boundingBox;
    if (!focus || focus.isEmpty()) return;
    var center = focus.getCenter(new THREE.Vector3());
    var radius = Math.max(focus.getSize(new THREE.Vector3()).length(), 0.05);
    // Depth along the optical axis. The y-up cloud stores OpenCV Z as -z.
    // 光轴上的深度。y 朝上的点云里，OpenCV 的 Z 存在 -z。
    var depth = Math.max(-center.z, radius * 0.25, 0.05);
    var K = scene.K;
    view.photo = {
      fx: K[0],
      fy: K[4],
      cx: K[2],
      cy: K[5],
      width: scene.width,
      height: scene.height
    };
    view.points.material.size = Math.max(depth * 4.8 / K[4], 0.0012);
    // Same sensor picture as the pose image. A floor grid would sit in front of it.
    // 和位姿图同一个传感器画面。地面格子会挡在这个画面前面。
    if (view.grid) {
      view.scene.remove(view.grid);
      if (window.WAPRStudio) WAPRStudio.dispose(view.grid);
      view.grid = null;
    }
    view.orbitId = 0;
    view.target.set(0, 0, -depth);
    view.spherical.setFromVector3(new THREE.Vector3(0, 0, depth));
    placeCamera();
    var farZ = Math.max(-focus.min.z, depth);
    view.camera.near = 0.01;
    view.camera.far = Math.max(farZ * 2.5, depth * 12, 8);
    resizeView();
  }

  function applyPhotoProjection() {
    var photo = view.photo;
    if (!photo) return;
    var near = view.camera.near;
    var far = view.camera.far;
    var span = far - near;
    var c = -(far + near) / span;
    var d = (-2 * far * near) / span;
    // Same K as the pose image, after the (x, -y, -z) flip. NDC y is up.
    // 和位姿图同一套 K，已经做了 (x, -y, -z)。NDC 的 y 朝上。
    view.camera.projectionMatrix.set(
      2 * photo.fx / photo.width, 0, 1 - 2 * photo.cx / photo.width, 0,
      0, 2 * photo.fy / photo.height, 2 * photo.cy / photo.height - 1, 0,
      0, 0, c, d,
      0, 0, -1, 0
    );
    view.camera.projectionMatrixInverse.copy(view.camera.projectionMatrix).invert();
  }

  function showCloud(buffer) {
    var n = scene.cloud_count;
    var xyz = new Float32Array(buffer, 0, n * 3);
    var rgb = new Uint8Array(buffer, n * 12, n * 3);
    var pos = new Float32Array(n * 3);
    var col = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {
      pos[i * 3] = xyz[i * 3];
      pos[i * 3 + 1] = -xyz[i * 3 + 1];
      pos[i * 3 + 2] = -xyz[i * 3 + 2];
      col[i * 3] = rgb[i * 3] / 255;
      col[i * 3 + 1] = rgb[i * 3 + 1] / 255;
      col[i * 3 + 2] = rgb[i * 3 + 2] / 255;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    view.points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.004,
      vertexColors: true,
      sizeAttenuation: true
    }));
    view.scene.add(view.points);
    // Frame from the cloud now. Meshes arrive later and must not move the camera.
    // 现在就按点云对好相机。网格稍后到来，不能再把相机拽走。
    frameView();
    var jobs = [];
    for (var k = 0; k < scene.objects.length; k++) {
      jobs.push((function (obj) {
        return loadMesh(obj.obj_id).then(function (asset) {
          return { obj: obj, asset: asset };
        });
      })(scene.objects[k]));
    }
    Promise.all(jobs).then(function (loaded) {
      for (var i = 0; i < loaded.length; i++) {
        var item = loaded[i];
        var mesh = new THREE.Mesh(item.asset.geometry, new THREE.MeshBasicMaterial({
          map: item.asset.texture || null,
          vertexColors: item.asset.mode === "vertex",
          color: 0xffffff,
          side: THREE.DoubleSide,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2
        }));
        mesh.matrixAutoUpdate = false;
        mesh.matrix.copy(poseMatrix(item.obj.pred_pose));
        mesh.userData.id = item.obj.id;
        mesh.renderOrder = 1;
        view.lines.add(mesh);
        var shell = new THREE.Mesh(item.asset.geometry, new THREE.MeshBasicMaterial({
          color: new THREE.Color("rgb(" + item.obj.color[0] + "," + item.obj.color[1] + "," + item.obj.color[2] + ")"),
          side: THREE.BackSide,
          toneMapped: false,
          polygonOffset: true,
          polygonOffsetFactor: -4,
          polygonOffsetUnits: -4
        }));
        shell.matrixAutoUpdate = false;
        shell.userData.id = item.obj.id;
        shell.visible = false;
        shell.renderOrder = 2;
        view.lines.add(shell);
        view.records.push({ obj: item.obj, mesh: mesh, shell: shell });
      }
      view.lines.updateMatrixWorld(true);
      syncMeshes();
      if (view.orbitId) focusInstance(view.orbitId);
    });
  }

  function syncMeshes() {
    if (!view || !view.records) return;
    var scale = new THREE.Matrix4().makeScale(1.045, 1.045, 1.045);
    for (var i = 0; i < view.records.length; i++) {
      var record = view.records[i];
      var pose = poseMatrix(record.obj.pred_pose);
      var on = record.obj.id === selected;
      record.mesh.matrix.copy(pose);
      record.mesh.material.color.setHex(on || selected === null ? 0xffffff : 0x9a9a9a);
      record.shell.matrix.multiplyMatrices(pose, scale);
      record.shell.visible = on;
    }
    view.lines.updateMatrixWorld(true);
  }

  function fillChips() {
    var host = $("lmo-chips");
    if (!host || !scene) return;
    host.innerHTML = "";
    var objects = scene.objects.slice().sort(function (a, b) {
      return b.score_6d - a.score_6d;
    });
    for (var i = 0; i < objects.length; i++) {
      var obj = objects[i];
      var button = document.createElement("button");
      button.type = "button";
      var highScore = obj.score_6d >= scene.score_6d_high_min;
      var scoreText = obj.score_6d.toFixed(3);
      var dot = document.createElement("span");
      dot.className = highScore ? "lmo-chip-dot lmo-chip-high" : "lmo-chip-dot lmo-chip-low";
      dot.setAttribute("aria-hidden", "true");
      var label = document.createElement("span");
      label.textContent = obj.name;
      var score = document.createElement("span");
      score.className = "lmo-chip-score";
      score.textContent = "6D " + scoreText;
      button.setAttribute("aria-pressed", obj.id === selected ? "true" : "false");
      button.setAttribute("aria-label", obj.name + " 6D " + scoreText);
      button.appendChild(dot);
      button.appendChild(label);
      button.appendChild(score);
      button.addEventListener("click", (function (id) {
        return function () { selectObject(id); };
      })(obj.id));
      host.appendChild(button);
    }
  }

  function selectObject(id) {
    selected = id;
    drawFiltered();
    drawPoses();
    fillChips();
    syncMeshes();
    focusInstance(id);
  }

  function redraw() {
    drawAll();
    drawFiltered();
    drawPoses();
  }

  function resizeView() {
    if (!view) return;
    var canvas = view.renderer.domElement;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    if (w < 2 || h < 2) return;
    if (view.cssW !== w || view.cssH !== h) {
      view.cssW = w;
      view.cssH = h;
      view.renderer.setSize(w, h, false);
    }
    if (!view.photo) {
      view.camera.aspect = w / h;
      view.camera.updateProjectionMatrix();
      view.vpX = 0;
      view.vpY = 0;
      view.vpW = w;
      view.vpH = h;
      return;
    }
    // Fit the sensor frame in the panel. Viewport numbers are CSS pixels.
    // three.js applies the pixel ratio. Buffer pixels crop the upper right.
    // 把传感器画面放进面板。视口用 CSS 像素。three.js 自己会乘像素比。
    // 传缓冲区像素时，放大的屏幕会把右上角裁掉。
    var imgAspect = view.photo.width / view.photo.height;
    var canvasAspect = w / h;
    var vw;
    var vh;
    var vx;
    if (canvasAspect > imgAspect) {
      vh = h;
      vw = h * imgAspect;
      vx = (w - vw) / 2;
    } else {
      vw = w;
      vh = w / imgAspect;
      vx = 0;
    }
    view.vpX = vx;
    view.vpY = h - vh;
    view.vpW = Math.max(1, vw);
    view.vpH = Math.max(1, vh);
    applyPhotoProjection();
  }

  function ensureView() {
    if (view || typeof THREE === "undefined") return;
    var canvas = $("lmo-cloud");
    if (!canvas) return;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    var scene3 = new THREE.Scene();
    if (window.WAPRStudio) WAPRStudio.applyBackdrop(scene3);
    var camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
    camera.up.set(0, 1, 0);
    var lines = new THREE.Group();
    scene3.add(lines);
    view = {
      renderer: renderer,
      scene: scene3,
      camera: camera,
      lines: lines,
      records: [],
      points: null,
      grid: null,
      target: new THREE.Vector3(),
      spherical: new THREE.Spherical(1, 1.1, 0.4),
      raycaster: new THREE.Raycaster(),
      drag: null
    };
    canvas.addEventListener("pointerdown", function (event) {
      view.drag = {
        x: event.clientX,
        y: event.clientY,
        moved: false,
        pan: event.button === 2 || event.shiftKey
      };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!view.drag) return;
      var dx = event.clientX - view.drag.x;
      var dy = event.clientY - view.drag.y;
      if (dx * dx + dy * dy > 9) view.drag.moved = true;
      view.drag.x = event.clientX;
      view.drag.y = event.clientY;
      if (!view.drag.moved) return;
      if (view.drag.pan) {
        var offset = new THREE.Vector3().setFromSpherical(view.spherical);
        var forward = offset.clone().normalize();
        var right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
        var up = new THREE.Vector3().crossVectors(right, forward).normalize();
        var scale = view.spherical.radius * 0.0016;
        view.target.addScaledVector(right, -dx * scale);
        view.target.addScaledVector(up, dy * scale);
      } else {
        view.spherical.theta -= dx * 0.005;
        view.spherical.phi = Math.max(0.08, Math.min(Math.PI - 0.08, view.spherical.phi - dy * 0.005));
      }
      placeCamera();
    });
    function endDrag(event) {
      var drag = view.drag;
      view.drag = null;
      if (!drag || drag.moved || drag.pan || !event) return;
      var id = meshAt(event);
      if (id) selectObject(id);
    }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", function () { view.drag = null; });
    canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      var factor = event.deltaY > 0 ? 1.08 : 0.92;
      view.spherical.radius = Math.max(0.05, Math.min(12, view.spherical.radius * factor));
      placeCamera();
    }, { passive: false });
    canvas.addEventListener("contextmenu", function (event) { event.preventDefault(); });
    window.addEventListener("resize", resizeView);
    function frame() {
      resizeView();
      var cssW = view.cssW || canvas.clientWidth;
      var cssH = view.cssH || canvas.clientHeight;
      if (view.photo && view.vpW) {
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, cssW, cssH);
        renderer.clear();
        renderer.setScissorTest(true);
        renderer.setViewport(view.vpX, view.vpY, view.vpW, view.vpH);
        renderer.setScissor(view.vpX, view.vpY, view.vpW, view.vpH);
      } else {
        renderer.setScissorTest(false);
        renderer.setViewport(0, 0, cssW, cssH);
      }
      renderer.render(scene3, camera);
    }
    WAPRRender.register(canvas, frame);
  }

  function boot() {
    var canvas = $("lmo-det-canvas");
    if (!canvas) return;
    WAPRAssets.fetch("demo/lmo_det/scene.json?v=17").then(function (res) {
      if (!res.ok) throw new Error("scene");
      return res.json();
    }).then(function (data) {
      scene = data;
      var best = null;
      for (var i = 0; i < scene.objects.length; i++) {
        scene.objects[i].corners = flatCorners(scene.objects[i].corners);
        if (!best || scene.objects[i].score_6d > best.score_6d) {
          best = scene.objects[i];
        }
      }
      selected = best ? best.id : null;
      fillChips();
      rgbImage = new Image();
      rgbImage.onload = function () { redraw(); };
      rgbImage.src = scene.rgb;
      contourImage = new Image();
      contourImage.onload = function () { drawPoses(); };
      contourImage.src = "demo/lmo_det/contour.png?v=13";
      document.addEventListener("wapr-lang-change", function () { redraw(); });
      WAPRAssets.fetch(scene.cloud).then(function (res) {
        if (!res.ok) throw new Error("cloud");
        return res.arrayBuffer();
      }).then(function (buffer) {
        ensureView();
        showCloud(buffer);
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
