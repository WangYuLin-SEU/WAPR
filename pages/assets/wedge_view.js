// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

/* White wedges, HCCEPose frame 000003. Display WAPR poses only.
   白色三角块，HCCEPose 的 000003 帧。只展示 WAPR 位姿。
   cloud.bin is OpenCV camera meters. The 3D view is y-up: (x, -y, -z).
   cloud.bin 是 OpenCV 相机系、米。三维视图 y 朝上：(x, -y, -z)。 */
(function () {
  var host = document.getElementById("wedge");
  if (!host) return;

  var BOX = [];
  for (var corner = 0; corner < 8; corner++) {
    for (var bit = 0; bit < 3; bit++) {
      var other = corner ^ (1 << bit);
      if (other > corner) BOX.push([corner, other]);
    }
  }
  // Axis-aligned box of models/wedge.ply after millimeters become meters.
  // models/wedge.ply 换成米之后的轴对齐盒子。
  var CORNERS = [
    [-0.05, -0.03, -0.03], [0.05, -0.03, -0.03],
    [-0.05, 0.03, -0.03], [0.05, 0.03, -0.03],
    [-0.05, -0.03, 0.03], [0.05, -0.03, 0.03],
    [-0.05, 0.03, 0.03], [0.05, 0.03, 0.03]
  ];

  var scene = null;
  var rgb = null;
  var view = null;
  var methodIndex = 0;
  var selected = null;
  var root = host.getAttribute("data-root") || "wedge/";
  var HIGH = "#245db3";
  var LOW = "#66717c";
  var UNSCORED = "#5f6b76";

  function $(id) { return document.getElementById(id); }

  function method() {
    return scene.methods[methodIndex];
  }

  function hypotheses() {
    return method().hypotheses;
  }

  function hue(index) {
    var predictionColors = ["#c62828", "#245db3", "#a349a4", "#b35c14", "#6d49ad", "#a83355"];
    return predictionColors[index % predictionColors.length];
  }

  function chipColor(item) {
    if (method().score_kind !== "score_6d") return UNSCORED;
    return item.score_6d >= 0.5 ? HIGH : LOW;
  }

  function chipText(item) {
    if (method().score_kind === "score_6d") return "wedge 6D " + item.score_6d.toFixed(3);
    if (method().score_kind === "similarity") return "wedge " + Math.round(item.score);
    return "wedge " + Math.round(item.score);
  }

  function rankScore(item) {
    return method().score_kind === "score_6d" ? item.score_6d : item.score;
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

  function project(pose, x, y, z, K) {
    var X = pose[0] * x + pose[1] * y + pose[2] * z + pose[3];
    var Y = pose[4] * x + pose[5] * y + pose[6] * z + pose[7];
    var Z = pose[8] * x + pose[9] * y + pose[10] * z + pose[11];
    if (Z <= 1.0e-4) return null;
    return [
      (K[0] * X + K[1] * Y + K[2] * Z) / Z,
      (K[3] * X + K[4] * Y + K[5] * Z) / Z
    ];
  }

  function boxTop(pose, K) {
    var top = null;
    for (var i = 0; i < CORNERS.length; i++) {
      var point = project(pose, CORNERS[i][0], CORNERS[i][1], CORNERS[i][2], K);
      if (!point) continue;
      if (!top || point[1] < top[1]) top = point;
    }
    return top;
  }

  function strokeRing(ctx, ring, color, width, alpha) {
    if (!ring || ring.length < 2) return;
    ctx.beginPath();
    for (var i = 0; i < ring.length; i++) {
      if (i === 0) ctx.moveTo(ring[i][0], ring[i][1]);
      else ctx.lineTo(ring[i][0], ring[i][1]);
    }
    ctx.closePath();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = "#111";
    ctx.lineWidth = width + 1.2;
    ctx.stroke();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawBox(ctx, pose, K, color, width, alpha) {
    ctx.beginPath();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (var e = 0; e < BOX.length; e++) {
      var a = CORNERS[BOX[e][0]];
      var b = CORNERS[BOX[e][1]];
      var pa = project(pose, a[0], a[1], a[2], K);
      var pb = project(pose, b[0], b[1], b[2], K);
      if (!pa || !pb) continue;
      ctx.moveTo(pa[0], pa[1]);
      ctx.lineTo(pb[0], pb[1]);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  function drawTag(ctx, x, y, text, color, taken, imageW, imageH) {
    ctx.save();
    ctx.font = "600 16px sans-serif";
    var pad = 5;
    var width = Math.ceil(ctx.measureText(text).width + pad * 2);
    var height = 22;
    var left = Math.round(Math.max(0, Math.min(x, imageW - width)));
    var top = Math.round(y - height - 4);
    if (top < 0) top = 0;
    for (var step = 0; step < 16; step++) {
      var hit = false;
      for (var i = 0; i < taken.length; i++) {
        var box = taken[i];
        if (left < box.x + box.w && left + width > box.x && top < box.y + box.h && top + height > box.y) {
          hit = true;
          break;
        }
      }
      if (!hit) break;
      top += height + 2;
      if (top + height > imageH) top = 0;
    }
    taken.push({ x: left, y: top, w: width, h: height });
    ctx.fillStyle = color;
    ctx.fillRect(left, top, width, height);
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(text, left + pad, top + height / 2);
    ctx.restore();
  }

  function drawPhoto() {
    var canvas = $("wedge-pose-canvas");
    if (!canvas || !rgb || !scene) return;
    var height = scene.height;
    // Crop only the right background to match the neighboring 4:3 viewport.
    // 只裁去右侧背景以匹配相邻 4:3 视图，投影仍使用原图像素坐标。
    var width = Math.min(scene.width, Math.round(height * 4 / 3));
    canvas.width = width;
    canvas.height = height;
    // Use the rounded pixel ratio on both sides so their caption baselines align.
    // 两侧采用相同的整数像素比例，使视图高度和图注基线对齐。
    host.querySelector(".lmo-cloud-wrap").style.aspectRatio = width + " / " + height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(rgb, 0, 0, scene.width, scene.height);
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    var rows = hypotheses();
    var thin = 1.6;
    var thick = 3;
    for (var pass = 0; pass < 2; pass++) {
      for (var i = 0; i < rows.length; i++) {
        var on = i === selected;
        if ((pass === 1) !== on) continue;
        if (selected !== null && !on) continue;
        var alpha = selected === null || on ? 1 : 0.28;
        var color = hue(i);
        var rings = rows[i].rings || [];
        for (var r = 0; r < rings.length; r++) strokeRing(ctx, rings[r], color, on ? thick : thin, alpha);
        drawBox(ctx, rows[i].pose, scene.K, color, on ? thick : thin, alpha);
      }
    }
    if (method().score_kind !== "score_6d") return;
    var taken = [];
    var order = [];
    for (var j = 0; j < rows.length; j++) order.push(j);
    order.sort(function (a, b) { return rows[b].score_6d - rows[a].score_6d; });
    for (var k = 0; k < order.length; k++) {
      if (selected !== null && order[k] !== selected) continue;
      var item = rows[order[k]];
      var top = boxTop(item.pose, scene.K);
      if (!top) continue;
      drawTag(ctx, top[0], top[1], "6D " + item.score_6d.toFixed(3), hue(order[k]), taken, width, height);
    }
  }

  function hitIndex(px, py) {
    var rows = hypotheses();
    var hit = null;
    var area = Infinity;
    for (var i = 0; i < rows.length; i++) {
      var minX = Infinity;
      var minY = Infinity;
      var maxX = -Infinity;
      var maxY = -Infinity;
      var count = 0;
      for (var c = 0; c < CORNERS.length; c++) {
        var point = project(rows[i].pose, CORNERS[c][0], CORNERS[c][1], CORNERS[c][2], scene.K);
        if (!point) continue;
        count += 1;
        if (point[0] < minX) minX = point[0];
        if (point[1] < minY) minY = point[1];
        if (point[0] > maxX) maxX = point[0];
        if (point[1] > maxY) maxY = point[1];
      }
      if (count < 4) continue;
      if (px < minX || py < minY || px > maxX || py > maxY) continue;
      var next = (maxX - minX) * (maxY - minY);
      if (next < area) {
        area = next;
        hit = i;
      }
    }
    return hit;
  }

  function fillMethods() {
    var bar = $("wedge-methods");
    if (!bar) return;
    bar.innerHTML = "";
    for (var i = 0; i < scene.methods.length; i++) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = scene.methods[i].name;
      button.setAttribute("aria-pressed", i === methodIndex ? "true" : "false");
      button.addEventListener("click", (function (index) {
        return function () {
          methodIndex = index;
          selected = null;
          fillMethods();
          fillChips();
          drawPhoto();
          rebuildMeshes();
          frameView();
        };
      })(i));
      bar.appendChild(button);
    }
  }

  function fillChips() {
    var hostChips = $("wedge-chips");
    if (!hostChips) return;
    hostChips.innerHTML = "";
    var rows = hypotheses();
    var order = [];
    for (var i = 0; i < rows.length; i++) order.push(i);
    order.sort(function (a, b) { return rankScore(rows[b]) - rankScore(rows[a]); });
    for (var k = 0; k < order.length; k++) {
      var index = order[k];
      var button = document.createElement("button");
      button.type = "button";
      button.style.background = chipColor(rows[index]);
      button.setAttribute("aria-pressed", index === selected ? "true" : "false");
      button.textContent = chipText(rows[index]);
      button.addEventListener("click", (function (id) {
        return function () { select(selected === id ? null : id); };
      })(index));
      hostChips.appendChild(button);
    }
  }

  function syncMeshes() {
    if (!view) return;
    for (var i = 0; i < view.records.length; i++) {
      var record = view.records[i];
      var on = record.index === selected;
      record.mesh.visible = selected === null || on;
      record.mesh.material.opacity = selected === null || on ? 1 : 0.22;
      record.mesh.material.transparent = !(selected === null || on);
      record.mesh.material.depthWrite = selected === null || on;
      record.mesh.material.needsUpdate = true;
      record.shell.visible = on;
    }
  }

  function select(index) {
    selected = index;
    drawPhoto();
    fillChips();
    syncMeshes();
    if (index === null) frameView();
    else focusInstance(index);
  }

  function placeCamera() {
    var offset = new THREE.Vector3().setFromSpherical(view.spherical);
    view.camera.position.copy(view.target).add(offset);
    view.camera.lookAt(view.target);
  }

  function focusInstance(index) {
    if (!view) return;
    var mesh = null;
    for (var i = 0; i < view.records.length; i++) {
      if (view.records[i].index === index) {
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

  function frameView() {
    if (!view || !view.lines) return;
    var focus = new THREE.Box3().setFromObject(view.lines);
    if (view.points) {
      var cloudBox = new THREE.Box3().setFromObject(view.points);
      if (!cloudBox.isEmpty()) focus.union(cloudBox);
    }
    if (focus.isEmpty()) return;
    var center = focus.getCenter(new THREE.Vector3());
    var depth = Math.max(0.2, -center.z);
    var fy = scene.K[4];
    view.camera.fov = 2 * Math.atan(scene.height / (2 * fy)) * 180 / Math.PI;
    view.target.set(0, 0, -depth);
    view.spherical.radius = depth;
    view.spherical.phi = Math.PI / 2;
    view.spherical.theta = 0;
    placeCamera();
    view.camera.near = 0.01;
    view.camera.far = Math.max(depth * 8, 3);
    view.camera.updateProjectionMatrix();
  }

  function showCloud(buffer) {
    var n = scene.cloud_count;
    var xyz = new Float32Array(buffer, 0, n * 3);
    var colors = new Uint8Array(buffer, n * 12, n * 3);
    var pos = new Float32Array(n * 3);
    var col = new Float32Array(n * 3);
    for (var i = 0; i < n; i++) {
      pos[i * 3] = xyz[i * 3];
      pos[i * 3 + 1] = -xyz[i * 3 + 1];
      pos[i * 3 + 2] = -xyz[i * 3 + 2];
      col[i * 3] = colors[i * 3] / 255;
      col[i * 3 + 1] = colors[i * 3 + 1] / 255;
      col[i * 3 + 2] = colors[i * 3 + 2] / 255;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    view.points = new THREE.Points(geo, new THREE.PointsMaterial({
      size: 0.0045,
      vertexColors: true,
      sizeAttenuation: true
    }));
    view.scene3.add(view.points);
    frameView();
  }

  function rebuildMeshes() {
    if (!view || !view.geometry) return;
    while (view.lines.children.length) view.lines.remove(view.lines.children[0]);
    view.records = [];
    var rows = hypotheses();
    for (var k = 0; k < rows.length; k++) {
      var mesh = new THREE.Mesh(view.geometry, new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2
      }));
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(poseMatrix(rows[k].pose));
      mesh.userData.index = k;
      view.lines.add(mesh);
      var shell = new THREE.Mesh(view.geometry, new THREE.MeshBasicMaterial({
        color: 0xc23b2e,
        side: THREE.BackSide
      }));
      shell.matrixAutoUpdate = false;
      shell.matrix.multiplyMatrices(poseMatrix(rows[k].pose), new THREE.Matrix4().makeScale(1.04, 1.04, 1.04));
      shell.visible = false;
      view.lines.add(shell);
      view.records.push({ index: k, mesh: mesh, shell: shell });
    }
    syncMeshes();
  }

  function showMesh(meta, buffer) {
    var n = meta.vertices;
    var f = meta.faces;
    var positions = new Float32Array(buffer, 0, n * 3);
    var indices = new Uint32Array(buffer, n * 12, f * 3);
    var rgbBytes = new Uint8Array(buffer, n * 12 + f * 12, n * 3);
    var color = new Float32Array(n * 3);
    for (var i = 0; i < n * 3; i++) color[i] = rgbBytes[i] / 255;
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(color, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    view.geometry = geo;
    rebuildMeshes();
    frameView();
  }

  function resizeView() {
    if (!view) return;
    var canvas = view.renderer.domElement;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    if (w < 2 || h < 2) return;
    view.renderer.setSize(w, h, false);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
  }

  function ensureView() {
    if (view || typeof THREE === "undefined") return;
    var canvas = $("wedge-cloud");
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0xd5dde6, 1);
    var scene3 = new THREE.Scene();
    var camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
    camera.up.set(0, 1, 0);
    var lines = new THREE.Group();
    scene3.add(lines);
    var raycaster = new THREE.Raycaster();
    view = {
      renderer: renderer,
      scene3: scene3,
      camera: camera,
      lines: lines,
      raycaster: raycaster,
      target: new THREE.Vector3(),
      spherical: new THREE.Spherical(1, Math.PI / 2, 0),
      records: [],
      drag: null,
      geometry: null,
      points: null
    };
    canvas.addEventListener("pointerdown", function (event) {
      view.drag = { x: event.clientX, y: event.clientY, moved: false, pan: event.button === 2 || event.shiftKey };
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
    canvas.addEventListener("pointerup", function (event) {
      var drag = view.drag;
      view.drag = null;
      if (!drag || drag.moved) return;
      var rect = canvas.getBoundingClientRect();
      var ndc = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1
      );
      raycaster.setFromCamera(ndc, camera);
      var hits = raycaster.intersectObjects(view.lines.children, false);
      for (var i = 0; i < hits.length; i++) {
        if (hits[i].object.userData.index !== undefined) {
          select(hits[i].object.userData.index);
          return;
        }
      }
    });
    canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      var factor = event.deltaY > 0 ? 1.08 : 0.92;
      view.spherical.radius = Math.max(0.05, Math.min(8, view.spherical.radius * factor));
      placeCamera();
    }, { passive: false });
    canvas.addEventListener("dblclick", function () { select(null); });
    canvas.addEventListener("contextmenu", function (event) { event.preventDefault(); });
    window.addEventListener("resize", resizeView);
    function frame() {
      resizeView();
      renderer.render(scene3, camera);
    }
    WAPRRender.register(canvas, frame);
  }

  function boot() {
    var photo = $("wedge-pose-canvas");
    if (!photo) return;
    photo.addEventListener("click", function (event) {
      if (!scene || !rgb) return;
      var rect = photo.getBoundingClientRect();
      // Map clicks to the cropped canvas; its origin remains the image origin.
      // 点击映射到裁切画布的像素坐标；裁切后原点仍与原图一致。
      var px = (event.clientX - rect.left) / rect.width * photo.width;
      var py = (event.clientY - rect.top) / rect.height * photo.height;
      var index = hitIndex(px, py);
      if (index !== null) select(selected === index ? null : index);
    });
    WAPRAssets.fetch(root + "scene.json?v=3").then(function (res) { return res.json(); }).then(function (data) {
      scene = data;
      // Local Line2D/PPF reproductions are retained as timing references only.
      // 本地 Line2D/PPF 复现仅作耗时参考，不进入图像、网格和候选控件。
      scene.methods = data.methods.filter(function (item) { return item.name === "WAPR"; });
      fillMethods();
      fillChips();
      rgb = new Image();
      rgb.onload = drawPhoto;
      rgb.src = root + "rgb.jpg";
      ensureView();
      WAPRAssets.fetch(root + "cloud.bin").then(function (res) { return res.arrayBuffer(); }).then(showCloud);
      WAPRAssets.fetch(root + "mesh.json").then(function (res) { return res.json(); }).then(function (meta) {
        return WAPRAssets.fetch(root + "mesh.bin").then(function (res) { return res.arrayBuffer(); }).then(function (buffer) {
          showMesh(meta, buffer);
        });
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
