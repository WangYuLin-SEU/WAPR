// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

/* Loc viewer. Left contours are images baked from each object's rendered-depth mask.
   定位查看器。左侧轮廓是事先用每个物体的渲染深度 mask 画好的图片。
   Point positions in cloud.bin are OpenCV camera meters. The view uses y-up: (x, -y, -z).
   cloud.bin 里的点是 OpenCV 相机系、米。视图用 y 朝上：(x, -y, -z)。
   The 3D camera starts at that same sensor: origin, looking down -Z, projection from scene.K.
   3D 相机一开始就放在同一台传感器上：原点，朝 -Z，投影用 scene.K。
   Mesh bins are centered object meters, then the pose, then the same y-up map.
   网格 bin 是中心化后的物体米制坐标，再乘位姿，然后做同样的 y 朝上变换。 */
(function () {
  var BOX = [];
  for (var corner = 0; corner < 8; corner++) {
    for (var bit = 0; bit < 3; bit++) {
      var other = corner ^ (1 << bit);
      if (other > corner) BOX.push([corner, other]);
    }
  }

  var assetV = "?v=45";
  var datasets = [];
  var datasetIndex = 0;
  var sceneIndex = 0;
  var loadToken = 0;
  var meshToken = 0;
  var meshAssets = {};
  var scene = null;
  var poseMode = "pred";
  var selected = 1;
  var rgbImage = null;
  var layers = null;
  var view = null;
  var strip = null;
  var stripToken = 0;

  function $(id) { return document.getElementById(id); }

  function setPanelStatus(id, english, chinese, failed) {
    var status = $(id);
    if (!status) return;
    status.hidden = !english;
    if (!english) return;
    status.querySelector(".en").textContent = english;
    status.querySelector(".zh").textContent = chinese;
    var retry = status.querySelector("button");
    retry.hidden = !failed;
    retry.onclick = failed ? function () {
      if (datasets.length) loadCurrent();
      else window.location.reload();
    } : null;
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

  function drawBoxes(ctx) {
    var K = scene.K;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    var objects = scene.objects.slice();
    objects.sort(function (a, b) { return (a.id === selected) - (b.id === selected); });
    for (var n = 0; n < objects.length; n++) {
      var obj = objects[n];
      var pose = poseOf(obj);
      var on = obj.id === selected;
      ctx.strokeStyle = poseMode === "gt" ? "#17833d" : "rgb(" + obj.color[0] + "," + obj.color[1] + "," + obj.color[2] + ")";
      ctx.lineWidth = on ? Math.max(2.5, scene.width / 240) : Math.max(1.25, scene.width / 420);
      ctx.beginPath();
      for (var e = 0; e < BOX.length; e++) {
        var a = project(pose, obj.corners, BOX[e][0], K);
        var b = project(pose, obj.corners, BOX[e][1], K);
        if (!a || !b) continue;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
    }
  }

  function poseOf(obj) {
    return poseMode === "gt" ? obj.gt_pose : obj.pred_pose;
  }

  function drawPhoto() {
    var canvas = $("demo-image");
    if (!canvas || !scene || !rgbImage || !rgbImage.complete || !rgbImage.naturalWidth) return;
    canvas.width = scene.width;
    canvas.height = scene.height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(rgbImage, 0, 0);
    if (layers) {
      var contour = poseMode === "gt" ? layers.gt : layers.pred;
      if (contour) ctx.drawImage(contour, 0, 0);
    }
    drawBoxes(ctx);
  }

  function decodePick(img, token, assign) {
    var scratch = document.createElement("canvas");
    scratch.width = scene.width;
    scratch.height = scene.height;
    var ctx = scratch.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(img, 0, 0, scene.width, scene.height);
    var data = ctx.getImageData(0, 0, scene.width, scene.height).data;
    var ids = new Uint8Array(scene.width * scene.height);
    var objects = scene.objects;
    for (var i = 0; i < ids.length; i++) {
      var byte = data[i * 4];
      if (!byte) continue;
      var best = 0;
      var bestDist = 6;
      for (var n = 0; n < objects.length; n++) {
        var dist = Math.abs(byte - objects[n].id);
        if (dist < bestDist) {
          bestDist = dist;
          best = objects[n].id;
        }
      }
      ids[i] = best;
    }
    if (token !== loadToken) return;
    assign(ids);
  }

  function loadLayers(base, token) {
    var pack = { pred: null, gt: null, predPick: null, gtPick: null };
    layers = pack;
    function loadImage(src, slot) {
      var img = new Image();
      img.onload = function () {
        if (token !== loadToken) return;
        pack[slot] = img;
        drawPhoto();
      };
      img.src = src;
    }
    function loadPick(src, slot) {
      var img = new Image();
      img.onload = function () {
        decodePick(img, token, function (ids) { pack[slot] = ids; });
      };
      img.src = src;
    }
    loadImage(base + "/contour_pred.png" + assetV, "pred");
    loadImage(base + "/contour_gt.png" + assetV, "gt");
    loadPick(base + "/pick_pred.png" + assetV, "predPick");
    loadPick(base + "/pick_gt.png" + assetV, "gtPick");
  }

  function fillChips() {
    var host = $("demo-chips");
    if (!host || !scene) return;
    host.innerHTML = "";
    for (var i = 0; i < scene.objects.length; i++) {
      var obj = scene.objects[i];
      var button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", obj.id === selected ? "true" : "false");
      button.textContent = "obj " + obj.obj_id + "  6D " + obj.score_6d.toFixed(3);
      button.addEventListener("click", (function (id) {
        return function () { selectObject(id); };
      })(obj.id));
      host.appendChild(button);
    }
  }

  function fillNote() {
    if (!scene) return;
    var en = "Scene " + scene.scene_id + ", frame " + scene.im_id + ". " +
      scene.n_inst + " known instances, kept by score_6d.";
    var zh = "场景 " + scene.scene_id + "，帧 " + scene.im_id + "。已知 " +
      scene.n_inst + " 个实例，按 score_6d 保留。";
    if (scene.val_scene && scene.name === "itodd") {
      en += " The public test split has no ground-truth poses, so this frame is the validation scene.";
      zh += " 公开测试集没有真值位姿，所以这一帧来自验证场景。";
    } else if (scene.val_scene) {
      en += " This test copy has no scene_gt, so this frame is the validation split.";
      zh += " 这份测试拷贝没有 scene_gt，所以这一帧来自验证划分。";
    }
    var enNode = $("demo-note-en");
    var zhNode = $("demo-note-zh");
    if (enNode) enNode.textContent = en;
    if (zhNode) zhNode.textContent = zh;
  }

  function meshKey(name, objId) {
    var id = String(objId);
    while (id.length < 6) id = "0" + id;
    return name + "/obj_" + id;
  }

  function srgbByteToLinear(byte) {
    var c = byte / 255;
    if (c <= 0.04045) return c / 12.92;
    return Math.pow((c + 0.055) / 1.055, 2.4);
  }

  function poseMatrix(pose) {
    // Row-major. Rows 1 and 2 flip the OpenCV camera into the y-up view.
    // 行主序。第 1、2 行把 OpenCV 相机系翻成 y 朝上的视图。
    var m = new THREE.Matrix4();
    m.set(
      pose[0], pose[1], pose[2], pose[3],
      -pose[4], -pose[5], -pose[6], -pose[7],
      -pose[8], -pose[9], -pose[10], -pose[11],
      0, 0, 0, 1
    );
    return m;
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

  function loadMeshAsset(name, objId) {
    var key = meshKey(name, objId);
    if (meshAssets[key]) return meshAssets[key];
    var pending = WAPRAssets.fetch("demo/meshes/" + key + ".json").then(function (res) {
      if (!res.ok) throw new Error(key);
      return res.json();
    }).then(function (meta) {
      return WAPRAssets.fetch("demo/meshes/" + key + ".bin").then(function (res) {
        if (!res.ok) throw new Error(key);
        return res.arrayBuffer();
      }).then(function (buffer) {
        return decodeMesh(meta, buffer, key);
      });
    }).catch(function (err) {
      delete meshAssets[key];
      throw err;
    });
    meshAssets[key] = pending;
    return pending;
  }

  function shellColor(rgb) {
    var color = new THREE.Color();
    color.setStyle("rgb(" + rgb[0] + "," + rgb[1] + "," + rgb[2] + ")");
    return color;
  }

  function clearMeshes() {
    if (!view) return;
    var group = view.lines;
    while (group.children.length) {
      var old = group.children[group.children.length - 1];
      group.remove(old);
      if (old.userData.disposeGeometry && old.geometry) old.geometry.dispose();
      disposeMaterial(old.material);
    }
    view.records = [];
  }

  function makeMaterial(asset) {
    // Unlit albedo. Offset the mesh toward the camera so it wins against the depth cloud.
    // 无光照反照率。网格略向相机偏移，避免和深度点云抢深度。
    return new THREE.MeshBasicMaterial({
      map: asset.texture || null,
      vertexColors: asset.mode === "vertex",
      color: 0xffffff,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2
    });
  }

  function placeMeshes(loaded) {
    clearMeshes();
    for (var i = 0; i < loaded.length; i++) {
      var item = loaded[i];
      if (!item) continue;
      var mesh = new THREE.Mesh(item.asset.geometry, makeMaterial(item.asset));
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(poseMatrix(poseOf(item.obj)));
      mesh.userData.id = item.obj.id;
      mesh.renderOrder = 1;
      view.lines.add(mesh);
      // Slightly larger back faces. Only the selected object shows this rim.
      // 略放大的背面。只有选中的物体显示这圈边。
      var shell = new THREE.Mesh(item.asset.geometry, new THREE.MeshBasicMaterial({
        color: poseMode === "gt" ? 0x17833d : shellColor(item.obj.color),
        side: THREE.BackSide,
        toneMapped: false,
        polygonOffset: true,
        polygonOffsetFactor: -4,
        polygonOffsetUnits: -4
      }));
      shell.matrixAutoUpdate = false;
      shell.frustumCulled = false;
      shell.userData.id = item.obj.id;
      shell.renderOrder = 2;
      shell.visible = false;
      view.lines.add(shell);
      view.records.push({ obj: item.obj, mesh: mesh, shell: shell });
    }
    view.lines.updateMatrixWorld(true);
    syncMeshes();
    rebuildStrip();
    if (view.orbitId) focusInstance(view.orbitId);
  }

  function syncMeshes() {
    if (!view || !view.records) return;
    for (var i = 0; i < view.records.length; i++) {
      var record = view.records[i];
      var pose = poseMatrix(poseOf(record.obj));
      var on = record.obj.id === selected;
      record.mesh.matrix.copy(pose);
      record.mesh.material.color.setHex(on ? 0xffffff : 0x9a9a9a);
      record.shell.matrix.multiplyMatrices(pose, view.shellScale);
      record.shell.material.color.set(poseMode === "gt" ? "#17833d" : shellColor(record.obj.color));
      record.shell.visible = on;
    }
    view.lines.updateMatrixWorld(true);
  }

  /* A click in the cloud, inside the same picture as the left photo.
     NDC is the sensor viewport, not the whole canvas: the panel is taller
     than the photo and the extra band sits at the bottom.
     点云里的一次点击，范围和左边那张图相同。
     NDC 用传感器视口，不用整张画布：面板比照片更高，多出来的一段在下面。 */
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

  function uniqueObjects(objects) {
    var seen = {};
    var list = [];
    for (var i = 0; i < objects.length; i++) {
      var objId = objects[i].obj_id;
      if (seen[objId]) continue;
      seen[objId] = true;
      list.push(objects[i]);
    }
    return list;
  }

  function ensureStrip() {
    if (strip || typeof THREE === "undefined") return;
    var canvas = $("mesh-strip-canvas");
    if (!canvas) return;
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
    renderer.autoClear = false;
    strip = { renderer: renderer, canvas: canvas, cells: [], drag: null, cssW: 0, cssH: 0, gap: 8 };
    canvas.addEventListener("pointerdown", function (event) {
      var cell = stripCellAt(event);
      if (!cell) return;
      strip.drag = { cell: cell, x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!strip.drag) return;
      var dx = event.clientX - strip.drag.x;
      var dy = event.clientY - strip.drag.y;
      strip.drag.x = event.clientX;
      strip.drag.y = event.clientY;
      var cell = strip.drag.cell;
      cell.spherical.theta -= dx * 0.01;
      cell.spherical.phi = Math.max(0.08, Math.min(Math.PI - 0.08, cell.spherical.phi - dy * 0.01));
      placeStripCamera(cell);
    });
    function endDrag() { strip.drag = null; }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", function (event) {
      var cell = stripCellAt(event);
      if (!cell) return;
      event.preventDefault();
      var factor = event.deltaY > 0 ? 1.08 : 0.92;
      cell.spherical.radius = Math.max(cell.minRadius, Math.min(cell.maxRadius, cell.spherical.radius * factor));
      placeStripCamera(cell);
    }, { passive: false });
    canvas.addEventListener("contextmenu", function (event) { event.preventDefault(); });
    function frame() {
      renderStrip();
    }
    WAPRRender.register(canvas, frame);
  }

  function placeStripCamera(cell) {
    // 正交取景：半幅等于当前缩放，距离只负责绕包围球转动。
    // Orthographic frame: half-extent is the zoom; distance only orbits the bounding sphere.
    var half = cell.spherical.radius;
    var dist = cell.bound * 4;
    var offset = new THREE.Vector3().setFromSpherical(
      new THREE.Spherical(dist, cell.spherical.phi, cell.spherical.theta)
    );
    cell.camera.position.copy(cell.target).add(offset);
    cell.camera.lookAt(cell.target);
    cell.camera.left = -half;
    cell.camera.right = half;
    cell.camera.top = half;
    cell.camera.bottom = -half;
    cell.camera.near = Math.max(cell.bound * 0.01, dist - cell.bound * 1.5);
    cell.camera.far = dist + cell.bound * 1.5;
    cell.camera.updateProjectionMatrix();
  }

  function stripCellAt(event) {
    if (!strip || !strip.cells.length) return null;
    var rect = strip.canvas.getBoundingClientRect();
    var x = (event.clientX - rect.left) * (strip.cssW / rect.width);
    var y = (event.clientY - rect.top) * (strip.cssH / rect.height);
    if (y < 0 || y > strip.cssH) return null;
    for (var i = 0; i < strip.cells.length; i++) {
      var cell = strip.cells[i];
      if (x >= cell.x && x < cell.x + cell.size) return cell;
    }
    return null;
  }

  function clearStripCells() {
    if (!strip) return;
    for (var i = 0; i < strip.cells.length; i++) {
      var cell = strip.cells[i];
      cell.scene.remove(cell.mesh);
      disposeMaterial(cell.mesh.material);
      if (cell.floor) {
        cell.scene.remove(cell.floor);
        if (window.WAPRStudio) WAPRStudio.dispose(cell.floor);
      }
    }
    strip.cells = [];
    strip.layoutKey = "";
    var labels = $("mesh-strip-labels");
    if (labels) labels.innerHTML = "";
    var frames = $("mesh-strip-frames");
    if (frames) frames.innerHTML = "";
  }

  function markStripLabels() {
    if (!strip || !scene) return;
    var labels = $("mesh-strip-labels");
    if (!labels) return;
    var selectedObj = null;
    for (var i = 0; i < scene.objects.length; i++) {
      if (scene.objects[i].id === selected) selectedObj = scene.objects[i].obj_id;
    }
    var frames = $("mesh-strip-frames");
    for (var c = 0; c < strip.cells.length; c++) {
      var on = strip.cells[c].objId === selectedObj;
      var span = labels.children[c];
      if (span) span.style.fontWeight = on ? "700" : "500";
      var frame = frames && frames.children[c];
      if (frame) frame.classList.toggle("is-on", on);
    }
  }

  function syncStripFrames() {
    var host = $("mesh-strip-frames");
    if (!host || !strip) return;
    var n = strip.cells.length;
    while (host.children.length > n) host.removeChild(host.lastChild);
    while (host.children.length < n) {
      var frame = document.createElement("div");
      frame.className = "mesh-card";
      host.appendChild(frame);
    }
    host.style.gap = strip.gap + "px";
    for (var i = 0; i < n; i++) {
      var frame = host.children[i];
      var size = strip.cells[i].size + "px";
      if (frame.style.width === size) continue;
      frame.style.flex = "0 0 " + size;
      frame.style.width = size;
      frame.style.height = size;
    }
  }

  function layoutStrip() {
    if (!strip) return;
    var n = strip.cells.length;
    var canvas = strip.canvas;
    var stage = $("mesh-strip-stage");
    var caption = $("mesh-strip-caption");
    if (!n) {
      canvas.style.display = "none";
      if (stage) stage.style.display = "none";
      if (caption) caption.hidden = true;
      return;
    }
    canvas.style.display = "block";
    if (stage) stage.style.display = "block";
    if (caption) caption.hidden = false;
    var photo = $("demo-image");
    var photoH = photo && photo.clientHeight > 40 ? photo.clientHeight : 360;
    var cap = Math.round(photoH * 0.5);
    var section = $("showcase");
    var maxW = section ? section.clientWidth : 1000;
    var gap = 16;
    var fit = Math.floor((maxW - gap * (n - 1)) / n);
    var cell = Math.min(cap, fit);
    if (cell < 48) cell = Math.max(36, fit);
    var cssW = n * cell + (n - 1) * gap;
    var cssH = cell;
    var layoutKey = cssW + "x" + cssH + "x" + n;
    strip.cssW = cssW;
    strip.cssH = cssH;
    strip.gap = gap;
    for (var i = 0; i < n; i++) {
      strip.cells[i].x = i * (cell + gap);
      strip.cells[i].size = cell;
    }
    syncStripFrames();
    if (strip.layoutKey === layoutKey) return;
    strip.layoutKey = layoutKey;
    var dpr = Math.min(window.devicePixelRatio || 1, 2);
    strip.renderer.setPixelRatio(dpr);
    strip.renderer.setSize(cssW, cssH, false);
    canvas.style.width = cssW + "px";
    canvas.style.height = cssH + "px";
    if (caption) caption.style.textAlign = "center";
    var labels = $("mesh-strip-labels");
    if (!labels) return;
    labels.style.display = "flex";
    labels.style.justifyContent = "center";
    labels.style.width = cssW + "px";
    labels.style.gap = gap + "px";
    labels.style.marginTop = "4px";
    var spans = labels.children;
    for (var s = 0; s < spans.length; s++) {
      spans[s].style.display = "block";
      spans[s].style.flex = "0 0 " + cell + "px";
      spans[s].style.width = cell + "px";
      spans[s].style.overflow = "hidden";
      spans[s].style.textAlign = "center";
      spans[s].style.whiteSpace = "nowrap";
    }
  }

  function renderStrip() {
    if (!strip || !strip.cells.length) return;
    layoutStrip();
    var renderer = strip.renderer;
    var fullW = strip.cssW;
    var fullH = strip.cssH;
    renderer.setScissorTest(true);
    renderer.setViewport(0, 0, fullW, fullH);
    renderer.setScissor(0, 0, fullW, fullH);
    renderer.clear(true, true, true);
    for (var i = 0; i < strip.cells.length; i++) {
      var cell = strip.cells[i];
      renderer.setViewport(cell.x, 0, cell.size, cell.size);
      renderer.setScissor(cell.x, 0, cell.size, cell.size);
      renderer.clear(true, true, true);
      renderer.render(cell.scene, cell.camera);
    }
  }

  function rebuildStrip() {
    ensureStrip();
    if (!strip || !scene) return;
    var token = ++stripToken;
    var objects = uniqueObjects(scene.objects);
    var jobs = [];
    for (var i = 0; i < objects.length; i++) {
      jobs.push((function (obj) {
        return loadMeshAsset(scene.name, obj.obj_id).then(function (asset) {
          return { obj: obj, asset: asset };
        });
      })(objects[i]));
    }
    Promise.all(jobs).then(function (loaded) {
      if (token !== stripToken || !strip) return;
      clearStripCells();
      var labels = $("mesh-strip-labels");
      for (var n = 0; n < loaded.length; n++) {
        var item = loaded[n];
        var scene3 = new THREE.Scene();
        var mesh = new THREE.Mesh(item.asset.geometry, new THREE.MeshBasicMaterial({
          map: item.asset.texture || null,
          vertexColors: item.asset.mode === "vertex",
          side: THREE.DoubleSide
        }));
        // BOP model frame is Z-up. This view is Y-up, so Rx(-90°): (x, y, z) -> (x, z, -y).
        // BOP 模型系 Z 朝上。这里的视图 Y 朝上，因此绕 X 转 -90°：(x, y, z) -> (x, z, -y)。
        mesh.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
        mesh.material.polygonOffset = true;
        mesh.material.polygonOffsetFactor = -1;
        mesh.material.polygonOffsetUnits = -1;
        scene3.add(mesh);
        mesh.updateMatrixWorld(true);
        var box = new THREE.Box3().setFromObject(mesh);
        item.asset.geometry.computeBoundingSphere();
        var sphere = item.asset.geometry.boundingSphere;
        var bound = Math.max(sphere.radius, 1.0e-4);
        var target = sphere.center.clone().applyQuaternion(mesh.quaternion);
        // Each card is its own grid viewport. The pad stays while the camera orbits.
        // 每张卡片是一块网格视口。相机绕着转，垫子留在原地。
        var floor = null;
        if (window.WAPRStudio) {
          var floorY = box.min.y - bound * 0.02;
          floor = WAPRStudio.placeFloor(scene3, null, target.x, floorY, target.z, bound * 6);
        }
        var camera = new THREE.OrthographicCamera(-bound, bound, bound, -bound, bound * 0.01, bound * 20);
        camera.up.set(0, 1, 0);
        var cell = {
          objId: item.obj.obj_id,
          scene: scene3,
          mesh: mesh,
          floor: floor,
          camera: camera,
          target: target,
          bound: bound,
          spherical: new THREE.Spherical(bound * 1.65, 1.02, 0.55),
          minRadius: bound * 0.55,
          maxRadius: bound * 6,
          x: 0,
          size: 1
        };
        placeStripCamera(cell);
        strip.cells.push(cell);
        if (labels) {
          var span = document.createElement("span");
          span.textContent = "obj " + item.obj.obj_id;
          labels.appendChild(span);
        }
      }
      layoutStrip();
      markStripLabels();
    }).catch(function () {});
  }

  function loadPoseMeshes(name, objects) {
    var jobs = [];
    for (var i = 0; i < objects.length; i++) {
      jobs.push((function (obj) {
        return loadMeshAsset(name, obj.obj_id).then(function (asset) {
          return { obj: obj, asset: asset };
        }).catch(function () {
          return null;
        });
      })(objects[i]));
    }
    return Promise.all(jobs);
  }

  function disposeMaterial(material) {
    if (!material) return;
    if (Array.isArray(material)) {
      for (var i = 0; i < material.length; i++) material[i].dispose();
    } else {
      material.dispose();
    }
  }

  function disposeCloud() {
    if (!view) return;
    if (view.points) {
      view.scene.remove(view.points);
      view.points.geometry.dispose();
      disposeMaterial(view.points.material);
      view.points = null;
    }
    if (view.grid) {
      view.scene.remove(view.grid);
      if (window.WAPRStudio) WAPRStudio.dispose(view.grid);
      view.grid = null;
    }
  }

  function showCloud(buffer) {
    disposeCloud();
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
    view.pointSize = 0.004;
    var mat = new THREE.PointsMaterial({
      size: view.pointSize,
      vertexColors: true,
      sizeAttenuation: true
    });
    view.points = new THREE.Points(geo, mat);
    view.scene.add(view.points);
    clearMeshes();
    // Sensor camera from this cloud. Meshes arrive later and must not retarget it.
    // 用这帧点云摆传感器相机。网格稍后到来，不能再把相机拽走。
    frameView();
    var token = ++meshToken;
    var name = scene.name;
    var objects = scene.objects.slice();
    loadPoseMeshes(name, objects).then(function (loaded) {
      if (token !== meshToken || !view) return;
      placeMeshes(loaded);
    });
  }

  function frameView() {
    if (!view || !scene || !scene.K) return;
    var focus = new THREE.Box3();
    if (view.points) {
      view.points.geometry.computeBoundingBox();
      focus.copy(view.points.geometry.boundingBox);
    }
    if (focus.isEmpty()) return;
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
    // A few image pixels at the cloud depth. Big enough to read as the photo,
    // small enough that a fine board does not turn into stripes.
    // 深度处大约几个图像像素。大到能看成那张照片，又不会把细密板子叠成条纹。
    view.pointSize = Math.max(depth * 4.8 / K[4], 0.0012);
    if (view.points) view.points.material.size = view.pointSize;
    // No studio floor here. The depth cloud is the table, and a grid through it
    // moires into stripes that jump on a small orbit.
    // 这里不铺地面格子。桌子就是深度点云，格子穿过去会叠成斜条，轻轻一转就跳。
    if (view.grid) {
      view.scene.remove(view.grid);
      if (window.WAPRStudio) WAPRStudio.dispose(view.grid);
      view.grid = null;
    }
    // Until a click, the orbit center stays on the optical axis.
    // The spherical offset matches that pose, so the first drag does not snap.
    // 还没点物体时，旋转中心留在光轴上。球坐标和这个姿态一致，第一下拖拽不会跳。
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
    // Same K as the left RGB, after the (x, -y, -z) flip. NDC y is up.
    // 和左侧 RGB 同一套 K，已经做了 (x, -y, -z)。NDC 的 y 朝上。
    view.camera.projectionMatrix.set(
      2 * photo.fx / photo.width, 0, 1 - 2 * photo.cx / photo.width, 0,
      0, 2 * photo.fy / photo.height, 2 * photo.cy / photo.height - 1, 0,
      0, 0, c, d,
      0, 0, -1, 0
    );
    view.camera.projectionMatrixInverse.copy(view.camera.projectionMatrix).invert();
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

  function resizeView() {
    if (!view) return;
    var canvas = view.renderer.domElement;
    var w = canvas.clientWidth;
    var h = canvas.clientHeight;
    if (w < 2 || h < 2) return;
    // The cloud frame is the RGB frame. Same width from the equal columns,
    // same height as the photo. The chips stay under the photo only.
    // 点云框和 RGB 同一个框。两列等宽，高度跟照片一样。结果按钮只留在照片下面。
    var photo = document.getElementById("demo-image");
    var wrap = canvas.parentElement;
    if (photo && wrap && photo.clientHeight > 2) {
      var frameH = photo.clientHeight;
      if (Math.abs(wrap.clientHeight - frameH) > 1 || wrap.style.alignSelf !== "start") {
        wrap.style.boxSizing = "content-box";
        wrap.style.height = frameH + "px";
        wrap.style.minHeight = "0";
        wrap.style.alignSelf = "start";
        w = canvas.clientWidth;
        h = canvas.clientHeight;
        if (w < 2 || h < 2) return;
        view.cssW = 0;
      }
    }
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
    // Fit the sensor frame in this box. It is the same size as the photo,
    // so the fit is the whole panel. Viewport numbers are CSS pixels.
    // three.js applies the pixel ratio. Buffer pixels crop the upper right.
    // 传感器画面放进这个盒子。盒子和照片一样大，所以画面铺满面板。
    // 视口用 CSS 像素。three.js 自己会乘像素比。传缓冲区像素会把右上角裁掉。
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
    var canvas = $("demo-cloud");
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
      target: new THREE.Vector3(),
      spherical: new THREE.Spherical(1, 1.1, 0.4),
      pointSize: 0.003,
      records: [],
      shellScale: new THREE.Matrix4().makeScale(1.045, 1.045, 1.045),
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
    canvas.addEventListener("pointerup", function (event) {
      var drag = view.drag;
      view.drag = null;
      if (!drag || drag.moved || drag.pan) return;
      var id = meshAt(event);
      if (id) selectObject(id);
    });
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

  function selectObject(id) {
    selected = id;
    drawPhoto();
    fillChips();
    syncMeshes();
    markStripLabels();
    focusInstance(id);
  }

  function setPose(mode) {
    poseMode = mode;
    var buttons = document.querySelectorAll("#showcase .pose-switch button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-pose") === mode ? "true" : "false");
    }
    syncMeshes();
    drawPhoto();
    if (view && view.orbitId) focusInstance(view.orbitId);
  }

  function markPressed(selector, index) {
    var buttons = document.querySelectorAll(selector);
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", i === index ? "true" : "false");
    }
  }

  function loadCurrent() {
    var token = ++loadToken;
    ++meshToken;
    ++stripToken;
    markPressed("#demo-pager button", datasetIndex);
    markPressed("#scene-pager button", sceneIndex);
    var sample = datasets[datasetIndex].scenes[sceneIndex];
    var base = "demo/" + sample.dir;
    scene = null;
    layers = null;
    rgbImage = null;
    var canvas = $("demo-image");
    canvas.width = 640;
    canvas.height = 480;
    // The RGB starts loading alongside scene metadata and already has the
    // correct 4:3 panel ratio for every scene in this viewer.
    // RGB 与场景元数据并行加载；本查看器所有帧的面板比例均为 4:3。
    canvas.style.backgroundImage = 'url("' + base + '/rgb.jpg' + assetV + '")';
    $("demo-chips").innerHTML = "";
    $("demo-note-en").textContent = "";
    $("demo-note-zh").textContent = "";
    disposeCloud();
    clearMeshes();
    clearStripCells();
    if (strip) layoutStrip();
    setPanelStatus("demo-photo-status", "Loading frame…", "正在加载图像…", false);
    setPanelStatus("demo-cloud-status", "Loading point cloud…", "正在加载点云…", false);
    WAPRAssets.fetch(base + "/scene.json" + assetV).then(function (res) {
      if (!res.ok) throw new Error("Scene metadata: HTTP " + res.status);
      return res.json();
    }).then(function (data) {
      if (token !== loadToken) return;
      scene = data;
      selected = scene.objects.length ? scene.objects[0].id : 1;
      canvas.width = scene.width;
      canvas.height = scene.height;
      fillNote();
      fillChips();
      loadLayers(base, token);
      rgbImage = new Image();
      rgbImage.onload = function () {
        if (token !== loadToken) return;
        drawPhoto();
        setPanelStatus("demo-photo-status", "", "", false);
      };
      rgbImage.onerror = function () {
        if (token !== loadToken) return;
        setPanelStatus("demo-photo-status", "Frame image could not load.", "图像加载失败。", true);
      };
      rgbImage.src = base + "/rgb.jpg" + assetV;
      WAPRAssets.fetch(base + "/cloud.bin" + assetV).then(function (res) {
        if (!res.ok) throw new Error("Point cloud: HTTP " + res.status);
        return res.arrayBuffer();
      }).then(function (buf) {
        if (token !== loadToken) return;
        if (buf.byteLength < scene.cloud_count * 15) throw new Error("Incomplete point cloud");
        ensureView();
        showCloud(buf);
        setPanelStatus("demo-cloud-status", "", "", false);
      }).catch(function (err) {
        if (token !== loadToken) return;
        console.error("WAPR localization point cloud:", err);
        setPanelStatus("demo-cloud-status", "Point cloud or 3D view could not load.", "点云或 3D 视图加载失败。", true);
      });
    }).catch(function (err) {
      if (token !== loadToken) return;
      console.error("WAPR localization scene:", err);
      setPanelStatus("demo-photo-status", "Scene data could not load.", "场景数据加载失败。", true);
      setPanelStatus("demo-cloud-status", "Scene data could not load.", "场景数据加载失败。", true);
    });
  }

  function buildScenePager() {
    var host = $("scene-pager");
    if (!host) return;
    host.innerHTML = "";
    var scenes = datasets[datasetIndex].scenes;
    for (var i = 0; i < scenes.length; i++) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = scenes[i].label;
      button.addEventListener("click", (function (index) {
        return function () {
          sceneIndex = index;
          loadCurrent();
        };
      })(i));
      host.appendChild(button);
    }
  }

  function buildPager() {
    var host = $("demo-pager");
    host.innerHTML = "";
    for (var i = 0; i < datasets.length; i++) {
      var button = document.createElement("button");
      button.type = "button";
      button.innerHTML = '<span class="en">' + datasets[i].title_en + '</span><span class="zh">' + datasets[i].title_zh + '</span>';
      button.addEventListener("click", (function (index) {
        return function () {
          datasetIndex = index;
          sceneIndex = 0;
          buildScenePager();
          loadCurrent();
        };
      })(i));
      host.appendChild(button);
    }
  }

  function boot() {
    var root = $("demo-pager");
    if (!root) return;
    var poseButtons = document.querySelectorAll("#showcase .pose-switch button");
    for (var i = 0; i < poseButtons.length; i++) {
      poseButtons[i].addEventListener("click", function (event) {
        setPose(event.currentTarget.getAttribute("data-pose"));
      });
    }
    var photo = $("demo-image");
    photo.addEventListener("click", function (event) {
      if (!scene || !layers) return;
      var pick = poseMode === "gt" ? layers.gtPick : layers.predPick;
      if (!pick) return;
      var rect = photo.getBoundingClientRect();
      var x = Math.round((event.clientX - rect.left) * photo.width / rect.width);
      var y = Math.round((event.clientY - rect.top) * photo.height / rect.height);
      if (x < 0 || y < 0 || x >= scene.width || y >= scene.height) return;
      var id = pick[y * scene.width + x];
      if (id) selectObject(id);
    });
    setPose("pred");
    WAPRAssets.fetch("demo/manifest.json").then(function (res) {
      if (!res.ok) throw new Error("Viewer manifest: HTTP " + res.status);
      return res.json();
    }).then(function (manifest) {
      datasets = manifest.datasets || [];
      if (!datasets.length && manifest.samples) {
        datasets = manifest.samples.map(function (sample) {
          return {
            title_en: sample.title_en,
            title_zh: sample.title_zh,
            scenes: [{ label: "000000", dir: sample.dir }]
          };
        });
      }
      if (!datasets.length) throw new Error("Viewer manifest contains no scenes");
      buildPager();
      buildScenePager();
      loadCurrent();
    }).catch(function (err) {
      console.error("WAPR localization manifest:", err);
      setPanelStatus("demo-photo-status", "Viewer data could not load.", "查看器数据加载失败。", true);
      setPanelStatus("demo-cloud-status", "Viewer data could not load.", "查看器数据加载失败。", true);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
