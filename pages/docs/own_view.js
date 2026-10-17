/* Custom-data frame: the white wedges. Click a chip, the photo, or a mesh.
   自定义数据这一帧：白色三角块。点下面的结果、照片，或右侧网格。
   cloud.bin is OpenCV camera meters. The 3D view is y-up: (x, -y, -z).
   cloud.bin 是 OpenCV 相机系、米。三维视图 y 朝上：(x, -y, -z)。 */
(function () {
  var BOX = [];
  for (var corner = 0; corner < 8; corner++) {
    for (var bit = 0; bit < 3; bit++) {
      var other = corner ^ (1 << bit);
      if (other > corner) BOX.push([corner, other]);
    }
  }

  var scene = null;
  var rgb = null;
  var view = null;
  var selected = 1;
  var base = "own_block/";

  function $(id) { return document.getElementById(id); }

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

  function project(pose, corners, index, K) {
    var x = corners[index][0];
    var y = corners[index][1];
    var z = corners[index][2];
    var X = pose[0] * x + pose[1] * y + pose[2] * z + pose[3];
    var Y = pose[4] * x + pose[5] * y + pose[6] * z + pose[7];
    var Z = pose[8] * x + pose[9] * y + pose[10] * z + pose[11];
    if (Z <= 1.0e-4) return null;
    return [
      (K[0] * X + K[1] * Y + K[2] * Z) / Z,
      (K[3] * X + K[4] * Y + K[5] * Z) / Z
    ];
  }

  function instanceAt(px, py) {
    var hit = null;
    var area = Infinity;
    for (var i = 0; i < scene.instances.length; i++) {
      var item = scene.instances[i];
      var box = item.bbox_xywh;
      if (px < box[0] || py < box[1] || px > box[0] + box[2] || py > box[1] + box[3]) continue;
      var next = box[2] * box[3];
      if (next < area) {
        area = next;
        hit = item.id;
      }
    }
    return hit;
  }

  function drawPhoto() {
    var canvas = $("own-image");
    if (!canvas || !rgb || !scene) return;
    var width = rgb.naturalWidth;
    var height = rgb.naturalHeight;
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext("2d");
    ctx.drawImage(rgb, 0, 0);
    var K = scene.K;
    for (var pass = 0; pass < 2; pass++) {
      for (var i = 0; i < scene.instances.length; i++) {
        var item = scene.instances[i];
        var on = item.id === selected;
        if ((pass === 0) === on) continue;
        var uv = [];
        for (var c = 0; c < 8; c++) {
          var point = project(item.pose, item.corners, c, K);
          uv.push(point);
        }
        ctx.lineWidth = on ? 3 : 1.5;
        ctx.strokeStyle = on ? "#d12b2b" : "#2f6fbf";
        ctx.beginPath();
        for (var e = 0; e < BOX.length; e++) {
          var a = uv[BOX[e][0]];
          var b = uv[BOX[e][1]];
          if (!a || !b) continue;
          ctx.moveTo(a[0], a[1]);
          ctx.lineTo(b[0], b[1]);
        }
        ctx.stroke();
        var box = item.bbox_xywh;
        ctx.strokeStyle = on ? "#1d4e89" : "rgba(29, 78, 137, 0.55)";
        ctx.lineWidth = on ? 2 : 1;
        ctx.strokeRect(box[0], box[1], box[2], box[3]);
        if (on) {
          ctx.font = "600 16px sans-serif";
          var label = item.id + "  2D " + item.score_2d.toFixed(2) + "  |  6D " + item.score_6d.toFixed(3);
          var tw = ctx.measureText(label).width + 10;
          var lx = Math.max(0, Math.min(box[0], width - tw));
          var ly = Math.max(0, box[1] - 22);
          ctx.fillStyle = "#1c1c1c";
          ctx.fillRect(lx, ly, tw, 20);
          ctx.fillStyle = "#ffffff";
          ctx.textBaseline = "middle";
          ctx.fillText(label, lx + 5, ly + 10);
        }
      }
    }
  }

  function fillChips() {
    var host = $("own-chips");
    if (!host || !scene) return;
    host.innerHTML = "";
    for (var i = 0; i < scene.instances.length; i++) {
      var item = scene.instances[i];
      var button = document.createElement("button");
      button.type = "button";
      button.setAttribute("aria-pressed", item.id === selected ? "true" : "false");
      button.textContent = item.id + "  2D " + item.score_2d.toFixed(2) + "  |  6D " + item.score_6d.toFixed(3);
      button.addEventListener("click", (function (id) {
        return function () { select(id); };
      })(item.id));
      host.appendChild(button);
    }
  }

  function syncMeshes() {
    if (!view) return;
    for (var i = 0; i < view.records.length; i++) {
      var record = view.records[i];
      var on = record.item.id === selected;
      record.mesh.material.color.setHex(0xffffff);
      record.mesh.material.opacity = on ? 1 : 0.22;
      record.mesh.material.transparent = !on;
      record.mesh.material.depthWrite = on;
      record.mesh.material.needsUpdate = true;
      record.mesh.renderOrder = on ? 3 : 1;
      record.shell.visible = on;
    }
  }

  function select(id) {
    selected = id;
    drawPhoto();
    fillChips();
    syncMeshes();
    focusInstance(id);
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
      if (view.records[i].item.id === id) {
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
    var focus = new THREE.Box3().setFromObject(view.lines);
    if (focus.isEmpty()) return;
    var center = focus.getCenter(new THREE.Vector3());
    var radius = Math.max(focus.getSize(new THREE.Vector3()).length(), 0.05);
    if (view.points) view.points.material.size = radius * 0.012;
    var gridSize = Math.max(radius * 2.8, 0.35);
    if (window.WAPRStudio) {
      view.grid = WAPRStudio.placeFloor(view.scene, view.grid, center.x, focus.min.y, center.z, gridSize);
      softenFloor(view.grid);
    }
    // Same optical axis as the photo. OpenCV +Z is this view's -Z.
    // 和照片同一条光轴。OpenCV 的 +Z 在这个视图里是 -Z。
    var depth = Math.max(0.2, -center.z);
    var fy = scene.K[4];
    view.camera.fov = 2 * Math.atan(scene.height / (2 * fy)) * 180 / Math.PI;
    view.orbitId = 0;
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
      size: 0.004,
      vertexColors: true,
      sizeAttenuation: true
    }));
    view.scene.add(view.points);
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
    for (var k = 0; k < scene.instances.length; k++) {
      var item = scene.instances[k];
      var mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2
      }));
      mesh.matrixAutoUpdate = false;
      mesh.matrix.copy(poseMatrix(item.pose));
      mesh.userData.id = item.id;
      mesh.renderOrder = 1;
      view.lines.add(mesh);
      var shell = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
        color: 0xc23b2e,
        side: THREE.BackSide,
        toneMapped: false
      }));
      shell.matrixAutoUpdate = false;
      shell.matrix.multiplyMatrices(poseMatrix(item.pose), new THREE.Matrix4().makeScale(1.04, 1.04, 1.04));
      shell.userData.id = item.id;
      shell.visible = false;
      shell.renderOrder = 2;
      view.lines.add(shell);
      view.records.push({ item: item, mesh: mesh, shell: shell });
    }
    view.lines.updateMatrixWorld(true);
    var pending = view.orbitId;
    syncMeshes();
    frameView();
    if (pending) focusInstance(pending);
  }

  /* The floor is a faint grid behind the cloud. It does not cover the points
     or mix into their color, including when the camera looks up from below.
     地面是点云后面的一层淡网格。它不挡住点，也不混进点的颜色，从下往上看也一样。 */
  function softenFloor(group) {
    group.traverse(function (child) {
      if (!child.material) return;
      var materials = Array.isArray(child.material) ? child.material : [child.material];
      for (var i = 0; i < materials.length; i++) {
        materials[i].transparent = true;
        materials[i].opacity = child.isLineSegments ? 0.62 : 0.28;
        materials[i].depthWrite = false;
        materials[i].depthTest = false;
      }
    });
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
    var canvas = $("own-cloud");
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
    renderer.autoClear = false;
    var scene3 = new THREE.Scene();
    if (window.WAPRStudio) WAPRStudio.applyBackdrop(scene3);
    var camera = new THREE.PerspectiveCamera(45, 1, 0.01, 20);
    camera.up.set(0, 1, 0);
    var lines = new THREE.Group();
    scene3.add(lines);
    var raycaster = new THREE.Raycaster();
    view = {
      renderer: renderer,
      scene: scene3,
      camera: camera,
      lines: lines,
      raycaster: raycaster,
      target: new THREE.Vector3(),
      spherical: new THREE.Spherical(1, 1.05, 0.6),
      records: [],
      drag: null
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
        if (hits[i].object.userData.id) {
          select(hits[i].object.userData.id);
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
    canvas.addEventListener("contextmenu", function (event) { event.preventDefault(); });
    window.addEventListener("resize", resizeView);
    function frame() {
      resizeView();
      renderer.clear(true, true, true);
      var grid = view.grid;
      var background = scene3.background;
      if (grid) {
        if (view.points) view.points.visible = false;
        view.lines.visible = false;
        renderer.render(scene3, camera);
        if (view.points) view.points.visible = true;
        view.lines.visible = true;
        grid.visible = false;
        scene3.background = null;
        renderer.clearDepth();
      }
      renderer.render(scene3, camera);
      scene3.background = background;
      if (grid) grid.visible = true;
    }
    WAPRRender.register(canvas, frame);
  }

  function boot() {
    if (!$("own-cloud") || typeof THREE === "undefined") return;
    var photo = $("own-image");
    photo.addEventListener("click", function (event) {
      if (!scene || !rgb) return;
      var rect = photo.getBoundingClientRect();
      var px = (event.clientX - rect.left) / rect.width * scene.width;
      var py = (event.clientY - rect.top) / rect.height * scene.height;
      var id = instanceAt(px, py);
      if (id) select(id);
    });
    WAPRAssets.fetch(base + "scene.json?v=render-nms1").then(function (res) { return res.json(); }).then(function (data) {
      scene = data;
      selected = scene.instances.length ? scene.instances[0].id : 1;
      fillChips();
      rgb = new Image();
      rgb.onload = drawPhoto;
      // Share identical RGB, depth cloud and mesh assets with the project page.
      // 与宣传页共用相同的 RGB、深度点云及网格文件。
      rgb.src = "../wedge/rgb.jpg";
      ensureView();
      WAPRAssets.fetch("../wedge/cloud.bin").then(function (res) { return res.arrayBuffer(); }).then(showCloud);
      WAPRAssets.fetch("../wedge/mesh.json").then(function (res) { return res.json(); }).then(function (meta) {
        return WAPRAssets.fetch("../wedge/mesh.bin").then(function (res) { return res.arrayBuffer(); }).then(function (buffer) {
          showMesh(meta, buffer);
        });
      });
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
