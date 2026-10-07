// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  var canvas = document.getElementById("language-mesh");
  if (!canvas || typeof THREE === "undefined") return;

  function hideIfMissing(id) {
    var img = document.getElementById(id);
    if (!img) return;
    img.addEventListener("error", function () {
      if (img.parentElement) img.parentElement.hidden = true;
    });
  }
  hideIfMissing("language-flow");
  hideIfMissing("language-overlay");

  var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
  var scene = new THREE.Scene();
  if (window.WAPRStudio) WAPRStudio.applyBackdrop(scene);
  var camera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
  var light = new THREE.DirectionalLight(0xffffff, 1.1);
  light.position.set(0.4, 0.8, 1);
  scene.add(light);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  var rig = new THREE.Group();
  scene.add(rig);
  var drag = null;
  var room = null;

  function disposeObject(obj) {
    obj.traverse(function (child) {
      if (child.geometry) child.geometry.dispose();
      var material = child.material;
      if (!material) return;
      if (material.length) {
        for (var i = 0; i < material.length; i++) material[i].dispose();
      } else {
        material.dispose();
      }
    });
  }

  // Floor plus two walls. The room stays still while the mesh turns.
  // 地面加两面墙。网格转动时，这个房间不动。
  function placeRoom(previous, span) {
    if (previous) {
      scene.remove(previous);
      disposeObject(previous);
    }
    var size = Math.max(span * 3.6, 0.45);
    var floorY = -span * 0.62;
    var back = -size * 0.5;
    if (window.WAPRStudio) return WAPRStudio.placeRoom(scene, null, size, floorY, back);
    return null;
  }
  var rulerM = 0.1;

  function resize() {
    var width = canvas.clientWidth || 240;
    var height = canvas.clientHeight || 180;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }

  function frameAll(lists) {
    var span = 0;
    for (var i = 0; i < lists.length; i++) {
      span = Math.max(span, lists[i][0], lists[i][1], lists[i][2]);
    }
    var radius = span * 0.72;
    room = placeRoom(room, span);
    camera.position.set(radius * 1.7, radius * 0.85, radius * 1.9);
    camera.lookAt(0, 0, 0);
    camera.near = Math.max(radius / 100, 1e-4);
    camera.far = radius * 40;
    camera.updateProjectionMatrix();
  }

  function placeRuler() {
    var el = document.getElementById("language-ruler");
    if (!el) return;
    var dist = camera.position.length();
    var worldH = 2 * Math.tan((camera.fov * Math.PI / 180) / 2) * dist;
    var px = rulerM / worldH * (canvas.clientHeight || 180);
    el.style.width = Math.max(px, 24) + "px";
  }

  function draw() {
    resize();
    placeRuler();
    renderer.render(scene, camera);
  }

  function wireBox(extents, color) {
    var hx = extents[0] / 2 + 0.001;
    var hy = extents[1] / 2 + 0.001;
    var hz = extents[2] / 2 + 0.001;
    var p = [
      [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
      [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz]
    ];
    var edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    var arr = new Float32Array(edges.length * 6);
    for (var i = 0; i < edges.length; i++) {
      arr.set(p[edges[i][0]], i * 6);
      arr.set(p[edges[i][1]], i * 6 + 3);
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: color }));
  }

  function boxForMesh(mesh, color) {
    mesh.geometry.computeBoundingBox();
    var bounds = mesh.geometry.boundingBox;
    var ext = [
      bounds.max.x - bounds.min.x,
      bounds.max.y - bounds.min.y,
      bounds.max.z - bounds.min.z
    ];
    var line = wireBox(ext, color);
    line.position.set(
      0.5 * (bounds.min.x + bounds.max.x),
      0.5 * (bounds.min.y + bounds.max.y),
      0.5 * (bounds.min.z + bounds.max.z)
    );
    return { line: line, ext: ext };
  }

  function meshFromBin(meta, buffer, material) {
    var n = meta.vertices;
    var f = meta.faces;
    var positions = new Float32Array(buffer, 0, n * 3);
    var indices = new Uint32Array(buffer, n * 12, f * 3);
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setIndex(new THREE.BufferAttribute(indices, 1));
    if (meta.mode === "uv") {
      var uv = new Float32Array(buffer, n * 12 + f * 12, n * 2);
      geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
    } else {
      var rgb = new Uint8Array(buffer, n * 12 + f * 12, n * 3);
      var colors = new Float32Array(n * 3);
      for (var c = 0; c < n * 3; c++) colors[c] = rgb[c] / 255;
      geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    }
    geo.computeVertexNormals();
    return new THREE.Mesh(geo, material);
  }

  function mm1(m) {
    return (Math.round(m * 10000) / 10).toFixed(1);
  }

  function triple(list) {
    return mm1(list[0]) + " × " + mm1(list[1]) + " × " + mm1(list[2]);
  }

  function loadJson(url) {
    return WAPRAssets.fetch(url).then(function (res) { return res.json(); });
  }

  function loadBin(url) {
    return WAPRAssets.fetch(url).then(function (res) { return res.arrayBuffer(); });
  }

  function loadTexture(url) {
    return new Promise(function (resolve, reject) {
      new THREE.TextureLoader().load(url, function (tex) {
        tex.generateMipmaps = false;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        resolve(tex);
      }, undefined, reject);
    });
  }

  canvas.addEventListener("pointerdown", function (event) {
    drag = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", function (event) {
    if (!drag) return;
    rig.rotation.y += (event.clientX - drag.x) * 0.01;
    rig.rotation.x += (event.clientY - drag.y) * 0.01;
    drag.x = event.clientX;
    drag.y = event.clientY;
    draw();
  });
  canvas.addEventListener("pointerup", function () { drag = null; });
  canvas.addEventListener("wheel", function (event) {
    event.preventDefault();
    camera.position.multiplyScalar(event.deltaY > 0 ? 1.08 : 0.92);
    draw();
  }, { passive: false });
  window.addEventListener("resize", draw);

  var query = "?v=43";
  function materialFor(meta, tex) {
    if (meta.mode === "uv" && tex) return new THREE.MeshLambertMaterial({ map: tex });
    return new THREE.MeshLambertMaterial({ vertexColors: true });
  }
  Promise.all([
    loadJson("demo/reconstruct/mustard.json" + query),
    loadBin("demo/reconstruct/mustard.bin" + query),
    loadJson("demo/reconstruct/mustard_before.json" + query),
    loadBin("demo/reconstruct/mustard_before.bin" + query),
    loadJson("demo/reconstruct/mustard_cad.json" + query),
    loadBin("demo/reconstruct/mustard_cad.bin" + query)
  ]).then(function (parts) {
    function texOf(meta) {
      if (meta.mode === "uv") return loadTexture("demo/reconstruct/" + meta.texture + query);
      return Promise.resolve(null);
    }
    return Promise.all([texOf(parts[0]), texOf(parts[2]), texOf(parts[4])]).then(function (tex) {
      return [parts, tex];
    });
  }).then(function (loaded) {
    var parts = loaded[0];
    var pred = meshFromBin(parts[0], parts[1], materialFor(parts[0], loaded[1][0]));
    var before = meshFromBin(parts[2], parts[3], materialFor(parts[2], loaded[1][1]));
    var cad = meshFromBin(parts[4], parts[5], materialFor(parts[4], loaded[1][2]));
    var predBox = boxForMesh(pred, 0xc62828);
    var beforeBox = boxForMesh(before, 0xc47a12);
    var cadBox = boxForMesh(cad, 0x17833d);
    before.visible = false;
    cad.visible = false;
    rig.add(pred);
    rig.add(before);
    rig.add(cad);
    rig.add(predBox.line);
    rig.add(beforeBox.line);
    rig.add(cadBox.line);
    var note = document.getElementById("language-size");
    if (note) {
      note.innerHTML =
        '<span class="en"><i class="recon-key before"></i>Before ' + triple(beforeBox.ext) +
        ' mm. <i class="recon-key pred"></i>After ' + triple(predBox.ext) +
        ' mm. <i class="recon-key gt"></i>Real ' + triple(cadBox.ext) +
        ' mm. The bar is 10 cm.</span>' +
        '<span class="zh"><i class="recon-key before"></i>对齐前 ' + triple(beforeBox.ext) +
        ' mm。<i class="recon-key pred"></i>对齐后 ' + triple(predBox.ext) +
        ' mm。<i class="recon-key gt"></i>真值 ' + triple(cadBox.ext) +
        ' mm。标尺 10 cm。</span>';
    }
    frameAll([predBox.ext, beforeBox.ext, cadBox.ext]);
    var buttons = document.querySelectorAll("#language-mesh-mode button");
    function showMesh(which) {
      pred.visible = which === "pred";
      before.visible = which === "before";
      cad.visible = which === "gt";
      for (var i = 0; i < buttons.length; i++) {
        buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-mesh") === which ? "true" : "false");
      }
      draw();
    }
    for (var b = 0; b < buttons.length; b++) {
      buttons[b].addEventListener("click", function (event) {
        showMesh(event.currentTarget.getAttribute("data-mesh"));
      });
    }
    showMesh("pred");
  }).catch(function () {
    var figure = canvas.parentElement ? canvas.parentElement.parentElement : null;
    if (figure) figure.hidden = true;
  });
})();
