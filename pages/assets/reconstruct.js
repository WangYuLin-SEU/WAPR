// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  var canvas = document.getElementById("recon-mesh");
  if (typeof THREE === "undefined") return;
  // The Docs gallery reuses the saved viewers without mounting the home-page case.
  // 文档展示复用保存网格的查看器，无需挂载宣传页的饼干盒案例。
  var gallery = document.querySelector("[data-reconstruction-gallery]");
  if (!canvas && !gallery) return;
  var dataRoot = gallery ? gallery.getAttribute("data-reconstruct-root") : "demo/reconstruct/";

  function hideIfMissing(id) {
    var img = document.getElementById(id);
    if (!img) return;
    img.addEventListener("error", function () {
      if (img.parentElement) img.parentElement.hidden = true;
    });
  }
  hideIfMissing("recon-prompt");
  var poseImages = document.querySelectorAll(".recon-poses img");
  for (var i = 0; i < poseImages.length; i++) hideIfMissing(poseImages[i].id);

  if (canvas) {
    var renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
    var scene = new THREE.Scene();
    if (window.WAPRStudio) WAPRStudio.applyBackdrop(scene);
    var camera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
    camera.position.set(0.35, 0.22, 0.45);
    var light = new THREE.DirectionalLight(0xffffff, 1.1);
    light.position.set(0.4, 0.8, 1);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var rig = new THREE.Group();
    scene.add(rig);
    var drag = null;
    var room = null;
  }

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
  // +X right, +Y up, +Z toward the camera. Walls sit on the far corner.
  // +X 向右，+Y 向上，+Z 朝向相机。墙在远处那个角上。
  function placeRoom(targetScene, previous, span) {
    if (previous) {
      targetScene.remove(previous);
      disposeObject(previous);
    }
    var size = Math.max(span * 3.6, 0.45);
    var floorY = -span * 0.62;
    var back = -size * 0.5;
    if (window.WAPRStudio) return WAPRStudio.placeRoom(targetScene, null, size, floorY, back);
    return null;
  }

  function resize() {
    var width = canvas.clientWidth || 240;
    var height = canvas.clientHeight || 180;
    renderer.setSize(width, height, false);
    camera.aspect = width / Math.max(height, 1);
    camera.updateProjectionMatrix();
  }

  function framePair(predM, cadM, beforeM) {
    var lists = [predM, cadM];
    if (beforeM) lists.push(beforeM);
    var span = 0;
    for (var i = 0; i < lists.length; i++) {
      span = Math.max(span, lists[i][0], lists[i][1], lists[i][2]);
    }
    var radius = span * 0.72;
    room = placeRoom(scene, room, span);
    camera.position.set(radius * 1.7, radius * 0.85, radius * 1.9);
    camera.lookAt(0, 0, 0);
    camera.near = Math.max(radius / 100, 1e-4);
    camera.far = radius * 40;
    camera.updateProjectionMatrix();
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

  function mm1(m) {
    return (Math.round(m * 10000) / 10).toFixed(1);
  }

  function triple(list) {
    return mm1(list[0]) + " × " + mm1(list[1]) + " × " + mm1(list[2]);
  }

  function deltaText(predM, cadM) {
    var parts = [];
    for (var i = 0; i < 3; i++) {
      var v = (predM[i] - cadM[i]) * 1000;
      var s = (Math.round(v * 10) / 10).toFixed(1);
      parts.push((v > 0 ? "+" : "") + s);
    }
    return parts.join(" × ");
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

  var rulerM = 0;

  function placeRuler() {
    var el = document.getElementById("recon-ruler");
    if (!el || !rulerM) return;
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

  if (canvas) {
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
      var scale = event.deltaY > 0 ? 1.08 : 0.92;
      camera.position.multiplyScalar(scale);
      draw();
    }, { passive: false });
  }

  function loadJson(url) {
    return WAPRAssets.fetch(url).then(function (res) {
      if (!res.ok) throw new Error(url);
      return res.json();
    });
  }

  function loadBin(url) {
    return WAPRAssets.fetch(url).then(function (res) {
      if (!res.ok) throw new Error(url);
      return res.arrayBuffer();
    });
  }

  function loadTexture(url) {
    return new Promise(function (resolve, reject) {
      new THREE.TextureLoader().load(url, function (tex) {
        if (THREE.SRGBColorSpace) tex.colorSpace = THREE.SRGBColorSpace;
        tex.generateMipmaps = false;
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        tex.anisotropy = 1;
        resolve(tex);
      }, undefined, reject);
    });
  }

  // Show only the selected box's saved measurements, inside its 3D viewport.
  // 在 3D 视图内只显示所选包围盒的原有测量值，不改动几何或误差计算。
  function writeSizes(predM, cadM, noteId, beforeM, which) {
    var note = document.getElementById(noteId || "recon-size");
    if (!note || !cadM) return;
    var sides = which === "gt" ? cadM : which === "before" ? beforeM : predM;
    var titleEn = which === "gt" ? "Ground truth" : which === "before" ? "Before" : "After";
    var titleZh = which === "gt" ? "真值" : which === "before" ? "对齐前" : "对齐后";
    var color = which === "gt" ? "gt" : which === "before" ? "before" : "pred";
    var gap = which === "gt" ? null : deltaText(sides, cadM);
    note.innerHTML =
      '<span class="en"><strong><i class="recon-key ' + color + '"></i>' + titleEn +
      ' · box sides</strong><span class="recon-size-value">' + triple(sides) + ' mm</span>' +
      (gap === null ? '' : '<small>Difference from CAD: ' + gap + ' mm</small>') + '</span>' +
      '<span class="zh"><strong><i class="recon-key ' + color + '"></i>' + titleZh +
      ' · 包围盒边长</strong><span class="recon-size-value">' + triple(sides) + ' mm</span>' +
      (gap === null ? '' : '<small>与参考 CAD 相差 ' + gap + ' mm</small>') + '</span>';
  }

  if (canvas) {
    Promise.all([
      loadJson(dataRoot + "compare.json?v=20"),
      loadJson(dataRoot + "aligned.json?v=44"),
      loadBin(dataRoot + "aligned.bin?v=44")
    ]).then(function (parts) {
      var compare = parts[0];
      var predMeta = parts[1];
      var predMaterial = new THREE.MeshLambertMaterial({ vertexColors: true });
      if (predMeta.mode === "uv") {
        return loadTexture(dataRoot + predMeta.texture + "?v=44").then(function (tex) {
          return [compare, predMeta, parts[2], new THREE.MeshLambertMaterial({ map: tex })];
        });
      }
      return [compare, predMeta, parts[2], predMaterial];
    }).then(function (loaded) {
      var compare = loaded[0];
      var pred = meshFromBin(loaded[1], loaded[2], loaded[3]);
      var predBox = boxForMesh(pred, 0xc62828);
      var cad = null;
      var before = null;
      var cadBox = null;
      var beforeBox = null;
      var want = "pred";
      var afterMap = loaded[3].map || null;
      rig.add(pred);
      rig.add(predBox.line);
      rulerM = compare.ruler_m;
      var buttons = document.querySelectorAll("#recon-mesh-mode button");
      function showMesh(which) {
        want = which;
        if (which === "gt" && !cad) return;
        if (which === "before" && !before) return;
        pred.visible = which === "pred";
        if (before) before.visible = which === "before";
        if (cad) cad.visible = which === "gt";
        predBox.line.visible = which === "pred";
        if (beforeBox) beforeBox.line.visible = which === "before";
        if (cadBox) {
          cadBox.line.visible = which === "gt";
          writeSizes(predBox.ext, cadBox.ext, "recon-size", beforeBox.ext, which);
        }
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
      framePair(predBox.ext, predBox.ext);
      draw();
      Promise.all([
        loadJson(dataRoot + "cad.json?v=23"),
        loadBin(dataRoot + "cad.bin?v=23"),
        loadTexture(dataRoot + "cad.jpg?v=23"),
        loadJson(dataRoot + "before.json?v=44"),
        loadBin(dataRoot + "before.bin?v=44")
      ]).then(function (rest) {
        cad = meshFromBin(rest[0], rest[1], new THREE.MeshLambertMaterial({ map: rest[2] }));
        cad.visible = want === "gt";
        cadBox = boxForMesh(cad, 0x17833d);
        var beforeMeta = rest[3];
        var beforeMat = beforeMeta.mode === "uv"
          ? new THREE.MeshLambertMaterial({ map: afterMap })
          : new THREE.MeshLambertMaterial({ vertexColors: true });
        before = meshFromBin(beforeMeta, rest[4], beforeMat);
        before.visible = want === "before";
        beforeBox = boxForMesh(before, 0xc47a12);
        rig.add(cad);
        rig.add(before);
        rig.add(cadBox.line);
        rig.add(beforeBox.line);
        framePair(predBox.ext, cadBox.ext, beforeBox.ext);
        showMesh(want);
      });
    }).catch(function () {
      canvas.parentElement.hidden = true;
    });

    window.addEventListener("resize", draw);
  }

  function mountExtra(canvas, stem) {
    if (!canvas) return;
    var extraRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    extraRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    extraRenderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
    var extraScene = new THREE.Scene();
    if (window.WAPRStudio) WAPRStudio.applyBackdrop(extraScene);
    var extraCamera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
    var extraLight = new THREE.DirectionalLight(0xffffff, 1.1);
    extraLight.position.set(0.4, 0.8, 1);
    extraScene.add(extraLight);
    extraScene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var extraRig = new THREE.Group();
    extraScene.add(extraRig);
    var extraDrag = null;
    var extraRoom = null;
    var rulerM = 0.1;

    function extraResize() {
      var width = canvas.clientWidth || 240;
      var height = canvas.clientHeight || 180;
      extraRenderer.setSize(width, height, false);
      extraCamera.aspect = width / Math.max(height, 1);
      extraCamera.updateProjectionMatrix();
    }

    function extraFrame(extents) {
      var span = Math.max(extents[0], extents[1], extents[2]);
      var radius = span * 0.72;
      extraCamera.position.set(radius * 1.7, radius * 0.85, radius * 1.9);
      extraCamera.lookAt(0, 0, 0);
      extraCamera.near = Math.max(radius / 100, 1e-4);
      extraCamera.far = radius * 40;
      extraCamera.updateProjectionMatrix();
    }

    function placeExtraRuler() {
      var el = document.getElementById(canvas.id + "-ruler");
      if (!el) return;
      var dist = extraCamera.position.length();
      var worldH = 2 * Math.tan((extraCamera.fov * Math.PI / 180) / 2) * dist;
      var px = rulerM / worldH * (canvas.clientHeight || 180);
      el.style.width = Math.max(px, 24) + "px";
    }

    function extraDraw() {
      extraResize();
      placeExtraRuler();
      extraRenderer.render(extraScene, extraCamera);
    }

    canvas.addEventListener("pointerdown", function (event) {
      extraDrag = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!extraDrag) return;
      extraRig.rotation.y += (event.clientX - extraDrag.x) * 0.01;
      extraRig.rotation.x += (event.clientY - extraDrag.y) * 0.01;
      extraDrag.x = event.clientX;
      extraDrag.y = event.clientY;
      extraDraw();
    });
    canvas.addEventListener("pointerup", function () { extraDrag = null; });
    canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      var scale = event.deltaY > 0 ? 1.08 : 0.92;
      extraCamera.position.multiplyScalar(scale);
      extraDraw();
    }, { passive: false });
    window.addEventListener("resize", extraDraw);

    var query = "?v=43";
    function materialFor(meta, tex) {
      if (meta.mode === "uv" && tex) return new THREE.MeshLambertMaterial({ map: tex });
      return new THREE.MeshLambertMaterial({ vertexColors: true });
    }
    function extraFrameAll(lists) {
      var span = 0;
      for (var i = 0; i < lists.length; i++) {
        span = Math.max(span, lists[i][0], lists[i][1], lists[i][2]);
      }
      var radius = span * 0.72;
      extraRoom = placeRoom(extraScene, extraRoom, span);
      extraCamera.position.set(radius * 1.7, radius * 0.85, radius * 1.9);
      extraCamera.lookAt(0, 0, 0);
      extraCamera.near = Math.max(radius / 100, 1e-4);
      extraCamera.far = radius * 40;
      extraCamera.updateProjectionMatrix();
    }
    Promise.all([
      loadJson(dataRoot + stem + ".json" + query),
      loadBin(dataRoot + stem + ".bin" + query),
      loadJson(dataRoot + stem + "_before.json" + query),
      loadBin(dataRoot + stem + "_before.bin" + query),
      loadJson(dataRoot + stem + "_cad.json" + query),
      loadBin(dataRoot + stem + "_cad.bin" + query)
    ]).then(function (parts) {
      var predMeta = parts[0];
      var beforeMeta = parts[2];
      var cadMeta = parts[4];
      function texOf(meta) {
        if (meta.mode === "uv") return loadTexture(dataRoot + meta.texture + query);
        return Promise.resolve(null);
      }
      return Promise.all([texOf(predMeta), texOf(beforeMeta), texOf(cadMeta)]).then(function (tex) {
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
      var want = "pred";
      before.visible = false;
      cad.visible = false;
      extraRig.add(pred);
      extraRig.add(before);
      extraRig.add(cad);
      extraRig.add(predBox.line);
      extraRig.add(beforeBox.line);
      extraRig.add(cadBox.line);
      extraFrameAll([predBox.ext, beforeBox.ext, cadBox.ext]);
      var buttons = document.querySelectorAll("#" + canvas.id + "-mode button");
      function showMesh(which) {
        want = which;
        pred.visible = which === "pred";
        before.visible = which === "before";
        cad.visible = which === "gt";
        predBox.line.visible = which === "pred";
        beforeBox.line.visible = which === "before";
        cadBox.line.visible = which === "gt";
        writeSizes(predBox.ext, cadBox.ext, canvas.id + "-size", beforeBox.ext, which);
        for (var i = 0; i < buttons.length; i++) {
          buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-mesh") === which ? "true" : "false");
        }
        extraDraw();
      }
      for (var b = 0; b < buttons.length; b++) {
        buttons[b].addEventListener("click", function (event) {
          showMesh(event.currentTarget.getAttribute("data-mesh"));
        });
      }
      showMesh(want);
    }).catch(function () {
      var figure = canvas.parentElement ? canvas.parentElement.parentElement : null;
      if (figure) figure.hidden = true;
    });
  }

  mountExtra(document.getElementById("recon-sugar"), "sugar");
  mountExtra(document.getElementById("recon-mustard"), "mustard");

  function mountTrack(canvas) {
    if (!canvas) return;
    var trackRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    trackRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    trackRenderer.setClearColor(window.WAPRStudio ? WAPRStudio.clearColor : 0xc5d3e0, 1);
    var trackScene = new THREE.Scene();
    if (window.WAPRStudio) WAPRStudio.applyBackdrop(trackScene);
    var trackCamera = new THREE.PerspectiveCamera(35, 1, 0.001, 50);
    var trackLight = new THREE.DirectionalLight(0xffffff, 1.1);
    trackLight.position.set(0.4, 0.8, 1);
    trackScene.add(trackLight);
    trackScene.add(new THREE.AmbientLight(0xffffff, 0.55));
    var trackRig = new THREE.Group();
    trackScene.add(trackRig);
    var trackDrag = null;
    var trackRoom = null;
    var rulerM = 0.1;

    function trackResize() {
      var width = canvas.clientWidth || 240;
      var height = canvas.clientHeight || 180;
      trackRenderer.setSize(width, height, false);
      trackCamera.aspect = width / Math.max(height, 1);
      trackCamera.updateProjectionMatrix();
    }

    function trackDraw() {
      trackResize();
      var el = document.getElementById("recon-track-ruler");
      if (el) {
        var dist = trackCamera.position.length();
        var worldH = 2 * Math.tan((trackCamera.fov * Math.PI / 180) / 2) * dist;
        var px = rulerM / worldH * (canvas.clientHeight || 180);
        el.style.width = Math.max(px, 24) + "px";
      }
      trackRenderer.render(trackScene, trackCamera);
    }

    canvas.addEventListener("pointerdown", function (event) {
      trackDrag = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", function (event) {
      if (!trackDrag) return;
      trackRig.rotation.y += (event.clientX - trackDrag.x) * 0.01;
      trackRig.rotation.x += (event.clientY - trackDrag.y) * 0.01;
      trackDrag.x = event.clientX;
      trackDrag.y = event.clientY;
      trackDraw();
    });
    canvas.addEventListener("pointerup", function () { trackDrag = null; });
    canvas.addEventListener("wheel", function (event) {
      event.preventDefault();
      trackCamera.position.multiplyScalar(event.deltaY > 0 ? 1.08 : 0.92);
      trackDraw();
    }, { passive: false });
    window.addEventListener("resize", trackDraw);

    function signedMm(deltaM) {
      var mm = Math.round(deltaM * 10000) / 10;
      return (mm > 0 ? "+" : "") + mm.toFixed(1);
    }

    function axisLine(reconExt, trackExt) {
      var names = ["X", "Y", "Z"];
      var en = [];
      var zh = [];
      for (var i = 0; i < 3; i++) {
        var delta = trackExt[i] - reconExt[i];
        if (Math.abs(delta) < 3.0e-4) {
          en.push(names[i] + " unchanged");
          zh.push(names[i] + " 未改");
        } else {
          en.push(names[i] + " " + signedMm(delta) + " mm");
          zh.push(names[i] + " " + signedMm(delta) + " mm");
        }
      }
      return [en.join(", "), zh.join("，")];
    }

    Promise.all([
      loadJson(dataRoot + "aligned.json?v=44"),
      loadBin(dataRoot + "aligned.bin?v=44"),
      loadTexture(dataRoot + "aligned.jpg?v=44"),
      loadJson(dataRoot + "track.json?v=7"),
      loadBin(dataRoot + "track.bin?v=7"),
      loadTexture(dataRoot + "track.jpg?v=7"),
      loadJson(dataRoot + "cad.json?v=23"),
      loadBin(dataRoot + "cad.bin?v=23"),
      loadTexture(dataRoot + "cad.jpg?v=23")
    ]).then(function (parts) {
      var recon = meshFromBin(parts[0], parts[1], new THREE.MeshLambertMaterial({ map: parts[2] }));
      var tracked = meshFromBin(parts[3], parts[4], new THREE.MeshLambertMaterial({ map: parts[5] }));
      var cad = meshFromBin(parts[6], parts[7], new THREE.MeshLambertMaterial({ map: parts[8] }));
      var reconBox = boxForMesh(recon, 0xc47a12);
      var trackBox = boxForMesh(tracked, 0xc62828);
      var cadBox = boxForMesh(cad, 0x17833d);
      var meshes = { recon: recon, track: tracked, gt: cad };
      var lines = { recon: reconBox.line, track: trackBox.line, gt: cadBox.line };
      trackRig.add(recon);
      trackRig.add(tracked);
      trackRig.add(cad);
      trackRig.add(reconBox.line);
      trackRig.add(trackBox.line);
      trackRig.add(cadBox.line);
      var span = 0;
      var extentLists = [reconBox.ext, trackBox.ext, cadBox.ext];
      for (var e = 0; e < extentLists.length; e++) {
        span = Math.max(span, extentLists[e][0], extentLists[e][1], extentLists[e][2]);
      }
      trackRoom = placeRoom(trackScene, trackRoom, span);
      var radius = span * 0.72;
      trackCamera.position.set(radius * 1.7, radius * 0.85, radius * 1.9);
      trackCamera.lookAt(0, 0, 0);
      trackCamera.near = Math.max(radius / 100, 1e-4);
      trackCamera.far = radius * 40;
      trackCamera.updateProjectionMatrix();
      var note = document.getElementById("recon-track-size");
      var changed = axisLine(reconBox.ext, trackBox.ext);
      if (note) {
        note.innerHTML =
          '<span class="en"><i class="recon-key before"></i>Reconstruction ' + triple(reconBox.ext) +
          ' mm. <i class="recon-key pred"></i>After tracking ' + triple(trackBox.ext) +
          ' mm (' + changed[0] + '). <i class="recon-key gt"></i>Real ' + triple(cadBox.ext) +
          ' mm. The bar is 10 cm.</span>' +
          '<span class="zh"><i class="recon-key before"></i>重建 ' + triple(reconBox.ext) +
          ' mm。<i class="recon-key pred"></i>跟踪之后 ' + triple(trackBox.ext) +
          ' mm（' + changed[1] + '）。<i class="recon-key gt"></i>真值 ' + triple(cadBox.ext) +
          ' mm。标尺 10 cm。</span>';
      }
      var buttons = document.querySelectorAll("#recon-track-mode button");
      function showMesh(which) {
        var key;
        for (key in meshes) {
          meshes[key].visible = key === which;
          lines[key].visible = key === which;
        }
        for (var i = 0; i < buttons.length; i++) {
          buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-mesh") === which ? "true" : "false");
        }
        trackDraw();
      }
      for (var b = 0; b < buttons.length; b++) {
        buttons[b].addEventListener("click", function (event) {
          showMesh(event.currentTarget.getAttribute("data-mesh"));
        });
      }
      showMesh("track");
    }).catch(function () {
      var figure = canvas.parentElement ? canvas.parentElement.parentElement : null;
      if (figure) figure.hidden = true;
    });
  }

  mountTrack(document.getElementById("recon-track"));
})();
