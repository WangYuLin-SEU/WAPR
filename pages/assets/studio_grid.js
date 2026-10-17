// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

/* Shared floor and room for the WebGL viewers.
   各个 WebGL 查看器共用的地面和房间。
   Lines stay thin. The fill is only a little deeper than the old flat gray.
   线条保持细。填充只比原来的浅灰再深一点。 */
(function (global) {
  var CELLS = 12;
  var gridMap = null;
  var backdropMap = null;

  function disposeGroup(group) {
    if (!group) return;
    group.traverse(function (child) {
      if (child.geometry) child.geometry.dispose();
      if (!child.material) return;
      var materials = Array.isArray(child.material) ? child.material : [child.material];
      for (var i = 0; i < materials.length; i++) materials[i].dispose();
    });
  }

  function gridTexture() {
    if (gridMap) return gridMap;
    var size = 1024;
    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#e7e9eb";
    ctx.fillRect(0, 0, size, size);
    var step = size / CELLS;
    ctx.strokeStyle = "#b7b7b7";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (var i = 0; i <= CELLS; i++) {
      var p = Math.round(i * step) + 0.5;
      ctx.moveTo(p, 0);
      ctx.lineTo(p, size);
      ctx.moveTo(0, p);
      ctx.lineTo(size, p);
    }
    ctx.stroke();
    ctx.strokeStyle = "#9c9c9c";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (var major = 0; major <= CELLS; major += 4) {
      var q = Math.round(major * step) + 0.5;
      ctx.moveTo(q, 0);
      ctx.lineTo(q, size);
      ctx.moveTo(0, q);
      ctx.lineTo(size, q);
    }
    ctx.stroke();
    gridMap = new THREE.CanvasTexture(canvas);
    gridMap.colorSpace = THREE.SRGBColorSpace;
    gridMap.anisotropy = 4;
    return gridMap;
  }

  function backdropTexture() {
    if (backdropMap) return backdropMap;
    var canvas = document.createElement("canvas");
    canvas.width = 8;
    canvas.height = 256;
    var ctx = canvas.getContext("2d");
    var fade = ctx.createLinearGradient(0, 0, 0, 256);
    fade.addColorStop(0, "#f7f8f9");
    fade.addColorStop(0.55, "#f1f2f4");
    fade.addColorStop(1, "#e6e8eb");
    ctx.fillStyle = fade;
    ctx.fillRect(0, 0, 8, 256);
    backdropMap = new THREE.CanvasTexture(canvas);
    backdropMap.colorSpace = THREE.SRGBColorSpace;
    backdropMap.magFilter = THREE.LinearFilter;
    return backdropMap;
  }

  function applyBackdrop(scene) {
    scene.background = backdropTexture();
  }

  function gridMaterial() {
    return new THREE.MeshBasicMaterial({
      map: gridTexture(),
      toneMapped: false,
      side: THREE.DoubleSide
    });
  }

  function edgedPlane(width, height) {
    var group = new THREE.Group();
    var face = new THREE.Mesh(new THREE.PlaneGeometry(width, height), gridMaterial());
    var rim = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(width, height)),
      new THREE.LineBasicMaterial({ color: 0xc4c4c4 })
    );
    rim.position.z = 0.001;
    group.add(face);
    group.add(rim);
    return group;
  }

  /* One floor under the cloud. cx, y, cz are the floor center. size is meters.
     点云下面的一块地面。cx、y、cz 是地面中心。size 是米。 */
  function placeFloor(scene, previous, cx, y, cz, size) {
    if (previous) {
      scene.remove(previous);
      disposeGroup(previous);
    }
    applyBackdrop(scene);
    var group = edgedPlane(size, size);
    group.rotation.x = -Math.PI / 2;
    group.position.set(cx, y, cz);
    scene.add(group);
    return group;
  }

  /* Floor plus two walls. The mesh turns; this room stays put.
     地面加两面墙。网格转动，房间不动。
     +X right, +Y up, +Z toward the camera. Walls meet at the far corner.
     +X 向右，+Y 向上，+Z 朝向相机。两面墙交在远处那个角。 */
  function placeRoom(scene, previous, size, floorY, back) {
    if (previous) {
      scene.remove(previous);
      disposeGroup(previous);
    }
    applyBackdrop(scene);
    var group = new THREE.Group();
    var floor = edgedPlane(size, size);
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = floorY;
    var wallZ = edgedPlane(size, size);
    wallZ.position.set(0, floorY + size / 2, back);
    var wallX = edgedPlane(size, size);
    wallX.rotation.y = Math.PI / 2;
    wallX.position.set(back, floorY + size / 2, 0);
    group.add(floor);
    group.add(wallZ);
    group.add(wallX);
    scene.add(group);
    return group;
  }

  global.WAPRStudio = {
    applyBackdrop: applyBackdrop,
    placeFloor: placeFloor,
    placeRoom: placeRoom,
    dispose: disposeGroup,
    clearColor: 0xf1f2f4
  };
})(window);
