// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  const photo = document.getElementById("photo")
  const world = document.getElementById("world")
  const photoCtx = photo.getContext("2d")
  const tabs = document.getElementById("tabs")
  const list = document.getElementById("list")
  const summary = document.getElementById("summary")
  const detail = document.getElementById("detail")
  const speeds = document.getElementById("speeds")
  const projCheck = document.getElementById("proj-check")

  const COLORS = { hit: "#245db3", wrong: "#f76707", extra: "#8b49a7", miss: "#868e96" }
  const NAMES = { hit: "正确", wrong: "没对上", miss: "没检出", extra: "额外" }

  let data = null
  let image = null
  let methodIndex = 0
  let selected = null
  let renderer = null
  let scene = null
  let camera = null
  let pivot = null
  let centroid = null
  let geometry = null
  let wire = null
  let meshes = []
  let drag = null

  function method() {
    return data.methods[methodIndex]
  }

  function contain(cw, ch, iw, ih) {
    const s = Math.min(cw / iw, ch / ih)
    const dw = iw * s
    const dh = ih * s
    return { s: s, ox: (cw - dw) / 2, oy: (ch - dh) / 2, dw: dw, dh: dh }
  }

  function poseMatrix(pose) {
    const m = new THREE.Matrix4()
    m.set(
      pose[0], pose[1], pose[2], pose[3],
      -pose[4], -pose[5], -pose[6], -pose[7],
      -pose[8], -pose[9], -pose[10], -pose[11],
      0, 0, 0, 1
    )
    return m
  }

  function localMatrix(pose) {
    const m = poseMatrix(pose)
    m.premultiply(new THREE.Matrix4().makeTranslation(-centroid.x, -centroid.y, -centroid.z))
    return m
  }

  function errorTag(hyp) {
    if (!hyp || (hyp.kind !== "wrong" && hyp.kind !== "extra")) return ""
    const score = hyp.score_6d
    if (score === null || score === undefined) return "错误"
    const high = score >= (data.score_6d_high_min || 0.5)
    return (high ? "6D 分数较高的错误 " : "6D 分数较低的错误 ") + Number(score).toFixed(2)
  }

  function errorSuffix(hyp) {
    const tag = errorTag(hyp)
    return tag ? " · " + tag : ""
  }

  function mmText(value) {
    return (value < 10 ? value.toFixed(2) : value.toFixed(1)) + " mm"
  }

  function kindColor(kind) {
    return COLORS[kind]
  }

  function counts(item) {
    const hit = item.parts.filter(function (row) { return row.kind === "hit" }).length
    const wrong = item.parts.filter(function (row) { return row.kind === "wrong" }).length
    const miss = item.parts.filter(function (row) { return row.kind === "miss" }).length
    const extra = item.hypotheses.filter(function (row) { return row.kind === "extra" }).length
    return { hit: hit, wrong: wrong, miss: miss, extra: extra, n: item.parts.length, hyps: item.hypotheses.length }
  }

  function buildSpeeds() {
    speeds.innerHTML = ""
    data.methods.forEach(function (item) {
      const card = document.createElement("div")
      card.className = "speed"
      const title = document.createElement("b")
      title.textContent = item.name
      const text = document.createElement("p")
      text.textContent = item.speed
      card.appendChild(title)
      card.appendChild(text)
      speeds.appendChild(card)
    })
  }

  function buildTabs() {
    tabs.innerHTML = ""
    data.methods.forEach(function (item, index) {
      const c = counts(item)
      const button = document.createElement("button")
      button.className = "tab" + (index === methodIndex ? " on" : "")
      button.textContent = item.name + "  正确 " + c.hit + "/" + c.n
      button.addEventListener("click", function () {
        const keep = selected && selected.gt
        methodIndex = index
        if (keep) selectPart(keep)
        else selected = null
        renderAll()
      })
      tabs.appendChild(button)
    })
  }

  function buildSummary() {
    const c = counts(method())
    summary.innerHTML = ""
    const line = document.createElement("div")
    line.textContent = "正确 " + c.hit + "/" + c.n
      + " · 没对上 " + c.wrong
      + " · 没检出 " + c.miss
      + " · 候选姿态 " + c.hyps + " 条"
      + " · 额外 " + c.extra + " 条"
    const note = document.createElement("div")
    note.className = "note"
    const high = data.score_6d_high_min
    note.textContent = "正确是模型点的平均距离不超过直径 76.2 mm 的 10%，也就是 "
      + data.threshold_mm + " mm。错误姿态标在轮廓旁边。WAPR 按 score_6d 分组，不低于 "
      + high + " 为高分组，低于 " + high + " 为低分组。这个阈值仅用于当前视图的展示，不表示正确概率。"
      + " Line2D、AAE、PPF 没有这个分数，错误姿态只标错误。"
    summary.appendChild(line)
    summary.appendChild(note)
  }

  function addRow(parent, label, sub, value, kind, low, onClick) {
    const button = document.createElement("button")
    button.className = "row"
    const a = document.createElement("span")
    a.textContent = label
    const b = document.createElement("span")
    b.textContent = sub
    const c = document.createElement("span")
    c.className = kind
    const dot = document.createElement("span")
    dot.className = "dot"
    dot.style.background = kindColor(kind)
    c.appendChild(dot)
    c.appendChild(document.createTextNode(value))
    const d = document.createElement("span")
    d.className = "low"
    d.textContent = low
    button.appendChild(a)
    button.appendChild(b)
    button.appendChild(c)
    button.appendChild(d)
    button.addEventListener("click", onClick)
    parent.appendChild(button)
    return button
  }

  function buildList() {
    list.innerHTML = ""
    const item = method()
    const head = document.createElement("div")
    head.className = "group"
    head.textContent = "11 个零件"
    list.appendChild(head)
    item.parts.forEach(function (part) {
      const gt = data.gt[part.id - 1]
      const low = gt.vis <= 0.6 ? "论文不计" : ""
      const linked = part.hyp === null || part.hyp === undefined ? null : item.hypotheses[part.hyp]
      const value = part.kind === "miss" ? "没检出" : mmText(part.add_mm) + errorSuffix(linked)
      const button = addRow(list, "零件 " + part.id, "可见 " + gt.vis.toFixed(2), value, part.kind, low, function () {
        selectPart(part.id)
        renderAll()
      })
      if (selected && selected.gt === part.id) button.classList.add("on")
    })
    const extras = item.hypotheses.filter(function (row) { return row.kind === "extra" })
    if (!extras.length) return
    const extraHead = document.createElement("div")
    extraHead.className = "group"
    extraHead.textContent = "额外候选姿态 " + extras.length + " 条"
    list.appendChild(extraHead)
    item.hypotheses.forEach(function (hyp, index) {
      if (hyp.kind !== "extra") return
      const button = addRow(list, "额外", "", "未对应零件" + errorSuffix(hyp), "extra", "", function () {
        selected = { gt: null, hyp: index }
        renderAll()
      })
      if (selected && selected.hyp === index && !selected.gt) button.classList.add("on")
    })
  }

  function selectPart(partId) {
    const part = method().parts[partId - 1]
    selected = { gt: partId, hyp: part.hyp }
  }

  function activeHyp() {
    if (!selected || selected.hyp === null || selected.hyp === undefined) return null
    return method().hypotheses[selected.hyp]
  }

  function fillDetail() {
    if (!selected) {
      detail.textContent = "还没有选中。点一个零件，看它的误差，以及网格和真值是不是叠在一起。"
      return
    }
    if (selected.gt) {
      const part = method().parts[selected.gt - 1]
      const gt = data.gt[selected.gt - 1]
      const bits = ["零件 " + part.id, "可见度 " + gt.vis.toFixed(2), NAMES[part.kind]]
      if (part.add_mm !== null) bits.push("误差 " + mmText(part.add_mm))
      if (part.iou !== null) bits.push("掩膜重叠 " + part.iou.toFixed(2))
      if (gt.vis <= 0.6) bits.push("这篇论文的计分不包含它")
      if (part.kind === "miss") bits.push("没有一条检测和它重叠到 0.5")
      const linked = part.hyp === null || part.hyp === undefined ? null : method().hypotheses[part.hyp]
      const tag = errorTag(linked)
      if (tag) bits.push(tag)
      detail.textContent = bits.join(" · ")
      return
    }
    const extra = method().hypotheses[selected.hyp]
    detail.textContent = "这条候选姿态没有分到任何一个零件。" + (errorTag(extra) ? " · " + errorTag(extra) : "")
  }

  function resizePhoto() {
    const rect = photo.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    photo.width = Math.max(1, Math.round(rect.width * dpr))
    photo.height = Math.max(1, Math.round(rect.height * dpr))
  }

  function drawPhoto() {
    resizePhoto()
    const box = contain(photo.width, photo.height, data.width, data.height)
    photoCtx.setTransform(1, 0, 0, 1, 0, 0)
    photoCtx.fillStyle = "#141414"
    photoCtx.fillRect(0, 0, photo.width, photo.height)
    photoCtx.drawImage(image, box.ox, box.oy, box.dw, box.dh)
    photoCtx.setTransform(box.s, 0, 0, box.s, box.ox, box.oy)
    photoCtx.lineJoin = "round"
    photoCtx.lineCap = "round"
    const cssPx = photo.width / Math.max(photo.getBoundingClientRect().width, 1) / box.s
    const item = method()
    const order = ["extra", "wrong", "hit"]
    order.forEach(function (kind) {
      item.hypotheses.forEach(function (hyp, index) {
        if (hyp.kind !== kind) return
        const on = selected && selected.hyp === index
        const alpha = !selected ? 1 : (on ? 1 : 0.2)
        strokeRings(ringsOf(hyp), on ? "#ffd43b" : kindColor(kind), (on ? 3.2 : 2.6) * cssPx, alpha, [])
      })
    })
    data.gt.forEach(function (gt) {
      const on = selected && selected.gt === gt.id
      strokeRings(ringsOf(gt), "#17833d", (on ? 2.2 : 1.7) * cssPx, on || !selected ? 1 : 0.35, [5 * cssPx, 4 * cssPx])
    })
    item.parts.forEach(function (part) {
      const gt = data.gt[part.id - 1]
      if (!gt.contour.length) return
      const p = centroid2(gt.contour)
      photoCtx.font = "bold 16px sans-serif"
      photoCtx.fillStyle = "#111"
      photoCtx.fillText(String(part.id), p[0] + 1, p[1] + 1)
      photoCtx.fillStyle = "#fff"
      photoCtx.fillText(String(part.id), p[0], p[1])
    })
    item.hypotheses.forEach(function (hyp) {
      const tag = errorTag(hyp)
      if (!tag) return
      const ring = (hyp.rings && hyp.rings[0]) || hyp.contour
      if (!ring || ring.length < 3) return
      const at = centroid2(ring)
      const known = hyp.score_6d !== null && hyp.score_6d !== undefined
      const tone = !known ? "plain" : (hyp.score_6d >= (data.score_6d_high_min || 0.5) ? "high" : "low")
      drawTag(at[0], at[1] - 18 * cssPx, tag, tone, cssPx)
    })
  }

  function drawTag(x, y, text, tone, cssPx) {
    photoCtx.save()
    photoCtx.globalAlpha = 1
    photoCtx.setLineDash([])
    const size = Math.max(13 * cssPx, 12)
    photoCtx.font = "bold " + size + "px sans-serif"
    const pad = 4 * cssPx
    const width = photoCtx.measureText(text).width + pad * 2
    const height = size + pad * 1.4
    photoCtx.fillStyle = tone === "high" ? "#c92a2a" : (tone === "low" ? "#495057" : "#e8590c")
    photoCtx.fillRect(x - width / 2, y - height / 2, width, height)
    photoCtx.fillStyle = "#fff"
    photoCtx.textAlign = "center"
    photoCtx.textBaseline = "middle"
    photoCtx.fillText(text, x, y)
    photoCtx.restore()
  }

  function ringsOf(item) {
    if (item.rings && item.rings.length) return item.rings
    if (item.contour && item.contour.length) return [item.contour]
    return []
  }

  function strokeRings(rings, color, width, alpha, dash) {
    rings.forEach(function (ring) {
      if (ring.length < 2) return
      photoCtx.beginPath()
      ring.forEach(function (p, i) {
        if (i === 0) photoCtx.moveTo(p[0], p[1])
        else photoCtx.lineTo(p[0], p[1])
      })
      photoCtx.closePath()
      photoCtx.globalAlpha = alpha
      photoCtx.setLineDash(dash && dash.length ? dash : [])
      photoCtx.strokeStyle = "#111"
      photoCtx.lineWidth = width + Math.max(width * 0.55, 1)
      photoCtx.stroke()
      photoCtx.strokeStyle = color
      photoCtx.lineWidth = width
      photoCtx.stroke()
    })
    photoCtx.setLineDash([])
    photoCtx.globalAlpha = 1
  }

  function centroid2(poly) {
    let x = 0
    let y = 0
    poly.forEach(function (p) { x += p[0]; y += p[1] })
    return [x / poly.length, y / poly.length]
  }

  function area(poly) {
    let sum = 0
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      sum += poly[j][0] * poly[i][1] - poly[i][0] * poly[j][1]
    }
    return Math.abs(sum)
  }

  function inside(poly, x, y) {
    let hit = false
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i][0]
      const yi = poly[i][1]
      const xj = poly[j][0]
      const yj = poly[j][1]
      const cross = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi)
      if (cross) hit = !hit
    }
    return hit
  }

  function eventToImage(canvas, ev) {
    const rect = canvas.getBoundingClientRect()
    const x = (ev.clientX - rect.left) * (canvas.width / rect.width)
    const y = (ev.clientY - rect.top) * (canvas.height / rect.height)
    const box = contain(canvas.width, canvas.height, data.width, data.height)
    return [(x - box.ox) / box.s, (y - box.oy) / box.s]
  }

  function pickPhoto(ev) {
    const xy = eventToImage(photo, ev)
    const item = method()
    let best = null
    let bestArea = Infinity
    item.hypotheses.forEach(function (hyp, index) {
      if (hyp.contour.length < 3 || !inside(hyp.contour, xy[0], xy[1])) return
      const size = area(hyp.contour)
      if (size < bestArea) {
        bestArea = size
        best = index
      }
    })
    if (best === null) {
      selected = null
      return
    }
    const hyp = item.hypotheses[best]
    selected = { gt: hyp.gt, hyp: best }
  }

  function setProjection() {
    const fx = data.K[0]
    const fy = data.K[4]
    const cx = data.K[2]
    const cy = data.K[5]
    const w = data.width
    const h = data.height
    const near = 0.02
    const far = 4
    const m = new THREE.Matrix4()
    m.set(
      2 * fx / w, 0, 1 - 2 * cx / w, 0,
      0, 2 * fy / h, 2 * cy / h - 1, 0,
      0, 0, (far + near) / (near - far), 2 * far * near / (near - far),
      0, 0, -1, 0
    )
    camera.projectionMatrix.copy(m)
    camera.projectionMatrixInverse.copy(m).invert()
  }

  function layoutWorld() {
    const rect = world.getBoundingClientRect()
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    renderer.setPixelRatio(dpr)
    renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false)
    const bufW = world.width
    const bufH = world.height
    const box = contain(bufW, bufH, data.width, data.height)
    const oy = bufH - box.oy - box.dh
    renderer.setScissorTest(false)
    renderer.setViewport(0, 0, bufW, bufH)
    renderer.setClearColor(0x141414, 1)
    renderer.clear()
    renderer.setScissorTest(true)
    renderer.setViewport(box.ox, oy, box.dw, box.dh)
    renderer.setScissor(box.ox, oy, box.dw, box.dh)
    setProjection()
  }

  function makeMaterial(hex, opacity) {
    return new THREE.MeshLambertMaterial({
      color: new THREE.Color(hex),
      transparent: opacity < 0.99,
      opacity: opacity,
      side: THREE.DoubleSide,
      depthWrite: opacity > 0.9
    })
  }

  function rebuildMeshes() {
    meshes.forEach(function (mesh) {
      pivot.remove(mesh)
      mesh.material.dispose()
    })
    meshes = []
    if (wire) pivot.remove(wire)
    const item = method()
    const focus = selected !== null
    item.hypotheses.forEach(function (hyp, index) {
      const on = selected && selected.hyp === index
      const opacity = !focus ? 0.92 : (on ? 1 : 0.18)
      const color = on ? "#ffd43b" : kindColor(hyp.kind)
      const mesh = new THREE.Mesh(geometry, makeMaterial(color, opacity))
      mesh.matrixAutoUpdate = false
      mesh.matrix.copy(localMatrix(hyp.pose))
      mesh.userData.hyp = index
      mesh.userData.gt = hyp.gt
      pivot.add(mesh)
      meshes.push(mesh)
    })
    wire = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: 0xe03131, wireframe: true, depthTest: false
    }))
    wire.matrixAutoUpdate = false
    wire.raycast = function () {}
    wire.visible = false
    if (selected && selected.gt) {
      wire.matrix.copy(localMatrix(data.gt[selected.gt - 1].pose))
      wire.visible = true
    }
    pivot.add(wire)
  }

  function checkProjection() {
    camera.updateMatrixWorld(true)
    let maxErr = 0
    data.gt.forEach(function (gt) {
      const z = gt.pose[11]
      const u = data.K[0] * gt.pose[3] / z + data.K[2]
      const v = data.K[4] * gt.pose[7] / z + data.K[5]
      const p = new THREE.Vector4(gt.pose[3], -gt.pose[7], -gt.pose[11], 1)
      p.applyMatrix4(camera.projectionMatrix)
      const pu = (p.x / p.w * 0.5 + 0.5) * data.width
      const pv = (1 - (p.y / p.w * 0.5 + 0.5)) * data.height
      maxErr = Math.max(maxErr, Math.abs(pu - u), Math.abs(pv - v))
    })
    projCheck.textContent = maxErr.toFixed(3)
  }

  function pickWorld(ev) {
    const xy = eventToImage(world, ev)
    if (xy[0] < 0 || xy[1] < 0 || xy[0] > data.width || xy[1] > data.height) return
    const ndc = new THREE.Vector2((xy[0] / data.width) * 2 - 1, 1 - (xy[1] / data.height) * 2)
    const raycaster = new THREE.Raycaster()
    camera.updateMatrixWorld(true)
    pivot.updateMatrixWorld(true)
    raycaster.setFromCamera(ndc, camera)
    const hits = raycaster.intersectObjects(meshes, false)
    if (!hits.length) {
      selected = null
      return
    }
    const mesh = hits[0].object
    selected = { gt: mesh.userData.gt, hyp: mesh.userData.hyp }
  }

  function resetView() {
    pivot.rotation.set(0, 0, 0)
    pivot.position.copy(centroid)
  }

  function renderAll() {
    buildTabs()
    buildSummary()
    buildList()
    fillDetail()
    drawPhoto()
    rebuildMeshes()
    layoutWorld()
    renderer.render(scene, camera)
  }

  function animate() {
    requestAnimationFrame(animate)
    if (!renderer) return
    renderer.render(scene, camera)
  }

  function boot3d() {
    const positions = new Float32Array(data.mesh.vertices)
    geometry = new THREE.BufferGeometry()
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    geometry.setIndex(data.mesh.faces)
    geometry.computeVertexNormals()
    scene = new THREE.Scene()
    camera = new THREE.PerspectiveCamera(45, 1, 0.02, 4)
    camera.position.set(0, 0, 0)
    camera.up.set(0, 1, 0)
    camera.quaternion.identity()
    camera.updateMatrixWorld(true)
    centroid = new THREE.Vector3()
    data.gt.forEach(function (gt) {
      centroid.x += gt.pose[3]
      centroid.y += -gt.pose[7]
      centroid.z += -gt.pose[11]
    })
    centroid.multiplyScalar(1 / data.gt.length)
    pivot = new THREE.Group()
    pivot.position.copy(centroid)
    scene.add(pivot)
    scene.add(new THREE.AmbientLight(0xffffff, 0.72))
    const sun = new THREE.DirectionalLight(0xffffff, 0.85)
    sun.position.set(0.2, 0.3, 1)
    scene.add(sun)
    renderer = new THREE.WebGLRenderer({ canvas: world, antialias: true })
    if (THREE.SRGBColorSpace) renderer.outputColorSpace = THREE.SRGBColorSpace
    setProjection()
    checkProjection()
    animate()
  }

  photo.addEventListener("click", function (ev) {
    pickPhoto(ev)
    renderAll()
  })
  world.addEventListener("pointerdown", function (ev) {
    drag = { x: ev.clientX, y: ev.clientY, rx: pivot.rotation.x, ry: pivot.rotation.y, moved: false }
    world.setPointerCapture(ev.pointerId)
  })
  world.addEventListener("pointermove", function (ev) {
    if (!drag) return
    const dx = ev.clientX - drag.x
    const dy = ev.clientY - drag.y
    if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true
    pivot.rotation.y = drag.ry + dx * 0.005
    pivot.rotation.x = drag.rx + dy * 0.005
  })
  world.addEventListener("pointerup", function (ev) {
    if (drag && !drag.moved) {
      pickWorld(ev)
      renderAll()
    }
    drag = null
  })
  world.addEventListener("dblclick", function () {
    resetView()
  })
  world.addEventListener("wheel", function (ev) {
    ev.preventDefault()
    pivot.position.z -= ev.deltaY * 0.00045
  }, { passive: false })
  window.addEventListener("resize", function () {
    if (!data) return
    drawPhoto()
    layoutWorld()
  })

  const assetRoot = (function () {
    const host = document.getElementById("robi")
    const root = host && host.getAttribute("data-root")
    return root ? root : ""
  })()

  fetch(assetRoot + "scene.json?v=9").then(function (res) {
    if (!res.ok) throw new Error("scene.json " + res.status)
    return res.json()
  }).then(function (payload) {
    data = payload
    image = new Image()
    image.onload = function () {
      buildSpeeds()
      boot3d()
      renderAll()
    }
    image.src = assetRoot + data.image
  }).catch(function (err) {
    detail.textContent = "这一帧的数据还没有准备好。"
    console.error(err)
  })
}())
