// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

// Four 2D panels share saved nearest-ADD matches; no inference or 3D viewer.
// 四幅二维图片共用已保存的最近 ADD 匹配，不执行推理，不创建三维视图。
(function () {
  function initialize(host) {
  var root = host.getAttribute("data-root");
  var data;
  var photo;
  var panels = [];
  var blue = "#245db3";
  var red = "#c62828";

  function words(en, zh) {
    return document.documentElement.getAttribute("data-lang") === "zh" ? zh : en;
  }

  function drawRings(ctx, rings, crop, color, width, alpha) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.globalAlpha = alpha;
    for (var i = 0; i < rings.length; i++) {
      var ring = rings[i];
      if (!ring.length) continue;
      ctx.beginPath();
      for (var j = 0; j < ring.length; j++) {
        var x = (ring[j][0] - crop[0]) * 640 / (crop[2] - crop[0]);
        var y = (ring[j][1] - crop[1]) * ctx.canvas.height / (crop[3] - crop[1]);
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function update(panel) {
    var selected = panel.selected === null ? null : panel.matches[panel.selected];
    var prediction = selected ? panel.predictions[selected.source_index] : null;
    var crop = data.comparison_crop_xyxy.slice();
    // Keep the common crop by default. Expand it only to reveal a selected
    // prediction outside the tray; this affects drawing, never the ADD metric.
    // 默认使用共同裁剪；选中预测落在料盒外时才扩展视野。仅影响绘图，不影响 ADD。
    if (prediction && !(crop[0] === 0 && crop[1] === 0 && crop[2] === data.width && crop[3] === data.height)) {
      prediction.rings.forEach(function (ring) {
        ring.forEach(function (point) {
          crop[0] = Math.max(0, Math.min(crop[0], point[0] - 16));
          crop[1] = Math.max(0, Math.min(crop[1], point[1] - 16));
          crop[2] = Math.min(data.width, Math.max(crop[2], point[0] + 16));
          crop[3] = Math.min(data.height, Math.max(crop[3], point[1] + 16));
        });
      });
      var side = Math.max(crop[2] - crop[0], crop[3] - crop[1]);
      side = Math.min(side, data.width, data.height);
      crop[0] = Math.max(0, Math.min((crop[0] + crop[2] - side) / 2, data.width - side));
      crop[1] = Math.max(0, Math.min((crop[1] + crop[3] - side) / 2, data.height - side));
      crop[2] = crop[0] + side;
      crop[3] = crop[1] + side;
    }
    // Preserve the original aspect ratio in both full-frame and cropped views.
    // 完整画面与裁剪画面均保持原始宽高比。
    panel.canvas.height = Math.round(640 * (crop[3] - crop[1]) / (crop[2] - crop[0]));
    var ctx = panel.canvas.getContext("2d");
    ctx.clearRect(0, 0, 640, panel.canvas.height);
    ctx.drawImage(photo, crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1], 0, 0, 640, panel.canvas.height);
    panel.method.hypotheses.forEach(function (hypothesis) {
      // Draw both outcomes; reused predictions appear once in the overview.
      // 正负结果均绘制；总览中被多个标注复用的预测只画一次。
      drawRings(ctx, hypothesis.rings, crop, hypothesis.within_threshold ? blue : red,
        2.2, selected ? 0.16 : 1);
    });
    panel.info.hidden = !selected;
    panel.reset.textContent = words("Reset", "重置");
    panel.count.textContent = words("Coverage ", "覆盖 ") + panel.method.matched_truth_count + "/" + panel.method.truth_count;
    if (panel.method.availability === "saved_final_only") panel.count.textContent += words(" · subset", " · 子集");
    panel.legend.textContent = words("Blue: positive < 0.1d · Red: negative ≥ 0.1d",
      "蓝：正样本 < 0.1d · 红：负样本 ≥ 0.1d");
    panel.buttons.forEach(function (button, index) {
      var match = panel.matches[index];
      var passing = match.add_mm < data.threshold_mm;
      button.setAttribute("aria-pressed", String(index === panel.selected));
      var description = words("Instance ", "实例 ") + match.truth_id.replace("Object_", "")
        + " · " + words(passing ? "Positive" : "Negative", passing ? "正样本" : "负样本")
        + " · ADD " + match.add_mm.toFixed(2) + " mm";
      button.setAttribute("aria-label", description);
      button.title = description;
    });
    if (!selected) {
      panel.readout.textContent = words("Click an instance number to inspect its status and ADD error.",
        "点击实例编号，查看正负状态与 ADD 误差。");
      panel.readout.classList.remove("is-failed", "is-passed");
      return;
    }
    var passing = selected.add_mm < data.threshold_mm;
    drawRings(ctx, prediction.rings, crop, "#fff", 6, 1);
    drawRings(ctx, prediction.rings, crop, passing ? blue : red, 3.4, 1);
    panel.info.classList.toggle("is-failed", !passing);
    panel.infoTitle.textContent = words("Instance #", "实例 #") + selected.truth_id.replace("Object_", "")
      + " · " + words(passing ? "Positive" : "Negative", passing ? "正样本" : "负样本");
    panel.infoMetric.textContent = "ADD " + selected.add_mm.toFixed(2) + " mm · ADD/d "
      + (selected.add_mm / data.diameter_mm).toFixed(3);
    // Scores are displayed only when supplied. Baseline file order is never a score.
    // 仅显示实际提供的评分；基线文件顺序不解释为置信度。
    panel.infoScore.textContent = prediction.score_6d === null || prediction.score_6d === undefined
      ? words("Confidence: not provided", "置信度：未提供")
      : "WBPS (6D) " + prediction.score_6d.toFixed(3);
    panel.readout.textContent = panel.infoTitle.textContent + " · " + panel.infoMetric.textContent
      + words(" · Threshold ", " · 阈值 ") + data.threshold_mm.toFixed(2) + " mm";
    panel.readout.classList.toggle("is-failed", !passing);
    panel.readout.classList.toggle("is-passed", passing);
    // Place the detail box opposite the selected contour's center.
    // 详情框放在选中轮廓中心的对侧，减少对目标的遮挡。
    var points = prediction.rings.reduce(function (all, ring) { return all.concat(ring); }, []);
    var centerX = points.length ? points.reduce(function (sum, p) { return sum + p[0]; }, 0) / points.length : crop[0];
    var centerY = points.length ? points.reduce(function (sum, p) { return sum + p[1]; }, 0) / points.length : crop[1];
    panel.info.classList.toggle("at-right", centerX < (crop[0] + crop[2]) / 2);
    panel.info.classList.toggle("at-bottom", centerY < (crop[1] + crop[3]) / 2);
  }

  function createPanel(name) {
    var method = data.methods.find(function (item) { return item.name === name; });
    var card = document.createElement("article");
    card.className = "robi-card";
    card.setAttribute("aria-label", name);
    var heading = document.createElement("div");
    heading.className = "robi-card-heading";
    var title = document.createElement("h3");
    title.textContent = name === "PPF" ? "PPF + ICP" : name;
    var count = document.createElement("span");
    var reset = document.createElement("button");
    reset.type = "button";
    reset.className = "robi-reset";
    heading.append(title, count, reset);
    var frame = document.createElement("div");
    frame.className = "robi-card-frame";
    var canvas = document.createElement("canvas");
    canvas.width = canvas.height = 640;
    canvas.setAttribute("aria-label", name + " · nearest ADD pose comparison");
    canvas.setAttribute("role", "img");
    var info = document.createElement("div");
    info.className = "robi-instance-info";
    info.setAttribute("aria-live", "polite");
    var infoTitle = document.createElement("strong");
    var infoMetric = document.createElement("span");
    var infoScore = document.createElement("span");
    info.append(infoTitle, infoMetric, infoScore);
    frame.append(canvas, info);
    var legend = document.createElement("p");
    legend.className = "robi-card-legend";
    var chips = document.createElement("div");
    chips.className = "robi-instance-buttons";
    chips.setAttribute("role", "group");
    chips.setAttribute("aria-label", name + " · instances");
    var readout = document.createElement("p");
    readout.className = "robi-instance-readout";
    readout.setAttribute("aria-live", "polite");
    var panel = { method: method, canvas: canvas, count: count, reset: reset, legend: legend,
      info: info, infoTitle: infoTitle, infoMetric: infoMetric, infoScore: infoScore, readout: readout,
      matches: method.nearest_matches.slice(), predictions: {}, buttons: [], selected: null };
    panel.matches.sort(function (a, b) {
      return Number(a.truth_id.replace("Object_", "")) - Number(b.truth_id.replace("Object_", ""));
    });
    method.hypotheses.forEach(function (hypothesis) { panel.predictions[hypothesis.source_index] = hypothesis; });
    panel.matches.forEach(function (match, index) {
      var button = document.createElement("button");
      button.type = "button";
      button.textContent = "#" + match.truth_id.replace("Object_", "");
      button.className = match.add_mm < data.threshold_mm ? "is-passed" : "is-failed";
      button.addEventListener("click", function () {
        panel.selected = panel.selected === index ? null : index;
        update(panel);
      });
      chips.appendChild(button);
      panel.buttons.push(button);
    });
    reset.addEventListener("click", function () { panel.selected = null; update(panel); });
    card.append(heading, frame, legend, chips, readout);
    host.appendChild(card);
    panels.push(panel);
    update(panel);
  }

  var requestNumber = 0;
  async function loadScene(sceneRoot) {
    var currentRequest = ++requestNumber;
    host.textContent = words("Loading comparisons…", "正在加载对比图片…");
    try {
      var response = await fetch(sceneRoot + "nearest_scene.json?v=20261005-time-audit");
      if (!response.ok) throw new Error("ROBI comparison data " + response.status);
      var saved = await response.json();
      var savedPhoto = new Image();
      await new Promise(function (resolve, reject) {
        savedPhoto.onload = resolve;
        savedPhoto.onerror = reject;
        savedPhoto.src = sceneRoot + saved.image;
      });
      // Rapid scene switching must not restore an earlier request's panels.
      // 快速切换场景时，较早请求不能覆盖当前场景。
      if (currentRequest !== requestNumber) return;
      data = saved;
      photo = savedPhoto;
      panels = [];
      host.replaceChildren();
      ["WAPR", "AAE", "PPF", "Line2D"].forEach(createPanel);
    } catch (error) {
      if (currentRequest === requestNumber) host.textContent = words("Comparison could not load. Please reload the page.", "对比图片加载失败，请刷新页面。");
    }
  }
  var figure = host.closest("figure");
  if (figure) figure.querySelectorAll("[data-robi-root]").forEach(function (button) {
    button.addEventListener("click", function () {
      figure.querySelectorAll("[data-robi-root]").forEach(function (item) {
        item.setAttribute("aria-pressed", String(item === button));
      });
      loadScene(button.getAttribute("data-robi-root"));
    });
  });
  document.addEventListener("wapr-lang-change", function () { panels.forEach(update); });
  loadScene(root);
  }
  document.querySelectorAll(".robi-interactive").forEach(initialize);
})();
