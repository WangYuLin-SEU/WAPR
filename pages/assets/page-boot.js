// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

// 宣传页按区块加载。首屏只保留文字和已经进入视口的图。
// The project page loads each heavy block when the reader is near it.
// Videos and the WebGL viewers stay idle until their sections approach.
(function () {
  var near = "720px 0px";

  function whenNear(id, start) {
    var node = document.getElementById(id);
    if (!node) return;
    if (typeof IntersectionObserver !== "function") {
      start();
      return;
    }
    var obs = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (!entries[i].isIntersecting) continue;
        obs.disconnect();
        start();
        return;
      }
    }, { rootMargin: near });
    obs.observe(node);
  }

  var pending = {};

  function loadScript(src) {
    if (pending[src]) return pending[src];
    pending[src] = new Promise(function (resolve, reject) {
      var script = document.createElement("script");
      script.src = src;
      script.onload = function () { resolve(); };
      script.onerror = function () {
        delete pending[src];
        reject(new Error(src));
      };
      document.body.appendChild(script);
    });
    return pending[src];
  }

  function withThree(src) {
    // Reuse Three.js after another 3D section has loaded it.
    // 其他三维区块已加载 Three.js 时复用同一实例。
    var threeReady = window.THREE ? Promise.resolve() : loadScript("assets/three.min.js");
    return threeReady.then(function () {
      return loadScript("assets/studio_grid.js?v=2");
    }).then(function () {
      return loadScript(src);
    });
  }

  whenNear("showcase", function () {
    withThree("assets/showcase.js?v=20261006-perf1").catch(function () {
      var ids = ["demo-photo-status", "demo-cloud-status"];
      for (var i = 0; i < ids.length; i++) {
        var status = document.getElementById(ids[i]);
        if (!status) continue;
        status.querySelector(".en").textContent = "Viewer script could not load.";
        status.querySelector(".zh").textContent = "查看器脚本加载失败。";
        var retry = status.querySelector("button");
        retry.hidden = false;
        retry.onclick = function () { window.location.reload(); };
      }
    });
  });
  whenNear("lmo-det", function () { withThree("assets/lmo_det.js?v=20261006-perf1"); });
  // The project page has one reconstruction showcase, selected by language.
  // 宣传页只展示语言提示重建，点提示查看器保留在文档内。
  whenNear("reconstruct", function () { withThree("assets/language.js?v=20261006-perf1"); });
  whenNear("robi", function () { loadScript("assets/robi_compare.js?v=20261005-time-audit"); });
  whenNear("wedge", function () { withThree("assets/wedge_view.js?v=20261006-perf1"); });

})();
