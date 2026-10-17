// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

// Documentation clips start muted and loop when their content is visible.
// 文档视频在内容可见时自动静音循环播放，无需点击播放按钮。
(function () {
  if (window.WAPRRender) return; // Shared loader owns playback. / 共用加载器接管播放。
  function boot() {
    var videos = Array.prototype.slice.call(document.querySelectorAll(".docs-main video"));
    if (!videos.length) return;
    var nearby = new Set();
    var observer;

    function shouldPlay(video) {
      return !document.hidden && video.getClientRects().length > 0
        && (!observer || nearby.has(video));
    }

    function update(video) {
      if (!shouldPlay(video)) {
        video.pause();
        return;
      }
      video.preload = "auto";
      var playing = video.play();
      // Loading or a language change may interrupt play; readiness events retry.
      // 加载或语言切换可能中断播放请求；数据就绪事件会再次尝试。
      if (playing && playing.catch) playing.catch(function () {});
    }

    function updateAll() {
      videos.forEach(update);
    }

    if (typeof IntersectionObserver === "function") {
      observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) nearby.add(entry.target);
          else nearby.delete(entry.target);
          update(entry.target);
        });
      }, { rootMargin: "240px 0px" });
    }
    videos.forEach(function (video) {
      video.autoplay = true;
      video.loop = true;
      video.defaultMuted = true;
      video.muted = true;
      video.playsInline = true;
      video.addEventListener("loadeddata", function () { update(video); });
      video.addEventListener("playing", function () {
        if (!shouldPlay(video)) video.pause();
      });
      if (observer) observer.observe(video);
      update(video);
    });
    // Language, disclosure and tab visibility changes also update playback.
    // 切换语言、折叠正文或切换标签页时，同步播放状态。
    document.addEventListener("wapr-lang-change", updateAll);
    document.addEventListener("toggle", updateAll, true);
    document.addEventListener("visibilitychange", updateAll);
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
