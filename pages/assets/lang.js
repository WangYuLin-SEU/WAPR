// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  var key = "wapr-lang";

  function saved() {
    try {
      return localStorage.getItem(key) === "zh" ? "zh" : "en";
    } catch (err) {
      return "en";
    }
  }

  function apply(lang) {
    document.documentElement.setAttribute("data-lang", lang);
    document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
    try {
      localStorage.setItem(key, lang);
    } catch (err) {
      /* keep the visible language even if storage is blocked */
    }
    var title = document.body && document.body.getAttribute("data-title-" + lang);
    if (title) {
      document.title = title;
    }
    var buttons = document.querySelectorAll("[data-set-lang]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].setAttribute("aria-pressed", buttons[i].getAttribute("data-set-lang") === lang ? "true" : "false");
    }
    document.dispatchEvent(new CustomEvent("wapr-lang-change", { detail: lang }));
    var clips = document.querySelectorAll("video.en, video.zh");
    for (var j = 0; j < clips.length; j++) {
      if (clips[j].offsetParent === null) clips[j].pause();
      else if (clips[j].autoplay && !clips[j].hasAttribute("data-autoplay")) {
        var play = clips[j].play();
        if (play && play.catch) play.catch(function () {});
      }
    }
  }

  document.addEventListener("DOMContentLoaded", function () {
    // Links from the bilingual READMEs select a language before the saved preference.
    // 双语 README 的链接优先指定语言；未指定时沿用用户已保存的选择。
    var requested = new URLSearchParams(location.search).get("lang");
    var initial = requested === "en" || requested === "zh"
      ? requested : document.documentElement.getAttribute("data-lang") || saved();
    apply(initial);
    var buttons = document.querySelectorAll("[data-set-lang]");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].addEventListener("click", function (event) {
        apply(event.currentTarget.getAttribute("data-set-lang"));
      });
    }
  });
})();
