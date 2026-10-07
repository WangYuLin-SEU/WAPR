// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

/* Promo image. Both page layouts use the same centered width. The picture itself
   does not scale and is not resized while scrolling. Scrolling covers the
   bottom of the picture and fades it toward 50%. The fade takes longer
   than the cover. Where the browser can, the cover is a scroll animation,
   so the wheel does not wait on a layout pass.
   宣传图。两种页面共用居中的宽度。图片本身不缩放，滚动时也不改它的尺寸。
   往下滚是把图片下半部分挡住，并淡到 50%。淡出比遮挡更慢。
   浏览器做得到时，遮挡是滚动动画，滚轮不用等一次布局。 */
(function () {
  var mast = document.querySelector(".mast");
  var hero = document.querySelector(".mast-hero");
  var bar = document.querySelector(".mast .docs-top") || document.querySelector(".mast .langbar");
  var spacer = document.querySelector(".mast-spacer");
  if (!mast || !hero || !bar || !spacer) return;

  var scheduled = false;
  var needMeasure = true;
  var aspect = 724 / 2172;
  /* Collapsed band is about 10% shorter than 0.48. The subtitle still clears.
     收起后比 0.48 再矮大约 10%，副标题还在。 */
  var keepFrac = 0.432;
  /* Crop tracks the scroll. A slower crop has to reserve empty space under
     the open picture, and that band is what shows at the top of the page.
     裁切跟滚动走。再放慢就要在展开的图下面空出一段，滚到顶时那段就是空白。 */
  var cropSlow = 1;
  /* Opacity reaches 50% later than the crop finishes.
     透明度比裁切更晚才到 50%。 */
  var fadeSlow = 4.2;
  var fullH = 300;
  var barH = 48;
  var crop = 1;
  /* Scroll-linked cover. The script does not move the bar on each wheel tick.
     遮挡跟滚动绑定。脚本不在每一次滚轮时去挪那条栏。 */
  var scrollLinked = window.CSS && CSS.supports("animation-timeline", "scroll()");
  var shownCover = "";
  var shownOp = "";
  var docs = document.body.classList.contains("docs");
  var navigationKey = "wapr-docs-banner-navigation";
  var initialProgress = 0;
  var scrollOffset = 0;
  if (docs) {
    /* Restore only this navigation's banner progress, not the old article scroll.
       History also retains the inherited offset on reload and back/forward.
       只恢复本次切页的顶图进度，不恢复旧正文滚动位置；历史项保留继承的偏移，
       供刷新及前进、后退使用。 */
    try {
      var saved = JSON.parse(sessionStorage.getItem(navigationKey) || "null");
      sessionStorage.removeItem(navigationKey);
      var progress = history.state && history.state.waprDocsBannerProgress;
      if (saved && saved.destination === location.href) progress = saved.progress;
      if (typeof progress === "number" && isFinite(progress)) {
        initialProgress = Math.max(0, Math.min(progress, fadeSlow));
      }
      var entryState = Object.assign({}, history.state || {});
      entryState.waprDocsBannerProgress = initialProgress;
      history.replaceState(entryState, "");
    } catch (error) {
      /* Storage may be unavailable; ordinary scroll animation still works.
         存储不可用时，保留正常的滚动动画。 */
    }
    document.addEventListener("click", function (event) {
      if (event.defaultPrevented || event.button !== 0 || event.ctrlKey ||
          event.metaKey || event.shiftKey || event.altKey) return;
      var link = event.target.closest && event.target.closest("a[href]");
      if (!link || link.hasAttribute("download") ||
          (link.target && link.target !== "_self")) return;
      var destination = new URL(link.href, location.href);
      var directory = location.pathname.slice(0, location.pathname.lastIndexOf("/") + 1);
      if (destination.origin !== location.origin ||
          !destination.pathname.startsWith(directory) ||
          !destination.pathname.endsWith(".html") ||
          destination.pathname === location.pathname) return;
      var y = window.scrollY || document.documentElement.scrollTop || 0;
      try {
        sessionStorage.setItem(navigationKey, JSON.stringify({
          destination: destination.href,
          progress: Math.min((y + scrollOffset) / crop, fadeSlow)
        }));
      } catch (error) {}
    });
  }

  function setVar(name, value) {
    document.documentElement.style.setProperty(name, value);
    document.body.style.setProperty(name, value);
  }

  function column() {
    /* Read the shared CSS dimensions so resize keeps both layouts aligned.
       从共用的 CSS 尺寸计算，让窗口缩放后两种页面仍然对齐。 */
    var style = getComputedStyle(document.body);
    var maxWidth = parseFloat(style.getPropertyValue("--mast-max-w"));
    var gutter = parseFloat(style.getPropertyValue("--mast-gutter"));
    var viewportWidth = mast.getBoundingClientRect().width;
    var width = Math.min(maxWidth, viewportWidth - 2 * gutter);
    return { left: (viewportWidth - width) / 2, width: width };
  }

  function measure() {
    if (hero.naturalWidth) aspect = hero.naturalHeight / hero.naturalWidth;
    var col = column();
    fullH = col.width * aspect;
    var keepH = fullH * keepFrac;
    barH = bar.offsetHeight || 48;
    crop = Math.max(fullH - keepH, 1);
    /* An inherited virtual scroll keeps crop/fade unchanged at the new page's top.
       Normalize it by crop so a viewport resize preserves the same progress.
       继承的虚拟滚动在新正文顶部保持裁切及淡出；按裁切距离归一化，窗口变化
       后仍保持相同进度。图片尺寸及原有动画速度不变。 */
    scrollOffset = initialProgress * crop;
    setVar("--mast-scroll-start", (-scrollOffset) + "px");
    setVar("--mast-offset", scrollOffset + "px");
    setVar("--mast-pin", (keepH + barH) + "px");
    setVar("--mast-img-left", col.left + "px");
    setVar("--mast-img-w", col.width + "px");
    setVar("--mast-img-h", fullH + "px");
    /* Cover distance and the longer fade. Neither value resizes the picture.
       遮挡距离，以及更长的淡出。这两个数都不改图片尺寸。 */
    setVar("--mast-crop", crop + "px");
    setVar("--mast-fade", (crop * fadeSlow) + "px");
    /* The box stays at the open size. Scrolling only covers it.
       盒子保持展开时的高度。滚动只做遮挡。 */
    mast.style.setProperty("--mast-h", (fullH + barH) + "px");
    mast.style.setProperty("--mast-bar-top", fullH + "px");
    spacer.style.height = (fullH + barH - Math.min(scrollOffset, crop)) + "px";
    mast.style.clipPath = "";
    bar.style.top = "";
    bar.style.transform = "";
    hero.style.opacity = "";
    shownCover = "";
    shownOp = "";
    /* A short page has no active scroll timeline; its base style still restores the banner.
       短页面可能没有有效滚动时间线，基础样式仍须恢复顶图状态。 */
    paint();
  }

  function paint() {
    /* Fallback for a browser without a scroll timeline. Writes two variables.
       It does not change the picture width or height, and it does not set top.
       没有滚动时间线时的退路。只写两个变量。不改图片宽高，也不写 top。 */
    var y = (window.scrollY || document.documentElement.scrollTop || 0) + scrollOffset;
    var span = crop * cropSlow;
    var fadeSpan = crop * fadeSlow;
    var tCrop = y <= 0 ? 0 : (y >= span ? 1 : y / span);
    var tFade = y <= 0 ? 0 : (y >= fadeSpan ? 1 : y / fadeSpan);
    var nextCover = (crop * tCrop).toFixed(2) + "px";
    var nextOp = (1 - 0.5 * tFade).toFixed(3);
    if (nextCover !== shownCover) {
      setVar("--mast-cover", nextCover);
      shownCover = nextCover;
    }
    if (nextOp !== shownOp) {
      setVar("--mast-op", nextOp);
      shownOp = nextOp;
    }
  }

  function schedule(remeasure) {
    if (remeasure) needMeasure = true;
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(function () {
      scheduled = false;
      if (needMeasure) {
        needMeasure = false;
        measure();
      } else {
        paint();
      }
    });
  }

  measure();
  requestAnimationFrame(function () { schedule(true); });
  if (!scrollLinked) {
    window.addEventListener("scroll", function () { schedule(false); }, { passive: true });
  }
  window.addEventListener("resize", function () { schedule(true); });
  /* Language changes may resize the bar; keep the sidebar's pin height in sync.
     语言切换可能改变栏高，同步目录吸顶高度，避免继续沿用另一语言的测量。 */
  document.addEventListener("wapr-lang-change", function () { schedule(true); });
  window.addEventListener("load", function () { schedule(true); });
  if (!hero.complete) hero.addEventListener("load", function () { schedule(true); });
})();
