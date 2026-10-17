// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  function pageFile() {
    var path = location.pathname.replace(/\/+$/, "");
    var name = path.split("/").pop();
    if (!name || name.indexOf(".") < 0) return "index.html";
    return name;
  }

  function hrefFile(href) {
    var clean = (href || "").split("#")[0].split("?")[0];
    if (clean === "" || clean === "." || clean === "./") return "index.html";
    return clean.split("/").pop();
  }

  function hrefHash(href) {
    var i = (href || "").indexOf("#");
    return i >= 0 ? href.slice(i) : "";
  }

  function setOpen(fold, btn, panel, open) {
    btn.setAttribute("aria-expanded", open ? "true" : "false");
    panel.hidden = !open;
    fold.classList.toggle("is-open", open);
  }

  function makeFold(link, kids, open) {
    var fold = document.createElement("div");
    fold.className = "nav-fold";
    var row = document.createElement("div");
    row.className = "nav-fold-row";
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nav-fold-btn";
    var closed = document.createElement("span");
    closed.className = "nav-fold-when-closed";
    closed.innerHTML = '<span class="en">Expand</span><span class="zh">展开</span>';
    var opened = document.createElement("span");
    opened.className = "nav-fold-when-open";
    opened.innerHTML = '<span class="en">Collapse</span><span class="zh">收起</span>';
    btn.appendChild(closed);
    btn.appendChild(opened);
    var panel = document.createElement("div");
    panel.className = "nav-fold-kids";
    row.appendChild(link);
    row.appendChild(btn);
    fold.appendChild(row);
    for (var i = 0; i < kids.length; i++) panel.appendChild(kids[i]);
    fold.appendChild(panel);
    setOpen(fold, btn, panel, open);
    btn.addEventListener("click", function () {
      setOpen(fold, btn, panel, btn.getAttribute("aria-expanded") !== "true");
    });
    return fold;
  }

  function foldLeaves(nav) {
    var nodes = Array.prototype.slice.call(nav.children);
    var i = 0;
    while (i < nodes.length) {
      var node = nodes[i];
      if (!node.tagName || node.tagName !== "A" || !node.classList.contains("sub")) {
        i += 1;
        continue;
      }
      var kids = [];
      var j = i + 1;
      while (j < nodes.length && nodes[j].tagName === "A" && nodes[j].classList.contains("sub3")) {
        kids.push(nodes[j]);
        j += 1;
      }
      if (kids.length) {
        // Start folded; the current page's ancestor groups open after marking it.
        // 默认收起；标记当前页后再展开其所属分组。
        nav.insertBefore(makeFold(node, kids, false), nodes[j] || null);
      }
      i = j;
    }
  }

  function foldSection(nav, pageName) {
    var nodes = Array.prototype.slice.call(nav.children);
    for (var i = 0; i < nodes.length; i++) {
      var node = nodes[i];
      if (!node.tagName || node.tagName !== "A") continue;
      if (node.classList.contains("sub") || node.classList.contains("sub3")) continue;
      if (hrefFile(node.getAttribute("href")) !== pageName) continue;
      var kids = [];
      var j = i + 1;
      while (j < nodes.length) {
        var next = nodes[j];
        var isSub = next.tagName === "A" && next.classList.contains("sub");
        var isFold = next.classList && next.classList.contains("nav-fold");
        if (!isSub && !isFold) break;
        kids.push(next);
        j += 1;
      }
      if (!kids.length) return;
      nav.insertBefore(makeFold(node, kids, false), nodes[j] || null);
      return;
    }
  }

  function markPlace() {
    var file = pageFile();
    var hash = location.hash;
    var links = document.querySelectorAll(".docs-nav a");
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var href = a.getAttribute("href") || "";
      var here = hrefFile(href) === file && (!hrefHash(href) || hrefHash(href) === hash);
      if (here) a.setAttribute("aria-current", "page");
      else a.removeAttribute("aria-current");
    }
  }

  function boot() {
    var nav = document.querySelector(".docs-nav");
    if (!nav || nav.getAttribute("data-folded") === "1") return;
    nav.setAttribute("data-folded", "1");
    nav.id = "docs-navigation";
    var menu = document.createElement("button");
    menu.type = "button";
    menu.className = "docs-menu-toggle";
    menu.setAttribute("aria-controls", nav.id);
    menu.setAttribute("aria-expanded", "true");
    menu.innerHTML = '<span class="en">Contents</span><span class="zh">目录</span>';
    // The mobile menu shows the chapter list, with only the current branch open.
    // 移动端显示章节列表，仅展开当前页所在分支。
    nav.setAttribute("data-mobile-open", "true");
    nav.parentNode.insertBefore(menu, nav);
    menu.addEventListener("click", function () {
      var open = menu.getAttribute("aria-expanded") !== "true";
      menu.setAttribute("aria-expanded", open ? "true" : "false");
      nav.setAttribute("data-mobile-open", open ? "true" : "false");
    });
    var chapters = Array.prototype.slice.call(nav.children).filter(function (link) {
      return link.tagName === "A" && !link.classList.contains("sub") && !link.classList.contains("sub3");
    });
    foldLeaves(nav);
    chapters.forEach(function (link) { foldSection(nav, hrefFile(link.getAttribute("href"))); });
    markPlace();
    // Open only the current page's ancestors, keeping other chapters folded.
    // 仅展开当前页的上级目录，其他章节保持收起。
    var current = nav.querySelector('a[aria-current="page"]');
    var ancestor = current ? current.parentNode : null;
    while (ancestor && ancestor !== nav) {
      if (ancestor.classList.contains("nav-fold")) {
        var row = ancestor.firstElementChild;
        setOpen(ancestor, row.querySelector(".nav-fold-btn"), ancestor.lastElementChild, true);
      }
      ancestor = ancestor.parentNode;
    }
    window.addEventListener("hashchange", markPlace);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
