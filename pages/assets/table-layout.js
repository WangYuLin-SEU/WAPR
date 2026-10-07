// Author: Yulin Wang (yulinwang@seu.edu.cn)
// School of Mechanical Engineering, Southeast University, China
// Copyright (c) 2026 Yulin Wang. All rights reserved, except as granted under LICENSE.
// SPDX-License-Identifier: LGPL-2.1-only
// 作者与版权人：Yulin Wang；使用、修改与再分发须遵守项目 LICENSE。
// Third-party portions retain their original notices and terms; see THIRD_PARTY_NOTICES.txt.

(function () {
  function languageText(element, language) {
    // Read labels without the other language or citation numbers.
    // 标签不包含另一种语言或引用编号；原单元格与引用链接保持原样。
    var copy = element.cloneNode(true);
    copy.querySelectorAll("." + (language === "en" ? "zh" : "en") + ", sup").forEach(function (node) {
      node.remove();
    });
    copy.querySelectorAll("br").forEach(function (node) { node.replaceWith(" "); });
    copy.querySelectorAll(".unit, .sub, .score-note").forEach(function (node) {
      node.prepend(document.createTextNode(" · "));
    });
    return copy.textContent.replace(/\s+/g, " ").trim();
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".docs-main table, .score-wrap table").forEach(function (table) {
      if (!table.tHead || !table.tBodies.length) return;
      // Resolve grouped headers by their actual column and row spans.
      // 按实际合并行列解析分组表头，使窄屏标签仍能区分方法、指标与单位。
      var headers = [];
      Array.from(table.tHead.rows).forEach(function (row, rowIndex) {
        if (!headers[rowIndex]) headers[rowIndex] = [];
        var column = 0;
        Array.from(row.cells).forEach(function (cell) {
          while (headers[rowIndex][column]) column++;
          for (var y = rowIndex; y < rowIndex + cell.rowSpan; y++) {
            if (!headers[y]) headers[y] = [];
            for (var x = column; x < column + cell.colSpan; x++) headers[y][x] = cell;
          }
          column += cell.colSpan;
        });
      });
      var columnLabels = [];
      for (var column = 0; column < headers[0].length; column++) {
        var cells = [];
        headers.forEach(function (row) {
          if (row[column] && cells.indexOf(row[column]) < 0) cells.push(row[column]);
        });
        columnLabels.push({
          en: cells.map(function (cell) { return languageText(cell, "en"); }).join(" · "),
          zh: cells.map(function (cell) { return languageText(cell, "zh"); }).join(" · ")
        });
      }

      Array.from(table.tBodies).forEach(function (body) {
        var bodyGrid = [];
        Array.from(body.rows).forEach(function (row, rowIndex) {
          if (!bodyGrid[rowIndex]) bodyGrid[rowIndex] = [];
          var column = 0;
          Array.from(row.cells).forEach(function (cell) {
            while (bodyGrid[rowIndex][column]) column++;
            var labels = columnLabels.slice(column, column + cell.colSpan);
            cell.setAttribute("data-label-en", labels.map(function (label) { return label.en; }).join(" / "));
            cell.setAttribute("data-label-zh", labels.map(function (label) { return label.zh; }).join(" / "));
            var lastRow = cell.rowSpan === 0 ? body.rows.length : rowIndex + cell.rowSpan;
            for (var y = rowIndex; y < lastRow; y++) {
              if (!bodyGrid[y]) bodyGrid[y] = [];
              for (var x = column; x < column + cell.colSpan; x++) bodyGrid[y][x] = cell;
            }
            column += cell.colSpan;
          });
          // Repeat only the context of carried cells, without copying their IDs.
          // 后续行补充跨行单元格的上下文，不复制原节点或引用 ID。
          var carried = bodyGrid[rowIndex].filter(function (cell, index, cells) {
            return cell.parentElement !== row && cells.indexOf(cell) === index;
          });
          if (carried.length) {
            ["en", "zh"].forEach(function (language) {
              row.setAttribute("data-row-group-" + language, carried.map(function (cell) {
                return cell.getAttribute("data-label-" + language) + ": " + languageText(cell, language);
              }).join(" · "));
            });
          }
        });
      });
      table.classList.add("responsive-table");
      // Use the available article width, including the desktop sidebar.
      // 依据实际正文宽度切换布局，桌面侧栏占用的空间也计算在内。
      var observer = new ResizeObserver(function (entries) {
        table.classList.toggle("table-stacked", entries[0].contentRect.width < 700);
      });
      observer.observe(table.parentElement);
    });
  });
})();
