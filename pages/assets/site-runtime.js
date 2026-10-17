// Shared media loading and redraw scheduling. / 共用媒体加载与按需绘制。
(function () {
  var views = [];
  var pending = false;
  function invalidate() {
    if (pending || document.hidden) return;
    pending = true;
    requestAnimationFrame(function () {
      pending = false;
      views.forEach(function (view) {
        var box = view.canvas.getBoundingClientRect();
        if (box.width && box.height && box.bottom > 0 && box.top < innerHeight) view.draw();
      });
    });
  }
  window.WAPRRender = {
    register: function (canvas, draw) {
      views.push({canvas: canvas, draw: draw});
      if (window.ResizeObserver) new ResizeObserver(invalidate).observe(canvas.parentElement);
      invalidate();
    },
    invalidate: invalidate
  };
  ['pointermove', 'pointerup', 'wheel', 'click', 'input', 'change', 'load', 'wapr-lang-change'].forEach(function (name) {
    document.addEventListener(name, invalidate, true);
  });
  window.addEventListener('resize', invalidate);
  window.addEventListener('scroll', invalidate, {passive: true});
  document.addEventListener('visibilitychange', invalidate);

  // Decode lossless mesh companions; older browsers retain the original path.
  // 解码无损网格副本；旧浏览器仍读取原始文件。
  window.WAPRAssets = {
    fetch: async function (url, options) {
      var response;
      if (/\.bin(?:\?|$)/.test(url) && window.DecompressionStream) {
        try {
          var packed = await fetch(url.replace(/\.bin(?=\?|$)/, '.bin.gz'), options);
          if (!packed.ok) throw new Error('Compressed asset unavailable');
          var bytes = await packed.arrayBuffer();
          // Some hosts decode gzip automatically. / 兼容托管服务自动解压。
          var header = new Uint8Array(bytes, 0, Math.min(2, bytes.byteLength));
          if (header[0] === 31 && header[1] === 139) {
            bytes = await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
          }
          response = new Response(bytes);
        } catch (error) { response = await fetch(url, options); }
      } else { response = await fetch(url, options); }
      // Redraw after callers have installed the decoded scene data.
      // 等调用方安装解码后的场景数据，再触发绘制。
      ['json', 'arrayBuffer'].forEach(function (method) {
        var read = response[method].bind(response);
        response[method] = function () {
          return read().then(function (data) { setTimeout(invalidate, 0); return data; });
        };
      });
      return response;
    }
  };

  document.addEventListener('DOMContentLoaded', function () {
    new MutationObserver(invalidate).observe(document.body, {childList: true, subtree: true, characterData: true});
    var videos = Array.from(document.querySelectorAll('video'));
    var near = new WeakSet();
    function update(video) {
      var visible = video.getClientRects().length > 0 && !document.hidden;
      if (!visible || !near.has(video)) { video.pause(); return; }
      if (!video.dataset.armed) {
        if (video.dataset.src) video.src = video.dataset.src;
        video.querySelectorAll('source[data-src]').forEach(function (source) { source.src = source.dataset.src; });
        video.dataset.armed = '1';
        video.preload = 'auto';
        video.load();
      }
      video.loop = true;
      video.muted = true;
      video.playsInline = true;
      var play = video.play();
      if (play && play.catch) play.catch(function () {});
    }
    if (window.IntersectionObserver) {
      var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) near.add(entry.target); else near.delete(entry.target);
          update(entry.target);
        });
      }, {rootMargin: '200px 0px'});
      videos.forEach(function (video) { observer.observe(video); });
    } else {
      // Scroll fallback still avoids loading offscreen clips. / 回退方案仍按可见范围加载。
      function scan() {
        videos.forEach(function (video) {
          var box = video.getBoundingClientRect();
          if (box.bottom > -200 && box.top < innerHeight + 200) near.add(video); else near.delete(video);
          update(video);
        });
      }
      window.addEventListener('scroll', scan, {passive: true});
      scan();
    }
    ['wapr-lang-change', 'visibilitychange', 'toggle'].forEach(function (name) {
      document.addEventListener(name, function () { videos.forEach(update); }, true);
    });
  });
})();
