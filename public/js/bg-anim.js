(function () {
  'use strict';

  // 防止 Swup SPA 导航重新执行本脚本
  if (window.__bgAnimInited) return;
  window.__bgAnimInited = true;

  // ── 壁纸缓存键 ──────────────────────────────────────────────
  var CACHE_KEY  = 'hermes_wallpaper_url';
  var TIME_KEY   = 'hermes_wallpaper_time';

  // ── 配置 ────────────────────────────────────────────────────
  var IMG_INTERVAL = 60 * 1000;     // 1 分钟切换一次
  var PRELOAD_LEAD = 5 * 1000;      // 提前 5 秒预加载下一张
  var IMG_FADE     = 2000;          // 淡入淡出时长（ms）
  var IMG_OPACITY  = 0.36;          // 克制背景透明度，保持公式与代码块清晰度

  // ── 缓存读写 ─────────────────────────────────────────────
  function getCached() {
    var url, time;
    try { url  = localStorage.getItem(CACHE_KEY); } catch (_) {}
    try { time = parseInt(localStorage.getItem(TIME_KEY), 10) || 0; } catch (_) { time = 0; }
    return { url: url, time: time };
  }

  function saveCache(url) {
    try { localStorage.setItem(CACHE_KEY, url); } catch (_) {}
    try { localStorage.setItem(TIME_KEY, String(Date.now())); } catch (_) {}
  }

  // ── 壁纸 API 请求 ─────────────────────────────────────────
  function fetchWallpaper(callback) {
    var landscape = window.innerWidth >= window.innerHeight;
    var primary = landscape
      ? 'https://api.yppp.net/pc.php?return=json'
      : 'https://api.yppp.net/pe.php?return=json';
    var fallback = 'https://api.horosama.com/random.php?type=pc&format=json';

    var controller = new AbortController();
    var timer = setTimeout(function () { controller.abort(); }, 8000);

    fetch(primary, { signal: controller.signal })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        clearTimeout(timer);
        var url = d.acgurl || d.url;
        if (url) {
          saveCache(url);
          callback(url);
        } else {
          throw new Error('no url');
        }
      })
      .catch(function () {
        clearTimeout(timer);
        fetch(fallback)
          .then(function (r) { return r.json(); })
          .then(function (d) {
            var url = d.url || null;
            if (url) saveCache(url);
            callback(url);
          })
          .catch(function () { callback(null); });
      });
  }

  // ── DOM 初始化 ─────────────────────────────────────────────
  function init() {
    var container = document.getElementById('bg-shapes');
    if (!container) return;

    // 创建两个壁纸图层，置于 #bg-gradient-layer 之后，用于纯 GPU 交叉淡入淡出
    var imgA = document.createElement('div');
    var imgB = document.createElement('div');
    var baseImgStyle = [
      'position:absolute', 'inset:0',
      'background-size:cover',
      'background-position:center center',
      'transition:opacity ' + IMG_FADE + 'ms ease',
      'will-change:opacity',
      'opacity:0',
    ].join(';');
    imgA.style.cssText = baseImgStyle;
    imgB.style.cssText = baseImgStyle;

    var gradEl = document.getElementById('bg-gradient-layer');
    var insertRef = gradEl ? gradEl.nextSibling : container.firstChild;
    container.insertBefore(imgA, insertRef);
    container.insertBefore(imgB, insertRef);

    var currentLayer = imgA;
    var nextLayer    = imgB;

    // ── 应用壁纸到目标图层（无强制同步回流） ──────────────────
    function applyWallpaper(url, instant) {
      if (!url) return;
      var img = new Image();
      img.onload = function () {
        nextLayer.style.backgroundImage = 'url(' + url + ')';
        nextLayer.style.backgroundSize = 'cover';
        nextLayer.style.opacity = '0';
        if (instant) {
          requestAnimationFrame(function () {
            nextLayer.style.opacity = String(IMG_OPACITY);
            currentLayer.style.opacity = '0';
            var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
          });
        } else {
          requestAnimationFrame(function () {
            requestAnimationFrame(function () {
              nextLayer.style.opacity = String(IMG_OPACITY);
              currentLayer.style.opacity = '0';
              var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
            });
          });
        }
      };
      img.src = url;
    }

    // ── 预加载下一张壁纸（提前下载图片，到点直接淡入） ────────
    var preloadedUrl = null;
    var preloading   = false;

    function preloadNext(callback) {
      if (preloading) return;
      preloading = true;
      fetchWallpaper(function (url) {
        preloading = false;
        if (!url) { callback && callback(null); return; }
        var img = new Image();
        img.onload = function () {
          preloadedUrl = url;
          callback && callback(url);
        };
        img.onerror = function () { callback && callback(null); };
        img.src = url;
      });
    }

    // ── 切换壁纸（标签页隐藏时跳过，回到前台再继续） ────────
    function switchImage() {
      if (document.hidden) return;
      if (preloadedUrl) {
        applyWallpaper(preloadedUrl, false);
        preloadedUrl = null;
        preloadNext(function () {});
      } else {
        fetchWallpaper(function (url) {
          if (url) applyWallpaper(url, false);
        });
      }
    }

    // ── 调度器 ──────────────────────────────────────────────
    function schedulePreload() {
      setTimeout(function () {
        if (!document.hidden) preloadNext(function () {});
      }, Math.max(0, IMG_INTERVAL - PRELOAD_LEAD));
    }

    function scheduleNext() {
      setTimeout(function () {
        switchImage();
        schedulePreload();
        scheduleNext();
      }, IMG_INTERVAL);
    }

    // ── 首次展示 ────────────────────────────────────────────
    var cached = getCached();
    if (cached.url) {
      applyWallpaper(cached.url, true);
    } else {
      fetchWallpaper(function (url) {
        if (url) applyWallpaper(url, false);
      });
    }

    preloadNext(function () {});
    schedulePreload();
    scheduleNext();
  }

  // 延迟 init，不影响首屏渲染
  if (window.requestIdleCallback) {
    requestIdleCallback(init, { timeout: 500 });
  } else {
    setTimeout(init, 200);
  }
})();
