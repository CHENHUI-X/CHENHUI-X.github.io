(function () {
  'use strict';

  // 防止 Swup SPA 导航重新执行本脚本
  if (window.__bgAnimInited) return;
  window.__bgAnimInited = true;
  console.log('[bg-anim] init ' + Date.now());

  // ── 壁纸缓存键 ──────────────────────────────────────────────
  var CACHE_KEY  = 'hermes_wallpaper_url';
  var TIME_KEY   = 'hermes_wallpaper_time';

  // ── 配置 ────────────────────────────────────────────────────
  var IMG_INTERVAL = 60 * 1000;     // 1 分钟切换一次
  var PRELOAD_LEAD = 5 * 1000;      // 提前 5 秒预加载下一张
  var IMG_FADE     = 2500;           // 淡入淡出时长（ms）
  var IMG_OPACITY  = 0.65;

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

    // 创建两个壁纸图层，用于交叉淡入淡出
    var imgA = document.createElement('div');
    var imgB = document.createElement('div');
    var baseImgStyle = [
      'position:absolute', 'inset:0',
      'background-size:cover',
      'background-position:center center',
      'transition:opacity ' + IMG_FADE + 'ms ease',
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

// 移动端 Wallpaper 防闪烁：图片加载后重置 background-size 确保 cover 生效
function setBgImage(el, url) {
  el.style.backgroundImage = 'url(' + url + ')';
  if (window.innerWidth < 1024) {
    void el.offsetWidth;
    requestAnimationFrame(function () {
      el.style.backgroundSize = 'cover';
      void el.offsetWidth;
    });
  }
}

    // ── 应用壁纸到目标图层（img 预加载完成后触发） ──────────
    function applyWallpaper(url, instant) {
      if (!url) return;
      var img = new Image();
      img.onload = function () {
        setBgImage(nextLayer, url);
        nextLayer.style.opacity = '0';
        void nextLayer.offsetWidth; // force reflow → 确保浏览器记录了 opacity:0
        if (instant) {
          // 首次展示：让缓存壁纸直接出现（无等待）
          nextLayer.style.opacity = String(IMG_OPACITY);
          currentLayer.style.opacity = '0';
          var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
        } else {
          // 定时切换：触发 CSS transition 交叉淡入淡出
          requestAnimationFrame(function () {
            nextLayer.style.opacity = String(IMG_OPACITY);
            currentLayer.style.opacity = '0';
            var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
          });
        }
      };
      img.src = url;
    }

    // ── 预加载下一张壁纸（提前下载图片，到点直接淡入） ────────
    var preloadedUrl = null;   // 已经加载完成的下一张 URL
    var preloading   = false;  // 是否正在预加载中

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

    // ── 切换壁纸（使用预加载的图片，无等待） ────────────────
    function switchImage() {
      if (preloadedUrl) {
        applyWallpaper(preloadedUrl, false);
        preloadedUrl = null;
        // 立即为下一次切换预加载
        preloadNext(function () {});
      } else {
        // 极端情况：预加载没完成，直接 fetch + 加载后切换
        fetchWallpaper(function (url) {
          if (url) applyWallpaper(url, false);
        });
      }
    }

    // ── 调度器 ──────────────────────────────────────────────
    // 每次切换后，在当前周期的末尾提前预加载下一张
    function schedulePreload() {
      setTimeout(function () {
        preloadNext(function () {});
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
      applyWallpaper(cached.url, true);  // 有缓存：秒出，绝对不触发新请求
    } else {
      // 完全没有缓存（首次访问）：立即 fetch 一张
      fetchWallpaper(function (url) {
        if (url) applyWallpaper(url, false);
      });
    }

    // 启动定时调度：只有定时器到点才会触发壁纸切换
    preloadNext(function () {});        // 为第一次切换提前下载
    schedulePreload();                  // 为以后每次切换提前下载
    scheduleNext();                     // 定时器到点才切换

    // ── 浮动形状 ──────────────────────────────────────────
    var SHAPES = [
      ['bg-float-square', 'border-radius:25%'],           // 大圆角方形（squircle）
      ['bg-float-shape',  'border-radius:40% 60% 50% 30%'], // 软性气泡A
      ['bg-float-shape',  'border-radius:55% 30% 45% 60%'], // 软性气泡B
      ['bg-float-circle', 'border-radius:50%'],           // 圆形
    ];

    // 移动端/平板减少形状数，提升流畅度
    var shapeCount = window.innerWidth < 1024 ? 8 : 20;

    for (var i = 0; i < shapeCount; i++) {
      var el       = document.createElement('div');
      var size     = Math.round(30 + Math.random() * 200);
      var left     = Math.round(Math.random() * 100);
      var delay    = (Math.random() * 25).toFixed(1);
      var duration = (15 + Math.random() * 20).toFixed(1);
      var hue      = Math.round(315 + Math.random() * 60);
      var sh       = SHAPES[Math.floor(Math.random() * SHAPES.length)];
      var color    = 'hsla(' + hue + ',65%,72%,0.12)';

      var css = [
        'position:absolute',
        'bottom:-' + size + 'px',
        'left:' + left + '%',
        'width:' + size + 'px',
        'height:' + size + 'px',
        'background:' + color,
        'animation:' + sh[0] + ' ' + duration + 's linear ' + delay + 's infinite',
        'will-change:transform',
        'transform:translateZ(0)',
      ].concat(sh[0] ? ['class:' + sh[0]] : []);
      if (sh[1]) css.push(sh[1]);
      el.style.cssText = css.join(';');
      container.appendChild(el);
    }

    // ── 滚动时暂停背景动画 ──────────────────────────────────
    var scrollTimer = null;
    function pauseOnScroll() {
      container.classList.add('bg-scroll-active');
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(function () {
        container.classList.remove('bg-scroll-active');
      }, 200);
    }
    window.addEventListener('scroll', pauseOnScroll, { passive: true });
  }

  // 延迟 init，不影响首屏渲染
  if (window.requestIdleCallback) {
    requestIdleCallback(init, { timeout: 500 });
  } else {
    setTimeout(init, 200);
  }
})();
