(function () {
  'use strict';

  // ── 壁纸 URL 缓存键 ─────────────────────────────────────────────
  var CACHE_KEY = 'hermes_wallpaper_url';

  // ── 立即发起壁纸 API 请求（不等 requestIdleCallback，减少等待） ──
  var pendingUrl = null;
  var pendingCallbacks = [];

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
          try { localStorage.setItem(CACHE_KEY, url); } catch (_) {}
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
            if (url) try { localStorage.setItem(CACHE_KEY, url); } catch (_) {}
            callback(url);
          })
          .catch(function () { callback(null); });
      });
  }

  // 立即开始获取新壁纸 URL（后续页面加载完成后再应用）
  fetchWallpaper(function (url) { pendingUrl = url; });

  // ── DOM 就绪后的初始化（requestIdleCallback 执行） ─────────────
  function init() {
    var container = document.getElementById('bg-shapes');
    if (!container) return;

    // 壁纸：淡入时间从 2s→500ms，让壁纸更快出现
    var IMG_INTERVAL = 30000;
    var IMG_FADE     = 500;
    var IMG_OPACITY  = 0.65;

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

    // 插在渐变层之后
    var gradEl = document.getElementById('bg-gradient-layer');
    var insertRef = gradEl ? gradEl.nextSibling : container.firstChild;
    container.insertBefore(imgA, insertRef);
    container.insertBefore(imgB, insertRef);

    var currentLayer = imgA;
    var nextLayer    = imgB;
    var isSwitching  = false;

    // 应用壁纸 URL 到图层
    function applyWallpaper(url, instant) {
      if (!url) return;
      var img = new Image();
      img.onload = function () {
        // 如果正在切换中，放弃这次设置（新的请求会覆盖）
        nextLayer.style.backgroundImage = 'url(' + url + ')';
        nextLayer.style.opacity = '0';
        void nextLayer.offsetWidth; // force reflow
        if (instant) {
          // 首次展示：直接显示，不做淡入（已经在缓存中或预加载完成）
          nextLayer.style.opacity = String(IMG_OPACITY);
          currentLayer.style.opacity = '0';
        } else {
          requestAnimationFrame(function () {
            nextLayer.style.opacity = String(IMG_OPACITY);
            currentLayer.style.opacity = '0';
          });
        }
        var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
      };
      img.src = url;
    }

    // 如果已经有缓存的壁纸 URL，预加载它（刷新时浏览器 HTTP 缓存很可能还在）
    var cachedUrl;
    try { cachedUrl = localStorage.getItem(CACHE_KEY); } catch (_) {}
    if (cachedUrl && pendingUrl && cachedUrl !== pendingUrl) {
      // 缓存的和新获取的不一样：先把缓存的显示出来，等待新的
      var tempImg = new Image();
      tempImg.onload = function () {
        applyWallpaper(cachedUrl, true);
        // 等待 pendingUrl 准备好后替换
        var checkPending = setInterval(function () {
          if (pendingUrl && pendingUrl !== cachedUrl) {
            clearInterval(checkPending);
            applyWallpaper(pendingUrl, false);
          }
        }, 100);
      };
      tempImg.src = cachedUrl;
    } else if (cachedUrl) {
      // 显示缓存的壁纸
      applyWallpaper(cachedUrl, true);
    }

    // 轮询等待网络请求的壁纸 ready
    function startFromPending() {
      if (pendingUrl) {
        applyWallpaper(pendingUrl, false);
      } else {
        // 还没回来，等下一轮
        var checkTimer = setInterval(function () {
          if (pendingUrl) {
            clearInterval(checkTimer);
            applyWallpaper(pendingUrl, false);
            // 保存到 localStorage
            try { localStorage.setItem(CACHE_KEY, pendingUrl); } catch (_) {}
          }
        }, 200);
        // 30 秒超时，不再等待
        setTimeout(function () { clearInterval(checkTimer); }, 30000);
      }
    }

    // 如果还没从缓存展示过，直接等网络请求
    if (!cachedUrl) {
      startFromPending();
    }

    // ── 定时轮播 ──
    function preloadNext() {
      // 预加载下一张（等当前显示稳定后再触发下一次 fetch）
      setTimeout(function () {
        fetchWallpaper(function (url) {
          pendingUrl = url;
          switchImage();
        });
      }, IMG_INTERVAL - 5000);
    }

    function switchImage() {
      if (!pendingUrl) return;
      var url = pendingUrl; pendingUrl = null;
      applyWallpaper(url, false);
      try { localStorage.setItem(CACHE_KEY, url); } catch (_) {}
      // 获取下一张
      fetchWallpaper(function (url) { pendingUrl = url; });
      preloadNext();
    }

    // 第一张显示后开始轮播
    preloadNext();

    // ── 浮动形状 ──────────────────────────────────────────────────
    var SHAPES = [
      ['bg-float-square', ''],
      ['bg-float-shape',  'clip-path:polygon(50% 0%,0% 100%,100% 100%)'],
      ['bg-float-shape',  'clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%)'],
      ['bg-float-circle', 'border-radius:50%'],
    ];

    for (var i = 0; i < 20; i++) {
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

    // ── 滚动时暂停背景动画 ──────────────────────────────────────────
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

  // 延迟执行（但只延迟 DOM 操作，网络请求已经在上方开始了）
  if (window.requestIdleCallback) {
    requestIdleCallback(init, { timeout: 500 });
  } else {
    setTimeout(init, 200);
  }
})();
