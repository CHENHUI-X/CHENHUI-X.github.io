(function () {
  // Keyframes for floating shapes
  var style = document.createElement('style');
  style.textContent = [
    '@keyframes bg-float-square {',
    '  0%   { transform:translateY(0) rotate(0deg);    opacity:1; border-radius:0; }',
    '  100% { transform:translateY(-1200px) rotate(720deg); opacity:0; border-radius:50%; }',
    '}',
    '@keyframes bg-float-shape {',
    '  0%   { transform:translateY(0) rotate(0deg);    opacity:1; }',
    '  100% { transform:translateY(-1200px) rotate(720deg); opacity:0; }',
    '}',
    '@keyframes bg-float-circle {',
    '  0%   { transform:translateY(0) rotate(0deg);    opacity:1; border-radius:50%; }',
    '  100% { transform:translateY(-1200px) rotate(720deg); opacity:0; border-radius:50%; }',
    '}',
  ].join('\n');
  document.head.appendChild(style);

  // ── rAF-driven body gradient ─────────────────────────────────────────────
  // Disable transition on body so color changes are instant each frame
  document.body.style.setProperty('transition', 'none', 'important');

  // 亮色：色相跨越粉→橙→紫→蓝→薄荷，约200度跨度，渐变流动明显可见
  var lightColors = [
    [255, 150, 170],  // 玫瑰粉
    [255, 195, 130],  // 桃橙
    [185, 145, 255],  // 薰衣草紫
    [130, 200, 255],  // 天蓝
    [140, 235, 200],  // 薄荷绿
  ];
  // 暗色：同样跨越多个色相，深色调
  var darkColors = [
    [90,  20, 45],   // 深玫瑰
    [85,  48, 12],   // 深橙棕
    [38,  12, 88],   // 深紫
    [12,  38, 88],   // 深蓝
    [12,  65, 52],   // 深绿
  ];

  function lerp(a, b, f) { return Math.round(a + (b - a) * f); }

  var t = 0, last = 0;

  function isDark() { return document.documentElement.classList.contains('dark'); }

  function gradFrame(ts) {
    if (ts - last < 50) { requestAnimationFrame(gradFrame); return; }
    var dt = Math.min((ts - last) / 1000, 0.05);
    last = ts;
    t += dt;

    var colors = isDark() ? darkColors : lightColors;
    var n = colors.length;

    // 顺序两色插值：每5秒从一种颜色过渡到下一种，循环往复
    // 这样颜色变化非常明显，不会被多色混合抵消
    var STEP = 5.0; // 每种颜色停留秒数
    var phase = (t % (STEP * n)) / STEP; // 0 ~ n
    var idx  = Math.floor(phase) % n;
    var next = (idx + 1) % n;
    var f = phase - Math.floor(phase); // 0~1 插值进度
    // smoothstep：让过渡在中间快、两端慢，更自然
    f = f * f * (3 - 2 * f);

    var rgb = [
      lerp(colors[idx][0], colors[next][0], f),
      lerp(colors[idx][1], colors[next][1], f),
      lerp(colors[idx][2], colors[next][2], f),
    ];
    document.body.style.setProperty(
      'background-color',
      'rgb(' + rgb[0] + ',' + rgb[1] + ',' + rgb[2] + ')',
      'important'
    );

    requestAnimationFrame(gradFrame);
  }
  requestAnimationFrame(gradFrame);

  // ── Floating shapes ──────────────────────────────────────────────────────
  var container = document.getElementById('bg-shapes');
  if (!container) return;

  // ── 背景图片切换（夜轻 API，横竖屏自适应，每30秒换一张） ──────────────────
  // 夜轻 API 支持 CORS，横屏 /pc.php，竖屏 /pe.php，返回 JSON 含图片 URL
  // 备用：赫萝 API https://api.horosama.com/random.php?type=pc&format=json
  var IMG_INTERVAL = 30000;
  var IMG_FADE     = 2000;
  var IMG_OPACITY  = 0.65;

  // 创建两个图片层，交替淡入淡出
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
  container.insertBefore(imgA, container.firstChild);
  container.insertBefore(imgB, container.firstChild);

  var currentLayer = imgA;
  var nextLayer    = imgB;

  // 根据屏幕方向选端点，返回 JSON 含 acgurl 字段
  function fetchWallpaper(callback) {
    var landscape = window.innerWidth >= window.innerHeight;
    var primary  = landscape
      ? 'https://api.yppp.net/pc.php?return=json'
      : 'https://api.yppp.net/pe.php?return=json';
    var fallback = 'https://api.horosama.com/random.php?type=pc&format=json';

    var controller = new AbortController();
    var timer = setTimeout(function() { controller.abort(); }, 8000);

    fetch(primary, { signal: controller.signal })
      .then(function(r) { return r.json(); })
      .then(function(d) {
        clearTimeout(timer);
        // 夜轻 API 返回 { acgurl: "..." }
        var url = d.acgurl || d.url;
        if (url) { callback(url); } else { throw new Error('no url'); }
      })
      .catch(function() {
        clearTimeout(timer);
        // 主 API 失败，尝试备用
        fetch(fallback)
          .then(function(r) { return r.json(); })
          .then(function(d) { callback(d.url || null); })
          .catch(function() { callback(null); });
      });
  }

  // 预加载图片，加载完成后执行淡入淡出切换；isSwitching 防止并发竞态
  var isSwitching = false;
  function doSwitch(url) {
    if (isSwitching) return;
    isSwitching = true;
    var img = new Image();
    img.onload = function() {
      nextLayer.style.backgroundImage = 'url(' + url + ')';
      nextLayer.style.opacity = '0';
      void nextLayer.offsetWidth;
      requestAnimationFrame(function() {
        nextLayer.style.opacity = String(IMG_OPACITY);
        currentLayer.style.opacity = '0';
        setTimeout(function() {
          var tmp = currentLayer; currentLayer = nextLayer; nextLayer = tmp;
          isSwitching = false;
        }, IMG_FADE);
      });
    };
    img.onerror = function() { isSwitching = false; /* 图片加载失败静默跳过 */ };
    img.src = url;
  }

  var nextUrl = null;

  function switchImage() {
    if (nextUrl) {
      var url = nextUrl; nextUrl = null;
      doSwitch(url);
      // 后台预拉下一张
      fetchWallpaper(function(u) { nextUrl = u; });
    } else {
      fetchWallpaper(function(url) {
        if (!url) return;
        doSwitch(url);
        fetchWallpaper(function(u) { nextUrl = u; });
      });
    }
  }

  // 页面加载立即获取第一张，之后每30秒切换
  switchImage();
  setInterval(switchImage, IMG_INTERVAL);

  var SHAPES = [
    ['bg-float-square', ''],
    ['bg-float-shape',  'clip-path:polygon(50% 0%,0% 100%,100% 100%)'],
    ['bg-float-shape',  'clip-path:polygon(50% 0%,100% 50%,50% 100%,0% 50%)'],
    ['bg-float-circle', 'border-radius:50%'],
  ];

  for (var i = 0; i < 50; i++) {
    var el       = document.createElement('div');
    var size     = Math.round(30 + Math.random() * 200);
    var left     = Math.round(Math.random() * 100);
    var delay    = (Math.random() * 25).toFixed(1);
    var duration = (10 + Math.random() * 35).toFixed(1);
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
    ];
    if (sh[1]) css.push(sh[1]);
    el.style.cssText = css.join(';');
    container.appendChild(el);
  }
})();
