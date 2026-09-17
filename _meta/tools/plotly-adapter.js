/* _meta/tools/plotly-adapter.js —— kind="plotly" 的 Plotly 后端（opt-in 的独立渲染器）

   为什么单独一个文件：Plotly 全量包 4.85 MB（3D 裁剪包 1.69 MB），不能塞进 widgets.js 让**每个**
   组件页都背。所以只有 spec.kind = "plotly" 的页面才内联它（由 make_widget.py 按 spec.plotly.bundle 选，
   见 _meta/tools/vendor/README.md）。本文件只在那些页面里出现。

   加载顺序（make_widget.py 生成的页面，脚本都在 body 末尾）：
    <style>widgets.css</style> → <script>vendor/plotly*.min.js</script> → <script>widgets.js</script>
     → <script>plotly-adapter.js</script> → boot
   本文件在 widgets.js 之后、boot 之前执行，所以这里只做一件事：
   WG.registerKind('plotly', drawPlotly) —— widgets.js 的 drawFigure 在内置分派之后、
   报"未知 kind"之前会查这张外部注册表，调用时给的 ctx 与内置渲染器完全一致：{it, spec, narrow, W}。

   用到的运行时接口（只依赖这些，不碰 widgets.js 的闭包）：
     it.el.figure          图区容器（每次重绘前运行时已 clear 掉）
     it.err(msg)           行内错误（自动加 "[错误] " 前缀）
     it.ev(expr, extra, where)  用运行时的表达式引擎求值一次
     it.setCaption([…])    图注：普通字符串内嵌也可见；WG.det('…') 标记的口径行只在完整版显示
     it.vars               写 it.vars.__plotly__ = {…} 给 readouts 引用
     it.theme              组件调色板（读的是组件页的 CSS 变量）
     it.figureWidth()      图区当前可用宽度（px）
     it.layout()           重新量一次并同步内嵌宿主 iframe 的高度

   spec 形状（除通用字段 title / uid / vars / controls / readouts / notes / theme 外）：
     "kind": "plotly",
     "plotly": {
       "data":   [ …Plotly trace 对象… ],                            // 必填、非空数组
      "layout": { …Plotly layout… },                                // 可选
      "config": { "displayModeBar": false, "scrollZoom": true },     // 可选（displayModeBar 由作者定）
      "height": 420,                                                  // 可选：覆盖按宽度算出的高度
      "bundle": "gl3d"                                                // 可选："full"（默认）|"gl3d"（只含 3D，页面小 3.2 MB）

   最小可用例子（一屏能跑的三维曲面 + 一个控件；表达式里的 a、n 来自 controls/vars）：
     {
       "schema": "widget/v1", "kind": "plotly", "uid": "",
       "title": "Plotly 三维曲面：z = a·x² + y²",
       "vars": { "n": 25 },
       "controls": [{ "key": "a", "type": "slider", "label": "曲率 a",
                      "min": 0.2, "max": 3, "step": 0.1, "value": 1 }],
       "plotly": {
     |      "data": [
     |        { "type": "surface",
     |          "x": {"by": "-2 + 4 * j / (n - 1)", "n": "=n"},
     |          "y": {"by": "-2 + 4 * i / (n - 1)", "n": "=n"},
     |          "z": {"by": "a * x * x + y * y", "rows": "=n", "cols": "=n"},
     |          "colorscale": "Blues" }
     |      ],
     |      "layout": { "scene": { "aspectmode": "cube" } },
     |      "config": { "displayModeBar": false }
     |    },
     |    "readouts": [{ "label": "网格点数", "expr": "__plotly__.points", "fmt": "0" }]
     |  }
     （上面是**示意**：左边的 `|` 是注释符，去掉它（与本行的缩进）就是合法 JSON。
      骨架的真实样子见 `python3 _meta/tools/make_widget.py --spec plotly`；
      仓库里已登记的实例是 交互组件示例-Plotly三维.md 那个损失曲面。）

  数据怎么写（两条路，spec 里可混用）：
   1) 直接写数组：Plotly 要什么就写什么（数字数组、[[x, y], …]、或 z 的二维数组）。
   2) **生成器**：`{"by": "<表达式>", "n": 25}`（一维）或
      `{"by": "<表达式>", "rows": 25, "cols": 25}`（二维，得到 rows×cols 的嵌套数组），
      本文件在 JS 侧逐元素调 it.ev 求值 —— 所以"跟着控件变的数据"不用手写几百个数字：
        · 一维生成器的作用域：i（0 基下标）、n（长度）、x = i
        · 二维生成器的作用域：i（行）、j（列）、n（行数）、m（列数）、x = i、y = j
        · n / rows / cols 可以是数字，也可以写 "=表达式"（求值一次后取整）
        · 生成器还可以带 "vars": {"名字": "表达式"}，在同一作用域里先求出来再算 by ——
          用来把"范围常量只写一遍"（放在 spec.vars）：x/y/z 三个生成器共用 a0/a1/b0/b1 这类常量
        · 上限：一维 ≤ 300 个元素；二维 ≤ 40000 格（超了报 [错误]，不静默截断）
        · 某一位求值不出有限数字时**跳过该位**（二维填 NaN）并报一条 [错误]，
          不伪造数据；为什么不能用表达式里的循环：见下面"表达式约定"。

  表达式约定（`=` 前缀）：
   · 用**同一套** AST 表达式引擎求值 —— 没有 eval、没有 new Function，因此也**没有对象字面量**、
     没有 for/while、没有 Array.from/map；只有算术、比较、?: 、数组字面量 [1, 2, 3] 与白名单函数。
     （这就是生成器存在的理由：要"按控件生成一列数"，靠表达式里的循环是做不到的。）
   · plotly.data / plotly.layout 里**以 = 开头的字符串**在渲染时交给 it.ev() 求值，
    作用域就是通用表达式的作用域：vars 的键、controls 的 key，以及 rand/randn/phi/quantile/
    sum/mean/sd/clamp/min/max/abs/exp/log/sqrt/pow/sin/cos/tan/floor/round/PI/E/ifelse/fmt
    （整体求值一次，不是逐点作用域——所以 x / i / item 在这里没有意义）。
    求值结果直接写回同一位置，于是拖控件就能改数据、改颜色、改坐标轴范围。
      "=a*x"            → 求值成数字；也可以返回数组（{"by":…,"n":…} 生成器更长更好读）
      "scatter"         → 不以 = 开头：原样传给 Plotly，不当表达式
    结果**不再二次求值**：单引号 / 双引号字符串字面量由运行时解析一次就够了。
    想拿到一个以 = 开头的字面量字符串，写 '="'=x'"'（外层是表达式，里层是字面量）。
    求值失败不会中断整页：it.ev 会把 [错误] 写进行内错误区（where 指出是哪一条路径），
    该位置保留 NaN，不伪造数据。

   渲染口径：
   - 白底默认。layout.paper_bgcolor / plot_bgcolor / font.color / 轴网格色 / hoverlabel 默认按
     浅色主题补；只有 spec.theme === "system" 且系统确实是深色时，才读 it.theme 用深色值。
   - **高度必须显式给**：内嵌 iframe 是 height:auto，Plotly 不会自己长高（会画成一条）。
     默认 clamp(figureWidth() × 0.72, 260, 520) px，spec.plotly.height 可覆盖。
     宽度交给 Plotly 按容器算（不写 layout.width）；窗口变宽由 widgets.js 的 ResizeObserver/resize 重绘接管。
   - 标题归页面 DOM：layout.title 一律丢掉，避免一页两个标题。
   - config 强制 `responsive:false`：Plotly 自己的 window resize 监听会在容器改宽、旧图暂时摘下时把高度清成 0；运行时已统一监听窗口/容器并以 Plotly.react 传入新宽度和显式高度，避免该竞态。其余 config（含 displayModeBar）按 spec.plotly.config。
    cfg.responsive = false;     // 宽度/高度由 widgets.js 的 ResizeObserver + Wg.layout/redraw 统一接管，避免 Plotly resize 竞态
   - 画之前就发布 it.vars.__plotly__ = {traces, points, height, width, mode}（冻结），
     这样同一帧里紧接着求值的 readouts 就能引用它；这些数是**按 spec 算出来的**，不是从像素反解。
   - 重绘幂等：同一个 figure 只建一次图。运行时每次重绘都会 clear(figure) 把上一帧的图摘下来，
     这里先把它挂回图区、再走 Plotly.react（react 需要原 DOM 还在，否则会更新到已摘除的节点上、
     图面静默变空）；容器真的被换掉时先 Plotly.purge 再重建，避免 WebGL 上下文越积越多。
   - 零外部引用：不发请求、不 import、不引 CDN——库是页面内联的。

   已知限制（不动 CSP，如实报错；细节见 vendor/README.md）：
   - 需要 blob: worker 的特性（parcoords）在 `default-src 'none'` 下不可用；
   - 地图类 trace 要联网取瓦片，`connect-src 'none'` 下不可用；
   - 需要 WebGL 的 3D trace 在宿主没有 WebGL 时画不出来（会走 [错误] 分支）；
     bundle: "gl3d" 里没有 2D trace（scatter/bar/pie…），写错了它不报错但画不出来——
     make_widget.py 的校验阶段会就这件事给 [提醒]；
   - figure 容器被整块替换（组件实例销毁重建）时，旧图的 WebGL 上下文交给浏览器回收：
     运行时没有销毁钩子，本文件无法主动 purge 那一刻的图。
*/
(function (global) {
  'use strict';

  var WG = global.WG;
  /* 运行时不在（本页没内联 widgets.js / 被 CSP 挡了）：什么都不做，别在控制台抛错 */
  if (!WG || typeof WG.registerKind !== 'function') return;

  var KIND = 'plotly';
  var H_RATIO = 0.72, H_MIN = 260, H_MAX = 520, H_FLOOR = 120, H_CEIL = 2000;
  var LIGHT = {
    bg: '#ffffff', panel: '#f8fafc', text: '#202938', dim: '#475569',
    border: '#cbd5e1', grid: '#e2e8f0'
  };
  /* WebGL 三维 trace（除它们之外的 type 都按二维算，包括地图类） */
  var TYPES_3D = ['scatter3d', 'surface', 'mesh3d', 'cone', 'streamtube', 'isosurface', 'volume'];
  /* 点数口径用的数组字段：x/y/z 取最大者，避免把同一批点的坐标重复计两遍 */
  var POINT_KEYS = ['x', 'y', 'z', 'values', 'labels', 'open', 'high', 'low', 'close',
    'r', 'theta', 'lat', 'lon'];

  /* ------------------------------------------------------------------ */
  /* 小工具（独立文件借不到 widgets.js 的闭包，只能自带这几行）           */
  /* ------------------------------------------------------------------ */
  function isPlainObj(v) { return !!v && typeof v === 'object' && !Array.isArray(v); }
  function msgOf(e) { return String((e && e.message) || e); }
  function has(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }

  /* 深拷贝：数据、layout 都可能被表达式就地改写，绝不改 author 的 spec 对象本身。
     只有普通对象与数组递归，其余（数字/字符串/布尔/null/函数等）原样带过去。 */
  function deepClone(v) {
    var out, i, k;
    if (Array.isArray(v)) {
      out = [];
      for (i = 0; i < v.length; i++) out.push(deepClone(v[i]));
      return out;
    }
    if (isPlainObj(v)) {
      out = {};
      for (k in v) if (has(v, k)) out[k] = deepClone(v[k]);
      return out;
    }
    return v;
  }

  function clampNum(v, lo, hi, dflt) {
    var x = (v === undefined || v === null || v === '') ? NaN : Number(v);
    if (!isFinite(x)) x = dflt;
    return Math.min(hi, Math.max(lo, x));
  }

  function describe(v) {
    if (v === undefined) return 'undefined';
    if (v === null) return 'null';
    if (Array.isArray(v)) return v.length ? (v.length + ' 个元素') : '空数组';
    return typeof v;
  }
  /* ------------------------------------------------------------------ */
  /* 表达式：以 = 开头的字符串按运行时引擎求值一次                        */
  /*   · 求值用的就是 widgets.js 自己那套 AST 引擎（没有 eval/new Function，            */
  /*     也就没有对象字面量、没有 for/while、没有 Array.from —— 只能写算术与白名单函数）。 */
  /*   · 想"按控件生成一列数"就用下面的生成器（{by, n} / {by, rows, cols}），          */
  /*     它是本文件在 JS 侧按元素调 it.ev 循环出来的，不走表达式里的循环。              */
  /* ------------------------------------------------------------------ */
  var GEN_MAX_N = 300;          // 一维生成器的元素个数上限（300 个点的曲线足够用）
  var GEN_MAX_CELLS = 40000;    // 二维生成器的格子数上限（与 heat 的 HEAT_CELLS_MAX 一致）
  var GEN_DEFAULT_N = 25;       // 省略 n 时的默认长度（与 --spec plotly 的骨架一致）

  /* n / rows / cols 允许写成数字，也允许写成 "=表达式"（求值一次后再取整夹紧） */
  function genSize(it, v, dflt, path) {
    var raw = v;
    if (typeof v === 'string' && v.charAt(0) === '=') raw = it.ev(v.slice(1), null, path);
    var num = Number(raw);
    if (raw === undefined || raw === null || raw === '') return dflt;
    if (!isFinite(num)) { it.err(path + ' 不是有限数字（' + describe(raw) + '）'); return dflt; }
    return Math.floor(num);
  }

  /* 一维：{by: "<表达式：可用 i 与 n>", n: 25} → [v0, …, v(n-1)]
     二维：{by: "<表达式：可用 i(行) 、j(列) 与 n(行数) 、m(列数)>", rows: 25, cols: 25} → [[…], …] */
  function makeGen(it, node, path) {
    var by = node.by;
    var two = (node.rows !== undefined || node.cols !== undefined);
    var n = two ? genSize(it, node.rows, GEN_DEFAULT_N, path + '.rows')
                : genSize(it, node.n, GEN_DEFAULT_N, path + '.n');
    var m = two ? genSize(it, node.cols, GEN_DEFAULT_N, path + '.cols') : 0;
    if (n < 1) { it.err(path + '：长度必须是 ≥ 1 的整数（当前 ' + n + '）'); return []; }
    if (two && m < 1) { it.err(path + '.cols 必须是 ≥ 1 的整数（当前 ' + m + '）'); return []; }
    if (!two && n > GEN_MAX_N) {
      it.err(path + '：一维生成器最多 ' + GEN_MAX_N + ' 个元素（当前 ' + n + '）——'
        + '再长就请直接把数组写进 spec（生成器是给"跟着控件变的曲线"用的）');
      return [];
    }
    if (two && n * m > GEN_MAX_CELLS) {
      it.err(path + '：二维生成器最多 ' + GEN_MAX_CELLS + ' 格（当前 ' + n + '×' + m + '）');
      return [];
    }
    var i, j, out, row, r, bad = 0, q, t, sc, kk;
    /* 生成器可以自己声明逐元素名字：{"by": "…", "vars": {"a": "a0 + (a1 - a0) * j / (n - 1)"}}
       作用是在**同一个作用域**里先求出这些名字，by 里就能直接用 a —— 这样"范围常量"只写一遍
       （放在 spec.vars 里），x / y / z 三个生成器不会各写一份区间而悄悄跑偏。 */
    var extraNames = isPlainObj(node.vars) ? Object.keys(node.vars) : [];
    function scopeOf(base) {
      var s = base;
      for (q = 0; q < extraNames.length; q++) {
        t = extraNames[q];
        if (s === base) { s = {}; for (kk in base) if (has(base, kk)) s[kk] = base[kk]; }
        s[t] = it.ev(node.vars[t], base, path + '.vars.' + t);
      }
      return s;
    }
    if (!two) {
      out = [];
      for (i = 0; i < n; i++) {
        r = it.ev(by, scopeOf({ i: i, n: n, x: i }), path + '[' + i + ']');
        if (typeof r !== 'number' || !isFinite(r)) { bad++; if (bad === 1) it.err(path + '[' + i + ']：'
          + '生成结果不是有限数字（' + describe(r) + '）——生成器的 by 必须返回数字'); continue; }
        out.push(r);
      }
      return out;                                  // 失败的位置**跳过**而不是塞 NaN：曲线短一截也比谎报数据好
    }
    out = [];
    for (i = 0; i < n; i++) {
      row = [];
      for (j = 0; j < m; j++) {
        r = it.ev(by, scopeOf({ i: i, j: j, n: n, m: m, x: i, y: j }), path + '[' + i + '][' + j + ']');
        if (typeof r !== 'number' || !isFinite(r)) {
          bad++;
          if (bad === 1) it.err(path + '[' + i + '][' + j + ']：生成结果不是有限数字（' + describe(r)
            + '）——二维生成器的 by 必须返回数字');
          r = NaN;
        }
        row.push(r);
      }
      out.push(row);
    }
    return out;
  }

  function evalTree(it, node, path) {
    var out, i, k;
    if (typeof node === 'string') {
      if (node.charAt(0) !== '=') return node;          // 不以 = 开头：原样交给 Plotly
      return it.ev(node.slice(1), null, path);          // 去掉前导 =；结果不再二次求值
    }
    if (Array.isArray(node)) {
      out = [];
      for (i = 0; i < node.length; i++) out.push(evalTree(it, node[i], path + '[' + i + ']'));
      return out;
    }
    if (isPlainObj(node)) {
      if (typeof node.by === 'string') return makeGen(it, node, path);   // 生成器，见 makeGen 注释
      out = {};
      for (k in node) if (has(node, k)) out[k] = evalTree(it, node[k], path ? (path + '.' + k) : k);
      return out;
    }
    return node;
  }

  /* ------------------------------------------------------------------ */
  /* 主题：默认浅色；只有 spec.theme="system" 且系统深色才跟随组件主题     */
  /* ------------------------------------------------------------------ */
  function prefersDark() {
    try {
      return !!(global.matchMedia && global.matchMedia('(prefers-color-scheme: dark)').matches);
    } catch (e) { return false; }
  }

  function themeColors(it) {
    var c = it.theme && it.theme.c;
    return Array.isArray(c) ? c.slice() : [];
  }

  /* it.theme 读的是组件页的 CSS 变量：浅色页里是白底深字，data-theme="system" + 系统深色时
     才是深色值。这里显式判一次系统偏好，是为了不依赖"宿主一定定义了变量"这件事。 */
  function palette(it, spec) {
    var c = themeColors(it);
    if (spec.theme !== 'system' || !prefersDark()) {
      return { dark: false, bg: LIGHT.bg, panel: LIGHT.panel, text: LIGHT.text, dim: LIGHT.dim,
        border: LIGHT.border, grid: LIGHT.grid, c: c };
    }
    var t = it.theme || {};
    return {
      dark: true,
      bg: t.bg || '#0c0e13', panel: t.card || t.panel || '#12151d', text: t.text || '#e6e8ee',
      dim: t.dim || '#a6afc0', border: t.border || '#262b38', grid: t.border || '#262b38', c: c
    };
  }

  /* ------------------------------------------------------------------ */
  /* layout / config 口径                                                */
  /* ------------------------------------------------------------------ */
  function axisDefaults(ax, pal) {
    var a = isPlainObj(ax) ? ax : {};
    if (!a.gridcolor) a.gridcolor = pal.grid;
    if (!a.linecolor) a.linecolor = pal.border;
    if (!a.zerolinecolor) a.zerolinecolor = pal.border;
    if (isPlainObj(a.tickfont)) { if (!a.tickfont.color) a.tickfont.color = pal.dim; }
    else a.tickfont = { color: pal.dim };
    return a;
  }

  function buildLayout(it, spec, p, W) {
    var pal = palette(it, spec);
    var h = Math.max(H_MIN, Math.min(H_MAX, Math.round(W * H_RATIO)));
    var height = clampNum(p.height, H_FLOOR, H_CEIL, h);
    var lay = deepClone(isPlainObj(p.layout) ? p.layout : {});
    delete lay.title;                                   // 标题归页面 DOM，不要两个标题
    /* 显式高度：height:auto 的 iframe 里 Plotly 不会自己长高 */
    lay.height = clampNum(lay.height, H_FLOOR, H_CEIL, height);
    if (!lay.paper_bgcolor) lay.paper_bgcolor = pal.bg;
    if (!lay.plot_bgcolor) lay.plot_bgcolor = pal.panel;
    var font = isPlainObj(lay.font) ? lay.font : {};
    if (!font.color) font.color = pal.text;
    lay.font = font;
    var hov = isPlainObj(lay.hoverlabel) ? lay.hoverlabel : {};
    var hovFont = isPlainObj(hov.font) ? hov.font : {};
    if (!hovFont.color) hovFont.color = pal.text;
    hov.font = hovFont;
    if (!hov.bgcolor) hov.bgcolor = pal.bg;
    if (!hov.bordercolor) hov.bordercolor = pal.border;
    lay.hoverlabel = hov;
    if (!Array.isArray(lay.colorway) && pal.c.length) lay.colorway = pal.c;   // 默认用组件调色板
    var mg = isPlainObj(lay.margin) ? lay.margin : {};
    if (mg.l === undefined) mg.l = 52;
    if (mg.r === undefined) mg.r = 14;
    if (mg.t === undefined) mg.t = 12;
    if (mg.b === undefined) mg.b = 40;
    lay.margin = mg;
    lay.xaxis = axisDefaults(lay.xaxis, pal);
    lay.yaxis = axisDefaults(lay.yaxis, pal);
    /* 三维轴只在作者写了 scene 时才补默认色：没写 scene 就别凭空塞一个对象进去 */
    if (isPlainObj(lay.scene)) {
      lay.scene.xaxis = axisDefaults(lay.scene.xaxis, pal);
      lay.scene.yaxis = axisDefaults(lay.scene.yaxis, pal);
      lay.scene.zaxis = axisDefaults(lay.scene.zaxis, pal);
    }
    return lay;
  }

  function buildConfig(p) {
    var cfg = deepClone(isPlainObj(p.config) ? p.config : {});
    cfg.responsive = false;     // 宽度/高度由 widgets.js 的 ResizeObserver + Wg.layout/redraw 统一接管，避免 Plotly resize 竞态
    return cfg;
  }

  /* ------------------------------------------------------------------ */
  /* 读数：traces / points / height / width / mode                       */
  /* ------------------------------------------------------------------ */
  function arrLen(v) {
    var inner = 0, i;
    if (!Array.isArray(v)) return 0;
    if (v.length && Array.isArray(v[0])) {              // z 是网格：rows × cols
      for (i = 0; i < v.length; i++) if (Array.isArray(v[i])) inner = Math.max(inner, v[i].length);
      return v.length * inner;
    }
    return v.length;
  }

  function tracePoints(t) {
    var max = 0, i;
    if (!isPlainObj(t)) return 0;
    for (i = 0; i < POINT_KEYS.length; i++) {
      if (t[POINT_KEYS[i]] !== undefined) max = Math.max(max, arrLen(t[POINT_KEYS[i]]));
    }
    return max;
  }

  function totalPoints(data) {
    var n = 0, i;
    for (i = 0; i < data.length; i++) n += tracePoints(data[i]);
    return n;
  }

  function modeOf(data) {
    var has3 = false, has2 = false, i, ty;
    for (i = 0; i < data.length; i++) {
      ty = String((isPlainObj(data[i]) && data[i].type) || 'scatter');
      if (TYPES_3D.indexOf(ty) >= 0) has3 = true; else has2 = true;
    }
    if (has3 && has2) return 'mixed';
    return has3 ? '3d' : '2d';
  }

  function frozen(o) {
    try { return Object.freeze(o); } catch (e) { return o; }
  }

  function hintFor(mode) {
    if (mode === '3d') return '拖动旋转视角、滚轮缩放；双击回到初始视角（悬停可看坐标）';
    if (mode === 'mixed') return '三维子图：拖动旋转、滚轮缩放；二维子图：拖动平移、框选缩放（悬停可看坐标）';
    return '左键拖动平移、框选缩放、双击回到初始视野（悬停可看坐标）';
  }

  /* ------------------------------------------------------------------ */
  /* 渲染器                                                             */
  /* 本页内联的是哪一份库 —— 从页面自己的 meta 标记读（make_widget.py 写的），读不到就只说版本号。
     这样做是为了让图注说的和页面里实际内联的那份一致：full（4.85 MB）与 gl3d（1.69 MB）用户能分辨。 */
  function vendorLabel(Plotly) {
    var v = (Plotly && Plotly.version) || '?';
    var name = '';
    try {
      var m = document.querySelector ? document.querySelector('meta[name="wg-vendor-path"]') : null;
      if (m) name = String(m.getAttribute('content') || '').split('/').pop();
    } catch (e) { name = ''; }
    return 'plotly.js v' + v + (name ? '（' + name + '）' : '');
  }

  /* ------------------------------------------------------------------ */
  var HOSTS = (typeof WeakMap === 'function') ? new WeakMap() : null;   // figure 容器 → {gd}
  var PLOT_CLASS = 'wg-plotly';

  function purgeQuiet(Plotly, gd) {
    try { Plotly.purge(gd); } catch (e) { /* 已经坏掉的图别让整页渲染中断 */ }
  }

  /* widgets.js 的 redraw 会在同步 drawFigure 返回后先 flushErrors()；Plotly 的
     Promise reject 到这里时，必须再刷新一次，才能让错误进入页面错误区。 */
  function flushErrorsQuiet(it) {
    if (!it || typeof it.flushErrors !== 'function') return;
    try { it.flushErrors(); } catch (e) { /* 错误刷新本身不能再让异步回调抛出 */ }
  }

  function reportPlotlyFailure(it, e) {
    var layoutError = null;
    /* 先重排：若宽度变化触发了同步 redraw，它会重置错误收集；下面再追加原始错误。 */
    try { it.layout(); } catch (reflow) { layoutError = reflow; }
    if (layoutError) it.err('重排失败：' + msgOf(layoutError));
    it.err('Plotly 渲染失败：' + msgOf(e));
    flushErrorsQuiet(it);
  }

  function reportLayoutFailure(it, e) {
    it.err('重排失败：' + msgOf(e));
    flushErrorsQuiet(it);
  }

  function checkRuntime(it) {
    if (typeof it.ev !== 'function' || typeof it.setCaption !== 'function' ||
        typeof it.figureWidth !== 'function' || typeof it.layout !== 'function' || !it.el || !it.el.figure) {
      it.err('plotly 需要较新的 widgets.js（it.ev / it.setCaption / it.figureWidth / it.layout / it.el.figure）；' +
        '本页的运行时太旧：请用 make_widget.py 重建组件');
      return false;
    }
    return true;
  }

  function drawPlotly(ctx) {
    var it = ctx.it, spec = ctx.spec;
    var Plotly = global.Plotly;
    if (!checkRuntime(it)) return;
    if (!Plotly || typeof Plotly.newPlot !== 'function' || typeof Plotly.react !== 'function') {
      it.err('plotly 需要页面内联的 Plotly 库，但本页没有 window.Plotly：' +
        '请用 make_widget.py 重建该组件（并确认 _meta/tools/vendor/ 下该 bundle 的 plotly*.min.js 存在，' +
        '清单见 _meta/tools/vendor/README.md）');
      return;
    }
    var p = isPlainObj(spec.plotly) ? spec.plotly : null;
    if (!p) { it.err('plotly 必须给 spec.plotly{data: […]}'); return; }
    if (!Array.isArray(p.data) || !p.data.length) {
      it.err('plotly.data 必须是非空数组（每个元素是一个 Plotly trace 对象）；当前 ' + describe(p.data));
      return;
    }
    if (p.layout !== undefined && !isPlainObj(p.layout)) {
      it.err('plotly.layout 必须是对象；当前 ' + describe(p.layout));
      return;
    }
    if (p.config !== undefined && !isPlainObj(p.config)) {
      it.err('plotly.config 必须是对象；当前 ' + describe(p.config));
      return;
    }

    var W = it.figureWidth();
    var data, layout, config;
    try {
      data = evalTree(it, p.data, 'plotly.data');
      /* layout 也要过一遍 `=` 求值（文档承诺过），再把求值后的结果交给 buildLayout 补默认值：
         否则 layout 里的 "=-1"、"=n" 会被原样交给 Plotly，静默不生效。 */
      layout = buildLayout(it, spec,
        isPlainObj(p.layout) ? Object.assign({}, p, {layout: evalTree(it, p.layout, 'plotly.layout')}) : p, W);
      config = buildConfig(p);
    } catch (e) {
      it.err('plotly spec 求值失败：' + msgOf(e));
      return;
    }

    /* 先发布读数再画：readouts 在同一帧里紧接着求值，等 Promise 落地就太晚了。
       这些数是按 spec 算的（不是从像素反解）。 */
    var info = { traces: data.length, points: totalPoints(data), height: layout.height,
      width: W, mode: modeOf(data) };
    it.vars.__plotly__ = frozen({ traces: info.traces, points: info.points, height: info.height,
      width: info.width, mode: info.mode });
    /* 口径与提示：普通字符串内嵌也看得见；方法/口径行用 WG.det(...) 标出来（内嵌精简模式会跳过） */
    it.setCaption([
      hintFor(info.mode),
      info.traces + ' 条轨迹、' + info.points + ' 个数据点，画布 ' + info.width + '×' + info.height + ' px',
      WG.det('画布由内联的 Plotly 独立渲染（' + vendorLabel(Plotly) + '）：高度按图区宽度取 ' +
        'clamp(宽度×0.72, 260, 520) px（spec.plotly.height 可覆盖），宽度由 Plotly 按容器算（responsive）。' +
        '标题由页面 DOM 承担，layout.title 被忽略。'),
      WG.det('底色默认固定浅色；只有 spec.theme="system" 且系统是深色时才读组件主题取深色。' +
        '颜色循环默认取组件调色板，layout.colorway 可覆盖；轴网格/hover 框同样只补作者没写的字段。'),
      WG.det('plotly.data / plotly.layout 里以 = 开头的字符串在渲染时按表达式求值一次' +
        '（作用域：vars 的键、controls 的 key 与助手函数；整体求值，不是逐点作用域），其余字符串原样传给 Plotly。')
    ]);

    var fig = it.el.figure;
    var entry = HOSTS ? HOSTS.get(fig) : null;
    var gd = (entry && entry.gd) ? entry.gd : null;
    if (gd && gd.parentNode !== fig) {
      if (!gd.parentNode && !fig.firstChild) {
        /* 运行时刚 clear 过图区：上一帧的图还活着，挂回去再 react（否则 react 会更新到已摘除的节点） */
        fig.appendChild(gd);
      } else {
        purgeQuiet(Plotly, gd);            // 容器被别的渲染器/别的图占了：旧图彻底放掉
        gd = null;
      }
    }
    if (!gd && HOSTS) {
      try { HOSTS['delete'](fig); } catch (e) { /* 忽略 */ }
    }

    var promise;
    try {
      if (gd) {
        promise = Plotly.react(gd, data, layout, config);       // 重绘走 react，不重复建图
      } else {
        gd = document.createElement('div');
        gd.className = PLOT_CLASS;
        fig.appendChild(gd);
        if (HOSTS) HOSTS.set(fig, { gd: gd });
        promise = Plotly.newPlot(gd, data, layout, config);
      }
    } catch (e) {
      it.err('Plotly 渲染失败：' + msgOf(e));
      return;
    }

    /* 画完再量一次高度：内嵌 iframe 是内容撑高的，画布长出来之后必须重新报一次 */
    Promise.resolve(promise).then(function () {
      try { it.layout(); }
      catch (e) { reportLayoutFailure(it, e); }
    }, function (e) {
      reportPlotlyFailure(it, e);
    });
  }

  WG.registerKind(KIND, drawPlotly);
})(typeof window !== 'undefined' ? window : this);
