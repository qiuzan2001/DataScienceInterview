/*
 * Dataview 自定义视图：把 Maps/_widgets 下的交互组件页内嵌进笔记。
 *
 * 笔记里这样用（dataviewjs 代码块）：
 *
 *   await dv.view("Maps/_tools/widget-embed", { file: "交互组件示例-箱线图.html" })
 *
 * 也可以写完整路径：
 *   await dv.view("Maps/_tools/widget-embed", { file: "Maps/_widgets/交互组件示例-箱线图.html" })
 *
 * 尺寸参数（都可以不写）：
 *   height: 数字         → 固定高度（px）。内容超出时窗口内滚动。
 *   height: "auto" / 不写 → 高度 = 组件内容的实际高度（默认），图不会被裁、也不会有上下滚动条。
 *   minHeight: 数字      → 自适应模式的兜底下限，默认 140。
 *   maxHeight: 数字      → 自适应模式的上限，默认不限（0）。给个值可以限制超长组件占屏。
 *   resizable: false     → 去掉右下角拖拽把手；默认显示。
 *
 * 右下角把手：拖动改宽/改高（宽超过笔记栏宽时容器内横向滚动），双击恢复"满宽 + 自适应"。
 *
 * ── 为什么是 srcdoc 而不是 src ────────────────────────────────────────
 * 这个环境里 iframe 的地址方案只有三种，前两种不可靠（都查证/实测过）：
 *   · file://        → 从非 file 源被 Chromium 拦掉（实测：body 长度为 0）
 *   · app://<id>/…   → 可用，但 <id> 每次启动随机生成（qe(36)），静态笔记写不了
 *   · srcdoc="<整页>" → 实测可用：内联 <style> 生效、<script> 执行
 * 于是这里把组件 HTML 读进来塞进 srcdoc：不需要服务、不占端口、不发网络请求。
 *
 * ── 自适应高度怎么来的 ────────────────────────────────────────────────
 * srcdoc 的 iframe 与笔记同源，所以能直接量它内部的文档高度：量 body 盒高 / body.scrollHeight /
 * .wg-wrap 盒高取最大值，回设到 iframe 的 style.height；组件自己长高（换控件、切窄屏布局、
 * 加了一行 readout）时它还会 postMessage 报高度，宿主收到就重新量一次并收敛几轮。
 * 四条兜底（都是为了让"旧组件页 + 新宿主"也长对）：
 *   1. 注入一段样式，强制 .wg-figure 不被 72vh 截断（组件旧版没有 data-wg-embed 规则时的保险）；
 *   2. 盯 iframe 宽度，变了就替浏览器把 resize 补送进 iframe —— 组件只在收到 resize/
 *      ResizeObserver 时才按新宽度重排，否则会"窗口变宽了图不变宽"甚至内部冒滚动条；
 *   3. 组件内交互（点、拖、键、改输入）后补量一次，覆盖旧组件不 postMessage 的情况；
 *   4. 万一量不到（比如 iframe 被沙箱化成不同源），就退回固定高度 420，不会白屏也不会卡死。
 * 组件侧配合：widgets.js 在内嵌时给 <html> 打 data-wg-embed，widgets.css 据此取消
 * .wg-figure 的 max-height:72vh —— 否则"视口"只是 iframe 自己，72vh 会把图挤出一条内滚动条。
 *
 * 关于 await：Dataview 执行视图文件时是 `new Function("dv","input", 源码)`，所以最外层这个
 * IIFE 的 promise 不会被传出去——`await dv.view(...)` 并不真的等我们。因此**错误必须由视图自己
 * 渲染到笔记里**（下面 try/catch 就是干这个的），否则读取失败会变成静默的未处理拒绝、笔记里什么都没有。
 *
 * 已知限制：Dataview 在**索引版本变化**时（新增/修改文件）会重新渲染 dataviewjs 块，
 * 那一刻组件会被重建——拖到一半的滑块回到初值，手动拖过的窗口尺寸也回到默认。这是 Dataview
 * 的既定行为（按 index.revision 重渲染），不是本视图能控制的；介意的话就把组件单独在浏览器里打开对照。
 */
(async () => {
  const host = dv.container;
  const showErr = (text) => { try { host.createEl("div", { cls: "frm-embed-err", text }); } catch (e) { /* 连错误都写不进去就算了 */ } };
  try {
  const spec = (input && typeof input === "object") ? input : { file: input };
  const raw = String(spec.file || spec.widget || "").trim();

  const show = (text) => host.createEl("div", { cls: "frm-embed-err", text });
  const num = (v, dflt, lo, hi) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.max(lo, Math.min(hi, Math.round(n))) : dflt;
  };

  if (!raw) {
    show('[错误] 没有指定组件文件。写法：dv.view("Maps/_tools/widget-embed", { file: "名字.html" })');
    return;
  }
  if (/^https?:/i.test(raw)) {
    show('[错误] 不要写 URL（本功能不需要服务与端口）：写 vault 相对路径或文件名即可。');
    return;
  }

  const rel = raw.replace(/\\/g, "/").replace(/^\.\//, "");
  if (rel.split("/").indexOf("..") >= 0) {
    show("[错误] 路径里不能有 ..");
    return;
  }
  if (!/\.html?$/i.test(rel)) {
    show("[错误] 只支持 .html 组件页：" + raw);
    return;
  }

  const adapter = dv.app.vault.adapter;
  /* 传完整路径时不猜；只传文件名时才读项目配置。
     这样默认仍是 Maps/_widgets，而 StateFarm 这种 `_widgets` 布局也不用把目录写死在视图里。 */
  let widgetDir = "Maps/_widgets";
  if (rel.indexOf("/") < 0) {
    try {
      const configPath = "widgets.config.json";
      const configExists = (typeof adapter.exists === "function") ? await adapter.exists(configPath) : false;
      if (configExists) {
        const configRaw = await adapter.read(configPath);
        const config = JSON.parse(configRaw);
        const configured = config && typeof config.widgetsDir === "string"
          ? config.widgetsDir.replace(/\\/g, "/").replace(/^\.\//, "") : "";
        if (configured && !configured.startsWith("/") && configured.split("/").indexOf("..") < 0) {
          widgetDir = configured;
        }
      }
    } catch (e) {
      /* 配置不存在/暂时读不到：继续默认布局；真正的组件文件不存在时再给可见错误 */
    }
  }
  const path = rel.indexOf("/") < 0 ? widgetDir + "/" + rel : rel;
  const exists = (typeof adapter.exists === "function") ? await adapter.exists(path) : true;
  if (!exists) {
    show("[错误] 找不到文件：" + path);
    return;
  }

  const htmlRaw = await adapter.read(path);

  /* ---- 精简模式（默认）：只留交互图 ----
     组件页里那些"读一遍就够"的解释性文字（副标题、来源链接、说明清单、页脚）在内嵌时会
     把图挤下去；它们应该写在**源笔记**（md）里，跟推导、出处、上下文放在一起。
     做法是往组件页 <head> 插一条 <meta name="wg-chrome" content="slim">：
     widgets.js 读到它就不生成那几个块；写 chrome:"full" 可以要回完整版（比如想在图边上
     直接看口径时）。真源（磁盘上的 html）一直不变，所以单独打开组件页永远是完整版。 */
  const chromeOpt = (spec.chrome === undefined || spec.chrome === null) ? "slim" : String(spec.chrome);
  const withMeta = (doc0, name, content) => {
    const tag = '<meta name="' + name + '" content="' + content + '">';
    const m = /<head[^>]*>/i.exec(doc0);
    if (m) return doc0.slice(0, m.index + m[0].length) + tag + doc0.slice(m.index + m[0].length);
    const h = /<\/head>/i.exec(doc0);
    if (h) return doc0.slice(0, h.index) + tag + doc0.slice(h.index);
    return tag + doc0;                       // 连 head 都没有的畸形页：插到最前面，总比不生效强
  };
  const html = withMeta(htmlRaw, "wg-chrome", chromeOpt === "full" ? "full" : "slim");

  /* ---- 尺寸模式 ---- */
  const hOpt = spec.height;
  const autoH = (hOpt === undefined || hOpt === null || hOpt === "" || String(hOpt).toLowerCase() === "auto");
  const minH = num(spec.minHeight, 140, 80, 4000);
  const maxH = num(spec.maxHeight, 0, 0, 8000);                 // 0 = 不限
  const fallbackH = autoH ? 420 : num(hOpt, 420, 120, 4000);    // 量不到内容高度时用这个
  const resizable = spec.resizable !== false;
  let mode = autoH ? "auto" : "fixed";

  const box = host.createEl("div", { cls: "frm-embed" });
  /* shell 是拖拽时的参照物：宽度超出去时由它横向滚动，把手始终贴在 iframe 右下角 */
  const shell = box.createEl("div", { cls: "frm-embed-shell" });
  const frame = shell.createEl("iframe", { cls: "frm-embed-frame" });
  frame.setAttribute("height", String(autoH ? minH : fallbackH));
  frame.setAttribute("referrerpolicy", "no-referrer");
  frame.style.height = (autoH ? minH : fallbackH) + "px";
  // 用属性而不是拼字符串：组件 HTML 里的引号与 </script> 都不会破坏外层文档
  frame.setAttribute("srcdoc", html);

  /* ---- 自适应高度 ---- */
  let curH = 0;
  const innerDoc = () => { try { return frame.contentDocument || null; } catch (e) { return null; } };
  /* 改高度之后隔一拍再量一轮：iframe 一变换高度，内部可能多出/少掉一条滚动条，
     内容宽度跟着变、组件重排，内容高度又变（实测能差 60px，表现为底部一大片空白）。
     再量一轮就收敛；最多连量 3 轮，防极端情况下反复抖。 */
  let settleTimer = 0, settleRounds = 0;
  const scheduleSettle = () => {
    if (settleTimer || settleRounds >= 3) return;
    settleTimer = setTimeout(() => {
      settleTimer = 0;
      settleRounds++;
      fit();
    }, 90);
  };
  /* 唯一的"改高度"出口：夹上下限、只在真的变了才写 DOM（避免和组件的 ResizeObserver 来回抖） */
  const applyHeight = (raw) => {
    if (mode !== "auto") return;
    if (!(raw > 0)) return;
    let h = Math.ceil(raw) + 2;                  // +2：亚像素差一像素就会冒滚动条
    if (maxH) h = Math.min(h, maxH);
    if (h < minH) h = minH;
    if (!curH || Math.abs(h - curH) > 0.5) {
      curH = h;
      frame.style.height = h + "px";
      frame.setAttribute("height", String(h));
      if (placeGrip) placeGrip();              // 把手得跟着新高度走（它按 iframe 实际尺寸摆）
      if (gripRef) gripRef.setAttribute("title", gripTitle());
      scheduleSettle();
    }
  };
  /* 外部触发的"重新量一次"：把收敛计数清零（新一轮变化允许再收敛 3 轮） */
  const refit = () => { settleRounds = 0; return fit(); };

  /* ---- 宿主侧宽度同步 ----------------------------------------------------
     组件只有在"它自己的窗口收到 resize / 它自己的 ResizeObserver 投递"时才会按新宽度重排。
     如果 iframe 的视口宽度变了、但那个帧的渲染生命周期没被投递（窗口被遮挡、后台标签、
     某些嵌入环境），组件就会一直按旧宽度画 —— 表现为"窗口变宽了图却没跟着变宽"、
     甚至内部冒出一条滚动条。宿主在主文档里盯 iframe 的宽度，一变就替浏览器把 resize
     事件补送进 iframe，再重新量一次高度；主文档的生命周期是活的，这条路可靠。 */
  let lastW = 0;
  let widthFixes = 0;               // "组件按带滚动条的窄宽度画的"修正次数，改宽后重置
  let roHost = null;
  const pushResize = () => {
    let iw = null;
    try { iw = frame.contentWindow; } catch (e) { iw = null; }
    if (iw && frame.contentDocument && typeof iw.Event === "function") {
      try { iw.dispatchEvent(new iw.Event("resize")); } catch (e) { /* 补送失败就算了 */ }
    }
  };
  const syncWidth = () => {
    const w = frame.offsetWidth || 0;
    if (!(w > 0) || Math.abs(w - lastW) <= 0.5) return;
    lastW = w;
    widthFixes = 0;                 // 新宽度：允许再修正几次"带滚动条量出来的宽度"
    pushResize();
    refit();
  };
  if (typeof ResizeObserver === "function") {
    try { roHost = new ResizeObserver(() => syncWidth()); roHost.observe(frame); } catch (e) { roHost = null; }
  }
  /* 浏览器窗口尺寸变了，笔记栏宽一般也跟着变：一并同步一次（RO 不可用时这是唯一的信号） */
  const onWindowResize = () => syncWidth();
  try { window.addEventListener("resize", onWindowResize); } catch (e) { /* 忽略 */ }
  /* 量组件内容高度：只看**内容**高度（body 盒高 / body.scrollHeight / .wg-wrap 盒高取最大）。
     特意不用 documentElement.scrollHeight：它是 max(内容, 视口高)，iframe 一旦比内容高，
     这个值就等于视口高，再加上下面那 2px 余量，每次重算都会把 iframe 撑高 2px —— 无界递增。
     返回 false 表示量不到（iframe 不是同源 / 还没 body），那时只能靠组件报来的高度。 */
  const fit = () => {
    const doc = innerDoc();
    if (!doc || !doc.body) return false;
    let h = 0;
    try { h = Math.max(h, doc.body.getBoundingClientRect().height || 0); } catch (e) { /* 量不到就算了 */ }
    try { h = Math.max(h, doc.body.scrollHeight || 0); } catch (e) { /* 同上 */ }
    try {
      const wrap = doc.body.querySelector && doc.body.querySelector(".wg-wrap");
      if (wrap) h = Math.max(h, wrap.getBoundingClientRect().height || 0);
    } catch (e) { /* 同上 */ }
    applyHeight(h);
    /* 组件量宽度时可能带过一条瞬时滚动条的宽度：iframe 一开始只有 minH 那么高、内容却更高，
       body 会短暂滚动；高度撑开后滚动条没了，但组件不会自己知道，于是它按窄 15px 画、
       窄屏断点也可能判错。发现"组件用的宽度 ≠ iframe 宽度"就再补一次 resize（每次改宽最多补 3 次）。 */
    try {
      const wrap = doc.body.querySelector && doc.body.querySelector(".wg-wrap");
      const rw = wrap ? Math.round(wrap.getBoundingClientRect().width) : 0;
      const fw = Math.round(frame.offsetWidth || 0);
      if (rw > 0 && fw > 0 && Math.abs(fw - rw) >= 8 && widthFixes < 3) {
        widthFixes++;
        pushResize();
      }
    } catch (e) { /* 量不到就算了 */ }
    return true;
  };
  /* ---- 组件自己报高度（触发器）----
     widgets.js 每次重绘/重排后都会 postMessage({wg:"widget-height"})。为什么需要它：
     宿主窗口被遮挡时 Chromium 会推迟渲染生命周期（rAF / ResizeObserver 都不投递），
     我们就收不到"该重新量一次了"的信号；而 postMessage 由事件循环投递，什么时候都到。
     被沙箱化成不同源时宿主读不到 iframe 内部文档，它还是唯一的高度来源。 */
  const onMessage = (ev) => {
    if (mode !== "auto") return;
    let w = null;
    try { w = frame.contentWindow; } catch (e) { w = null; }
    if (!w || ev.source !== w) return;                    // 只认自己这个 iframe 发来的
    const d = ev.data;
    if (!d || d.wg !== "widget-height") return;
    /* 能自己量就用自己量的（更新鲜、也更贴身）；量不到才用组件报来的值 */
    if (!refit()) applyHeight(Number(d.height));
  };
  try { window.addEventListener("message", onMessage); } catch (e) { /* 挂不上就算了，还有 fit() */ }

  /* 兜底：ResizeObserver 盯组件内部的盒子。用 iframe 自己那个 realm 的构造函数 ——
     观察别的文档里的元素时，跨 realm 的 observer 不保证收到通知。 */
  let ro = null, mo = null;
  const stopObservers = () => {
    try { if (ro) ro.disconnect(); } catch (e) { /* 忽略 */ }
    try { if (mo) mo.disconnect(); } catch (e) { /* 忽略 */ }
    // message 监听不在这里摘：整块被换掉时才由 cleanup() 摘（见下）
    ro = null; mo = null;
  };
  const watch = () => {
    const doc = innerDoc();
    if (!doc || !doc.body) return;
    stopObservers();
    let RO = null;
    try { RO = (frame.contentWindow && frame.contentWindow.ResizeObserver) || null; } catch (e) { RO = null; }
    if (!RO && typeof ResizeObserver === "function") RO = ResizeObserver;
    if (RO) {
      try {
        ro = new RO(() => refit());
        ro.observe(doc.body);
        const wrap = doc.body.querySelector && doc.body.querySelector(".wg-wrap");
        if (wrap) ro.observe(wrap);
      } catch (e) { ro = null; }
    }
    /* Dataview 重渲染会把整个块换掉：那时把手/observer 都该收摊，别留着盯一个死 iframe */
    if (typeof MutationObserver === "function") {
      try {
        mo = new MutationObserver(() => { if (!box.isConnected) cleanup(); });
        if (box.parentNode) mo.observe(box.parentNode, { childList: true });
      } catch (e) { mo = null; }
    }
  };

  /* 收摊：observers 关掉，事件监听都摘掉（整块被 Dataview 换掉时才会走到这里） */
  /* 用户在组件里动一下（拖滑杆、点按钮、改格子）之后补量一次：新版 widgets.js 会自己
     postMessage，旧版不会；这条兜底让"旧组件页 + 新宿主"也能长对高度。 */
  const INTERACT_EVENTS = ["pointerup", "click", "keyup", "change", "input"];
  var onInteract = null;
  let iTimer = 0;
  const cleanup = () => {
    stopObservers();
    try { window.removeEventListener("resize", onWindowResize); } catch (e) { /* 忽略 */ }
    try { if (onGripResize) window.removeEventListener("resize", onGripResize); } catch (e) { /* 忽略 */ }
    try { if (roHost) { roHost.disconnect(); roHost = null; } } catch (e) { /* 忽略 */ }
    if (iTimer) { clearTimeout(iTimer); iTimer = 0; }
    const d = innerDoc();
    if (d && onInteract) {
      INTERACT_EVENTS.forEach((t) => { try { d.removeEventListener(t, onInteract, true); } catch (e) { /* 忽略 */ } });
    }
    try { window.removeEventListener("message", onMessage); } catch (e) { /* 忽略 */ }
  };
  /* ---- 右下角把手：拖宽拖高，双击回默认 ----
     把手不能挂在 shell 的 right/bottom 上：shell 宽度是整栏宽，而手动拖过的 iframe 可能比它窄，
     那样把手会飘在 iframe 右边缘之外（框已经变窄了，把手还贴在栏边）。
     所以用 JS 按 iframe 实际尺寸摆（shell 是定位祖先）。 */
  var placeGrip = null, gripRef = null, onGripResize = null;
  /* 把手上的说明实时写当前尺寸：用户一悬停就知道"现在多大、双击回到哪" */
  function gripTitle() {
    return "拖动改宽/改高（当前 " + Math.round(frame.offsetWidth || 0) + " × " +
      Math.round(frame.offsetHeight || 0) + " px）；双击恢复" + (autoH ? "满宽 + 自适应" : "默认尺寸");
  }
  if (resizable) {
    const grip = shell.createEl("div", { cls: "frm-embed-grip" });
    gripRef = grip;
    grip.setAttribute("title", gripTitle());
    grip.setAttribute("aria-label", "拖动改变内嵌窗口大小，双击恢复默认");
    placeGrip = () => {
      const w = frame.offsetWidth || 0, h = frame.offsetHeight || 0;
      if (w > 0) grip.style.left = Math.max(0, w - 22) + "px";     // 22 = 把手 18px + 4px 内缩
      if (h > 0) grip.style.top = Math.max(0, h - 22) + "px";
    };
    placeGrip();
    onGripResize = () => placeGrip();
    window.addEventListener("resize", onGripResize);
    let drag = null;
    const stop = (ev) => {
      if (!drag) return;
      drag = null;
      shell.removeAttribute("data-dragging");
      try { if (ev && grip.releasePointerCapture) grip.releasePointerCapture(ev.pointerId); } catch (e) { /* 忽略 */ }
    };
    grip.addEventListener("pointerdown", (ev) => {
      try { ev.preventDefault(); } catch (e) { /* 忽略 */ }
      const r = frame.getBoundingClientRect();
      drag = { x: ev.clientX, y: ev.clientY, w: r.width, h: r.height };
      mode = "fixed";                       // 一旦手动拖过，就不再被内容高度牵着走
      shell.setAttribute("data-dragging", "1");
      try { if (grip.setPointerCapture) grip.setPointerCapture(ev.pointerId); } catch (e) { /* 忽略 */ }
    });
    grip.addEventListener("pointermove", (ev) => {
      if (!drag) return;
      const w = Math.max(220, Math.round(drag.w + (ev.clientX - drag.x)));
      const h = Math.max(120, Math.round(drag.h + (ev.clientY - drag.y)));
      frame.style.width = w + "px";
      frame.style.height = h + "px";
      frame.setAttribute("height", String(h));
      curH = h;
      if (placeGrip) placeGrip();
      grip.setAttribute("title", gripTitle());
    });
    grip.addEventListener("pointerup", stop);
    grip.addEventListener("pointercancel", stop);
    grip.addEventListener("dblclick", () => {
      drag = null;
      frame.style.width = "100%";
      if (autoH) { mode = "auto"; curH = 0; refit(); }
      else {
        mode = "fixed";
        curH = fallbackH;
        frame.style.height = fallbackH + "px";
        frame.setAttribute("height", String(fallbackH));
      }
      if (placeGrip) placeGrip();
      grip.setAttribute("title", gripTitle());
    });
  }

  frame.addEventListener("load", () => {
    /* ---- 组件页里的 obsidian:// 链接（"在 Obsidian 中打开源笔记"）----
       在 iframe 内点它会被 Obsidian 的导航闸拦下（只放行 http(s)）。srcdoc 文档与主窗口同源，
       所以直接给它的 document 挂捕获监听，拦下这类链接改走 Obsidian 自己的 API。 */
    let doc = null;
    try { doc = frame.contentDocument; } catch (e) { doc = null; }
    if (!doc) {
      /* 读不到内部文档（例如 iframe 被沙箱化成不同源）：先给一个可用的兜底高度，
         之后靠组件自己 postMessage 报高度来修正（onMessage 不依赖同源）。
         不改成 fixed 模式——那样就把组件报来的高度也一起挡掉了。 */
      curH = fallbackH;
      frame.style.height = fallbackH + "px";
      frame.setAttribute("height", String(fallbackH));
      return;
    }
    if (doc.addEventListener) {
      doc.addEventListener("click", (ev) => {
        const a = (ev.target && ev.target.closest) ? ev.target.closest('a[href^="obsidian://"]') : null;
        if (!a) return;
        ev.preventDefault();
        let file = "";
        try { file = new URL(String(a.getAttribute("href"))).searchParams.get("file") || ""; } catch (e) { file = ""; }
        if (!file) return;
        let decoded = file;
        try { decoded = decodeURIComponent(file); } catch (e) { /* 已经是明文就用原样 */ }
        try {
          const mc = dv.app.metadataCache;
          if (mc && typeof mc.getFirstLinkpathDest === "function") {
            const dest = mc.getFirstLinkpathDest(decoded, "");
            if (!dest) return;                     // 源笔记不在了：不开，也不顺手建空笔记
            dv.app.workspace.openLinkText(dest.path, "", false);
          } else {
            dv.app.workspace.openLinkText(decoded, "", false);
          }
        } catch (e) { /* 打不开就算了 */ }
      }, true);
    }
    /* ---- 兜底样式 ----------------------------------------------------
       万一组件页是旧版本（没有 html[data-wg-embed] 那条规则），.wg-figure 的 max-height:72vh
       会把图截出一条内部滚动条（"窗口高度固定、图要上下滚"就是这么来的）。宿主再压一层：
       无论组件新旧，图都按内容撑开、不内部滚动。同源才做得到；不同源就静默跳过。 */
    try {
      const st = doc.createElement("style");
      st.setAttribute("data-frm-embed-fallback", "1");
      /* 横向用 clip 而不是 hidden：hidden 会把 body 变成滚动容器（并隐式把 overflow-y 变成 auto），
         组件量到的宽度就会少掉一条滚动条、窄屏断点跟着抖；clip 只裁不滚。 */
      st.textContent = ".wg-figure{max-height:none !important;}" +
        ".wg-wrap{padding:12px 12px 14px !important;}" +
        "html,body{overflow-x:clip;}" +
        /* 精简模式的兜底：旧组件页（它的内联 CSS 里没有 data-wg-chrome 那几条、它的
           widgets.js 也不认识 wg-chrome）照样能把解释性文字收起来。 */
        (chromeOpt === "full" ? "" :
          ".wg-sub,.wg-src,.wg-notes,.wg-foot{display:none !important;}" +
          ".wg-head{margin-bottom:9px !important; padding-bottom:7px !important;}");
      (doc.head || doc.documentElement).appendChild(st);
    } catch (e) { /* 注入失败不影响主流程 */ }
    /* ---- 交互后补量一次（组件自己会报时这条不会有害，只是多量一次）---- */
    onInteract = () => {
      if (mode !== "auto") return;
      if (iTimer) clearTimeout(iTimer);
      iTimer = setTimeout(() => { iTimer = 0; refit(); }, 160);
    };
    INTERACT_EVENTS.forEach((t) => {
      try { doc.addEventListener(t, onInteract, true); } catch (e) { /* 忽略 */ }
    });
    /* 首次把宽度同步一遍：也顺便记下 lastW，之后再变才补 resize */
    syncWidth();
    /* 字体/布局落定后多量几次：一次 load 往往还量不到最终高度 */
    watch();
    refit();
    try { if (doc.fonts && doc.fonts.ready && doc.fonts.ready.then) doc.fonts.ready.then(refit).catch(() => {}); } catch (e) { /* 忽略 */ }
    try {
      const raf = (typeof requestAnimationFrame === "function") ? requestAnimationFrame : (f) => setTimeout(f, 16);
      raf(() => refit());
    } catch (e) { /* 忽略 */ }
    setTimeout(refit, 60);
    setTimeout(refit, 250);
  });
  } catch (e) {
    // 读取/渲染链路任何一步失败都要在笔记里看得见，而不是静默无输出
    showErr('[错误] 内嵌组件失败：' + ((e && e.message) || e));
  }
})();
