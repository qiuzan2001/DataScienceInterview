'use strict';
/* Dataview 视图（Maps/_tools/widget-embed/view.js）测试——在 Obsidian 之外跑
 *
 * 按 Dataview 的真实调用方式执行这份文件：`new Function("dv", "input", 源码)`，
 * 所以这里能真验到"视图源码 + dv/input"这一层的解析与 srcdoc 装配。
 * 不验 Obsidian 里的视觉外观（那要人眼）。
 *
 * 跑法（两种都行；找不到 vault 时只跳过"必须读真 vault 文件"的断言，退出码仍是 0）：
 *   · 在 vault 里：node Maps/_tools/test_widget_embed_view.js
 *     （当前目录下若有 Maps/_widgets/ 就认它；也可以用 KW_VAULT=/path/to/vault 显式指定）
 *   · 在别的项目里：工具链被拷进 <项目>/.widgets/toolkit/ 后照常
 *     node .widgets/toolkit/test_widget_embed_view.js
 */
const fs = require('fs');
const path = require('path');
const assert = require('assert');
/* window 替身：view.js 会挂 message 监听（组件用 postMessage 报高度），这里捕获下来，
   好在测试里手工投递一条消息。 */
/* window 替身：view.js 会挂一个 windmessage 监听（组件用 postMessage 报高度），
   这里把它捕获下来，好在测试里手工投递一条消息。 */
const msgListeners = [];
/* 别的类型（resize）也要能记下来并手工投递：视图用 window resize 兜底同步 iframe 宽度 */
const winListeners = {};
global.window = {
  addEventListener(type, fn) {
    (winListeners[type] = winListeners[type] || []).push(fn);
    if (type === 'message') msgListeners.push(fn);
  },
  removeEventListener(type, fn) {
    const i = msgListeners.indexOf(fn); if (i >= 0) msgListeners.splice(i, 1);
    const arr = winListeners[type] || [];
    const j = arr.indexOf(fn); if (j >= 0) arr.splice(j, 1);
  },
};
function fireWin(type, ev) { (winListeners[type] || []).slice().forEach((fn) => fn(ev || {})); }
function postToView(source, data) {
  msgListeners.slice().forEach((fn) => fn({ source: source, data: data }));
}

const HERE = __dirname;                      // 本测试自己所在目录（= 工具链目录）
const DEFAULT_TOOLS_DIR = 'Maps/_tools';
const DEFAULT_WIDGETS_DIR = 'Maps/_widgets';
function cleanProjectPath(raw, fallback) {
  const value = String(raw || '').replace(/\\/g, '/').replace(/^\.\//, '');
  return value && !value.startsWith('/') && !value.split('/').includes('..') ? value : fallback;
}
function layoutFor(root) {
  const out = { toolsDir: DEFAULT_TOOLS_DIR, widgetsDir: DEFAULT_WIDGETS_DIR };
  try {
    const cfg = JSON.parse(fs.readFileSync(path.join(root, 'widgets.config.json'), 'utf8'));
    if (cfg && typeof cfg === 'object' && !Array.isArray(cfg)) {
      out.toolsDir = cleanProjectPath(cfg.toolsDir, out.toolsDir);
      out.widgetsDir = cleanProjectPath(cfg.widgetsDir, out.widgetsDir);
    }
  } catch (e) { /* 没有配置或不是项目时使用默认布局 */ }
  return out;
}
const CANDIDATE_ROOT = process.env.KW_VAULT || process.cwd();
const PROJECT_LAYOUT = layoutFor(CANDIDATE_ROOT);
const looksLikeVault = fs.existsSync(path.join(CANDIDATE_ROOT, PROJECT_LAYOUT.widgetsDir));
const VAULT = looksLikeVault ? CANDIDATE_ROOT : null;
const TOOLS_DIR = PROJECT_LAYOUT.toolsDir;
const WIDGETS_DIR = PROJECT_LAYOUT.widgetsDir;
/* 指定了 KW_VAULT 但那个目录不像项目：不替你猜别的目录，但把原因说出来。 */
if (process.env.KW_VAULT && !looksLikeVault) {
  console.error('[提醒] KW_VAULT=' + process.env.KW_VAULT + ' 下没有 ' + PROJECT_LAYOUT.widgetsDir + '/：'
    + '这不是一个交互组件项目，依赖项目的断言会跳过。');
}
const VIEW = (VAULT && fs.existsSync(path.join(VAULT, TOOLS_DIR, 'widget-embed/view.js')))
  ? path.join(VAULT, TOOLS_DIR, 'widget-embed/view.js')
  : path.join(HERE, 'widget-embed', 'view.js');
const REGISTRY = VAULT ? path.join(VAULT, TOOLS_DIR, 'widgets-index.json') : null;
let REAL_WIDGET_PATHS = [];
if (VAULT && REGISTRY && fs.existsSync(REGISTRY)) {
  try {
    const registered = JSON.parse(fs.readFileSync(REGISTRY, 'utf8')).widgets;
    if (Array.isArray(registered)) REAL_WIDGET_PATHS = registered.map((w) => w && w.html).filter(Boolean);
  } catch (e) { /* 真实注册表错误由 make_widget --check 报告 */ }
}

/* ---- 没有项目时使用桩页面；有项目时优先验收注册表里的真实活跃页面 ---- */
const STUB_NAMES = ['交互组件示例-箱线图.html', '交互组件示例-经验分布.html', '交互组件示例-正态QQ.html',
  '交互组件示例-等高线.html', '交互组件示例-梯度场.html', '交互组件示例-剪切矩阵.html'];
const STUB_HTML = '<!doctype html>' + String.fromCharCode(10)
  + '<html><head><meta charset="utf-8"><title>桩组件页</title></head>' + String.fromCharCode(10)
  + '<body><div class="wg-wrap"><div class="wg-figure">桩组件页（没有项目时的最小替身）</div></div></body></html>' + String.fromCharCode(10);
const stubAt = (rel) => (rel.indexOf(DEFAULT_WIDGETS_DIR + '/') === 0
    && STUB_NAMES.indexOf(rel.slice((DEFAULT_WIDGETS_DIR + '/').length)) >= 0) ? STUB_HTML : null;
const FIXTURE_PATH = (VAULT && REAL_WIDGET_PATHS.length) ? REAL_WIDGET_PATHS[0]
  : DEFAULT_WIDGETS_DIR + '/' + STUB_NAMES[0];
const FIXTURE_FILE = path.basename(FIXTURE_PATH);
const FIXTURE_REQUESTS = VAULT && REAL_WIDGET_PATHS.length
  ? REAL_WIDGET_PATHS.map((file) => path.basename(file)) : STUB_NAMES.slice();

const NO_VAULT_HINT = '未找到 vault：设置 KW_VAULT=/path/to/vault（要求该目录下有 Maps/_widgets/）后重跑';
const skipped = [];
function skip(label, extra) {
  skipped.push(label);
  console.log('✓ (跳过 ' + label + ') ' + (extra ? extra + '；' : '') + NO_VAULT_HINT);
}

function makeEl(tag, opts) {
  const el = {
    tagName: String(tag).toUpperCase(),
    className: (opts && opts.cls) || '',
    textContent: (opts && opts.text) || '',
    attrs: {}, children: [], style: {}, parentNode: null, offsetWidth: 800, offsetHeight: 400,
    setAttribute(k, v) { this.attrs[k] = String(v); },
    getAttribute(k) { return Object.prototype.hasOwnProperty.call(this.attrs, k) ? this.attrs[k] : null; },
    removeAttribute(k) { delete this.attrs[k]; },
    /* 拖拽把手要用它拿初始宽高；宽度给 800 是为了让"拖宽"能算出确定值 */
    getBoundingClientRect() { return { width: 800, height: 400, left: 0, top: 0, right: 800, bottom: 400 }; },
    createEl(t, o) { const c = makeEl(t, o); c.parentNode = this; this.children.push(c); return c; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    listeners: {},
    addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); },
    fire(type, ev) { (this.listeners[type] || []).slice().forEach((fn) => fn(ev)); },
    findByClass(cls) {
      const hit = [];
      (function walk(n) {
        if (n.className && String(n.className).split(/\s+/).indexOf(cls) >= 0) hit.push(n);
        n.children.forEach(walk);
      })(this);
      return hit;
    },
  };
  return el;
}

/* 组件页内部的假文档：view.js 会量 .wg-wrap 与 body 的盒高来定 iframe 高度。
   注意只给"内容高度"——view.js 特意不看 documentElement.scrollHeight（那个会被视口污染）。 */
function makeInnerDoc(height) {
  const wrap = makeEl('div', { cls: 'wg-wrap' });
  const body = makeEl('body');
  const rect = () => ({ height: height, width: 900, left: 0, top: 0 });
  wrap.getBoundingClientRect = rect;
  body.getBoundingClientRect = rect;
  body.scrollHeight = height;
  body.querySelector = (sel) => (sel === '.wg-wrap' ? wrap : null);
  const doc = makeEl('html');
  doc.body = body;
  doc.documentElement = { scrollHeight: height };
  doc.fonts = null;
  return doc;
}

let readFails = false;                                   // true = 模拟读取抛错

const adapter = {
  async exists(rel) {
    if (!VAULT) return stubAt(rel) !== null;           // 没 vault：只认桩组件页（其余名字 = 文件不存在）
    try { fs.accessSync(path.join(VAULT, rel)); return true; } catch (e) { return false; }
  },
  async read(rel) {
    if (readFails && rel !== 'widgets.config.json') throw new Error('模拟读取失败');
    if (!VAULT) {
      const stub = stubAt(rel);
      if (stub === null) throw new Error('桩组件页不存在：' + rel);
      return stub;
    }
    return fs.readFileSync(path.join(VAULT, rel), 'utf8');
  },
};

const openCalls = [];        // 记录 obsidian:// 链接打开过哪些笔记

let metadataResolvable = true;      // false = 模拟源笔记已被删除

function runView(input) {
  const container = makeEl('div');
  const dv = {
    container,
    app: {
      vault: { adapter },
      workspace: { openLinkText: (file) => { openCalls.push(file); } },
      metadataCache: { getFirstLinkpathDest: (f) => (metadataResolvable ? { path: f } : null) },
    },
  };
  const src = fs.readFileSync(VIEW, 'utf8');
  // 与 Dataview 一致：new Function("dv", "input", 源码)
  const fn = new Function('dv', 'input', 'return (' + src.trim().slice(0, -1) + ');');
  return Promise.resolve(fn(dv, input)).then(() => container);
}

(async () => {
  // ---- 1) 正常内嵌 ----
  const c1 = await runView({ file: FIXTURE_FILE, height: 380 });
  const f1 = c1.findByClass('frm-embed-frame');
  assert.strictEqual(f1.length, 1, '应挂上 1 个 iframe');
  const doc1 = f1[0].getAttribute('srcdoc');
  assert.strictEqual(f1[0].getAttribute('height'), '380');
  /* srcdoc = 磁盘文件 + 宿主动态插入的一条 <meta name="wg-chrome">（精简模式开关）。
     把这条 meta 剥掉之后必须与磁盘文件逐字相同 —— 组件页本身（真源）不能被改写。
     这一条与"真页有 50k 以上体量"都得读到 vault 里的真组件页，没 vault 就只能跳过。 */
  if (VAULT) {
    assert.ok(doc1 && doc1.length > 50000, 'srcdoc 应有内容，实际 ' + (doc1 ? doc1.length : 0));
    const disk1 = fs.readFileSync(path.join(VAULT, FIXTURE_PATH), 'utf8');
    const stripped = doc1.replace(/<meta name="wg-chrome" content="[^"]*">/, '');
    assert.strictEqual(stripped, disk1, 'srcdoc 去掉注入的 meta 后应与磁盘文件一致');
  } else {
    assert.ok(doc1 && doc1.length > 0, 'srcdoc 应有内容，实际 ' + (doc1 ? doc1.length : 0));
    skip('1-srcdoc 与真组件页逐字一致（含真页体量 >50k）');
  }
  assert.ok(/<meta name="wg-chrome" content="slim">/.test(doc1), '默认应注入精简模式标记');
  console.log('✓ 1. 正常内嵌：' + doc1.length + ' 字符 = ' + (VAULT ? '磁盘文件' : '桩组件页') + ' + 注入的精简模式标记');

  // ---- 2) 只写文件名 → 自动补 Maps/_widgets/ ----
  const c2 = await runView({ file: FIXTURE_FILE });
  assert.strictEqual(c2.findByClass('frm-embed-frame').length, 1, '只写文件名也应成功');
  console.log('✓ 2. 只写文件名可用（自动补 Maps/_widgets/）');

  // ---- 3) 不写 height = 自适应：初始只是个兜底高度，load 后按内容高度回设 ----
  const c3 = await runView({ file: FIXTURE_FILE });
  const f3 = c3.findByClass('frm-embed-frame')[0];
  assert.strictEqual(f3.getAttribute('height'), '140', '自适应模式的初始高度 = minHeight 兜底');
  f3.contentDocument = makeInnerDoc(512);
  f3.fire('load');
  assert.strictEqual(f3.getAttribute('height'), '514', '自适应：内容 512px → iframe = 512 + 2 余量');
  assert.strictEqual(f3.style.height, '514px');
  console.log('✓ 3. 不写 height 时自适应：内容 512px → iframe 514px（不再固定 420）');

  // ---- 3b) height: "auto" 显式写法、minHeight / maxHeight 边界 ----
  const c3b = await runView({ file: FIXTURE_FILE, height: 'auto', maxHeight: 300 });
  const f3b = c3b.findByClass('frm-embed-frame')[0];
  f3b.contentDocument = makeInnerDoc(900);
  f3b.fire('load');
  assert.strictEqual(f3b.getAttribute('height'), '300', 'maxHeight 生效：900px 内容被压到 300');
  const c3c = await runView({ file: FIXTURE_FILE, minHeight: 260 });
  const f3c = c3c.findByClass('frm-embed-frame')[0];
  f3c.contentDocument = makeInnerDoc(40);
  f3c.fire('load');
  assert.strictEqual(f3c.getAttribute('height'), '260', 'minHeight 生效：40px 内容抬到 260');
  console.log('✓ 3b. height:"auto" + minHeight/maxHeight 边界都生效');

  // ---- 3c) 量不到内部文档（例如被沙箱化成不同源）→ 退回固定高度，不留矮框 ----
  const c3d = await runView({ file: FIXTURE_FILE });
  const f3d = c3d.findByClass('frm-embed-frame')[0];
  f3d.contentDocument = null;
  f3d.fire('load');
  assert.strictEqual(f3d.getAttribute('height'), '420', '量不到内容时退回 420，不留在 140 的矮框里');
  console.log('✓ 3c. 量不到内部文档时退回固定高度 420');

  // ---- 3d) 右下角把手：拖动改宽高，双击回自适应 ----
  const c3e = await runView({ file: FIXTURE_FILE });
  const f3e = c3e.findByClass('frm-embed-frame')[0];
  const grips = c3e.findByClass('frm-embed-grip');
  assert.strictEqual(grips.length, 1, '默认应有一个拖拽把手');
  const grip = grips[0];
  f3e.contentDocument = makeInnerDoc(500);
  f3e.fire('load');
  assert.strictEqual(f3e.getAttribute('height'), '502');
  grip.fire('pointerdown', { clientX: 100, clientY: 100, pointerId: 1, preventDefault() {} });
  grip.fire('pointermove', { clientX: 180, clientY: 160, pointerId: 1 });
  assert.strictEqual(f3e.style.width, '880px', '拖动改宽：800 + 80');
  assert.strictEqual(f3e.getAttribute('height'), '460', '拖动改高：400 + 60');
  /* 把手要跟着 iframe 实际尺寸走，不能吊在 shell（整栏宽）的右下角。
     假 DOM 的 offsetWidth/offsetHeight 固定是 800×400，所以这里验的是"按 offset 摆放"这件事本身。 */
  assert.strictEqual(grip.style.left, '778px', '把手 x = iframe 实际宽 − 22（把手 18px + 4px 内缩）');
  assert.strictEqual(grip.style.top, '378px', '把手 y = iframe 实际高 − 22');
  f3e.fire('load');
  assert.strictEqual(f3e.getAttribute('height'), '460', '手动尺寸优先于自适应');
  grip.fire('pointerup', { pointerId: 1 });
  grip.fire('dblclick', {});
  assert.strictEqual(f3e.style.width, '100%', '双击恢复满宽');
  assert.strictEqual(f3e.getAttribute('height'), '502', '双击恢复自适应（内容 500px）');
  /* resizable:false 时不挂把手 */
  const c3f = await runView({ file: FIXTURE_FILE, resizable: false });
  assert.strictEqual(c3f.findByClass('frm-embed-grip').length, 0, 'resizable:false 应去掉把手');
  console.log('✓ 3d. 把手：拖宽拖高、手动优先、双击回自适应、可关闭');

  // ---- 3e) 组件自己报高度（postMessage）——不依赖同源、也不依赖渲染帧 ----
  const c3g = await runView({ file: FIXTURE_FILE });
  const f3g = c3g.findByClass('frm-embed-frame')[0];
  f3g.contentWindow = { tag: 'inner' };
  f3g.contentDocument = makeInnerDoc(400);
  f3g.fire('load');
  assert.strictEqual(f3g.getAttribute('height'), '402', '先按内部文档量一次');
  postToView({ tag: 'somebody-else' }, { wg: 'widget-height', height: 999 });
  assert.strictEqual(f3g.getAttribute('height'), '402', '别家窗口发来的高度消息要忽略');
  postToView(f3g.contentWindow, { wg: 'other', height: 999 });
  assert.strictEqual(f3g.getAttribute('height'), '402', '非高度消息要忽略');
  /* 同源时：消息只是"该重新量了"的触发器，高度以现场实测为准（组件报的数只作参考） */
  f3g.contentDocument = makeInnerDoc(700);
  postToView(f3g.contentWindow, { wg: 'widget-height', height: 620 });
  assert.strictEqual(f3g.getAttribute('height'), '702', '同源时重新实测：内容 700 → 702');
  /* 不同源（读不到内部文档）时：只能用组件报来的高度 —— 这正是这一步的意义 */
  const c3j = await runView({ file: FIXTURE_FILE });
  const f3j = c3j.findByClass('frm-embed-frame')[0];
  f3j.contentWindow = { tag: 'sandboxed' };
  f3j.contentDocument = null;
  f3j.fire('load');
  assert.strictEqual(f3j.getAttribute('height'), '420', '读不到内部文档：先给兜底 420');
  postToView({ tag: 'somebody-else' }, { wg: 'widget-height', height: 999 });
  assert.strictEqual(f3j.getAttribute('height'), '420', '别家窗口发来的高度消息要忽略');
  postToView(f3j.contentWindow, { wg: 'other', height: 999 });
  assert.strictEqual(f3j.getAttribute('height'), '420', '非高度消息要忽略');
  postToView(f3j.contentWindow, { wg: 'widget-height', height: 620 });
  assert.strictEqual(f3j.getAttribute('height'), '622', '读不到内部文档时采用组件报的 620 → 622');
  postToView(f3j.contentWindow, { wg: 'widget-height', height: 20000 });
  assert.strictEqual(f3j.getAttribute('height'), '20002', '没有 maxHeight 时不夹上限');
  // 固定高度模式：组件报高度也不该改尺寸
  const c3h = await runView({ file: FIXTURE_FILE, height: 350 });
  const f3h = c3h.findByClass('frm-embed-frame')[0];
  f3h.contentWindow = { tag: 'inner2' };
  f3h.contentDocument = makeInnerDoc(400);
  f3h.fire('load');
  postToView(f3h.contentWindow, { wg: 'widget-height', height: 880 });
  assert.strictEqual(f3h.getAttribute('height'), '350', 'height:350 固定模式下不采纳报来的高度');
  // 手动拖过之后也不采纳
  const c3i = await runView({ file: FIXTURE_FILE });
  const f3i = c3i.findByClass('frm-embed-frame')[0];
  f3i.contentWindow = { tag: 'inner3' };
  f3i.contentDocument = makeInnerDoc(400);
  f3i.fire('load');
  const gripI = c3i.findByClass('frm-embed-grip')[0];
  gripI.fire('pointerdown', { clientX: 0, clientY: 0, pointerId: 2, preventDefault() {} });
  gripI.fire('pointermove', { clientX: 0, clientY: 40, pointerId: 2 });
  postToView(f3i.contentWindow, { wg: 'widget-height', height: 700 });
  assert.strictEqual(f3i.getAttribute('height'), '440', '手动拖过之后不再被组件报的高度覆盖');
  console.log('✓ 3e. postMessage 报高度：认来源、只在自适应模式生效、可被手动尺寸压住');

  // ---- 4) 错误分支都给出 [错误] 且不挂 iframe ----
  const cases = [
    [{}, '缺 file'],
    [{ file: 'http://127.0.0.1:8790/a.html' }, '写了 URL'],
    [{ file: '../secret.html' }, '带 ..'],
    [{ file: '笔记.md' }, '非 html'],
    [{ file: '不存在的组件.html' }, '文件不存在'],
  ];
  for (const [input, label] of cases) {
    const c = await runView(input);
    assert.strictEqual(c.findByClass('frm-embed-frame').length, 0, label + '：不该挂 iframe');
    const errs = c.findByClass('frm-embed-err');
    assert.strictEqual(errs.length, 1, label + '：应有一个错误框');
    assert.ok(errs[0].textContent.indexOf('[错误]') === 0, label + '：应以 [错误] 开头');
    console.log('   · ' + label + ' → ' + errs[0].textContent.slice(0, 40) + '…');
  }
  console.log('✓ 4. 五类错误分支都给出 [错误]，不静默失败');

  // ---- 5) 所有可用组件页全部可内嵌 ----
  for (const n of FIXTURE_REQUESTS) {
    const c = await runView({ file: n, height: 400 });
    const f = c.findByClass('frm-embed-frame');
    assert.strictEqual(f.length, 1, n + ' 应成功');
    /* 体量阈值验收真实组件页（内联了 widgets.js/CSS）；桩页只需非空 */
    if (VAULT) assert.ok(f[0].getAttribute('srcdoc').length > 50000, n + ' 内容过少');
    else assert.ok(f[0].getAttribute('srcdoc').length > 0, n + ' srcdoc 为空');
  }
  if (!VAULT) skip('5-真实组件页体量（>50k）');
  console.log('✓ 5. ' + FIXTURE_REQUESTS.length + ' 个组件页全部可内嵌');

  // ---- 6) obsidian:// 链接接线 ----
  openCalls.length = 0;
  const c6 = await runView({ file: FIXTURE_FILE, height: 400 });
  const f6 = c6.findByClass('frm-embed-frame')[0];
  assert.ok(f6.listeners['load'] && f6.listeners['load'].length === 1, '应挂一个 load 监听');
  const docStub = makeEl('div');
  f6.contentDocument = docStub;
  f6.fire('load');
  assert.ok(docStub.listeners['click'] && docStub.listeners['click'].length >= 1, 'load 后应挂 click 监听');
  assert.ok(docStub.listeners['pointerup'] && docStub.listeners['pointerup'].length === 1,
    'load 后应挂 pointerup 监听（交互后补量一次高度）');

  const link = makeEl('a');
  link.closest = () => link;
  link.getAttribute = () => 'obsidian://open?vault=FRM-Vault&file=Books%2FP1B4%2FCh57.md';
  let prevented = false;
  docStub.fire('click', { target: link, preventDefault: () => { prevented = true; } });
  assert.strictEqual(prevented, true, '应 preventDefault');
  assert.deepStrictEqual(openCalls, ['Books/P1B4/Ch57.md'], '应解码后交给 openLinkText，实际 ' + JSON.stringify(openCalls));
  /* 解码、preventDefault、openLinkText 这几条都是纯逻辑（metadataCache 是 mock），到哪儿都跑。
     只有一件事得真 vault 才能回答：href 指向的那篇笔记是不是真存在（示例 href 只是个解码样本，
     所以这条不写成断言，只在没有 vault 时记一笔）。 */
  if (!VAULT) skip('6-链接指向的源笔记在 vault 里是否存在', '示例 href 指向 Books/P1B4/Ch57.md，本测试用 mock metadataCache');

  const plain = makeEl('a');
  plain.closest = () => null;
  docStub.fire('click', { target: plain, preventDefault: () => {} });
  assert.strictEqual(openCalls.length, 1, '普通点击不该触发打开笔记');
  // 源笔记不在了：不该打开、也不该建空笔记
  openCalls.length = 0;
  metadataResolvable = false;
  const c7 = await runView({ file: FIXTURE_FILE });
  const f7 = c7.findByClass('frm-embed-frame')[0];
  const doc7 = makeEl('div');
  f7.contentDocument = doc7;
  f7.fire('load');
  doc7.fire('click', { target: link, preventDefault: () => {} });
  assert.strictEqual(openCalls.length, 0, '笔记不存在时不该打开/新建');
  metadataResolvable = true;
  console.log('✓ 6. obsidian:// 链接接线正确（含"笔记不存在就不动"分支）');

  /* ---- 7) 每篇笔记里的 dv.view 调用都必须指向真实存在的组件文件（要读真 vault 的 md 与组件页）---- */
  if (VAULT) {
  /* 扫全部 md（跳过 .git/.obsidian/Books/题库/_attachments/_exports/_drafts/_tools）：
     组件入口可以出现在根目录示例、Maps/Notes、Subnotes 或 Maps/_drafts，硬编码一篇笔记
     会让新加的笔记漏检（本测试就是这么发现漏检的）。 */
  const SKIP_DIRS = new Set(['.git', '.obsidian', '.pi', '_meta', 'node_modules', 'Books', '题库', '_attachments', '_exports', '_drafts', '_venv', '_tools']);
  const mdFiles = [];
  (function walk(dir) {
    for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
      if (ent.name.startsWith('.') && ent.name !== '.') continue;
      const full = path.join(dir, ent.name);
      if (ent.isDirectory()) { if (!SKIP_DIRS.has(ent.name)) walk(full); }
      else if (ent.name.endsWith('.md')) mdFiles.push(full);
    }
  })(VAULT);
  let calls = 0, notesWithEmbeds = 0;
  for (const notePath of mdFiles) {
    const note = fs.readFileSync(notePath, 'utf8');
    const found = [...note.matchAll(/dv\.view\("([^"]+)",\s*\{\s*file:\s*"([^"]+\.html)"(?:,\s*height:\s*(\d+))?/g)];
    if (!found.length) continue;
    notesWithEmbeds++;
    for (const [, viewPath, file, height] of found) {
      calls++;
      const where = path.relative(VAULT, notePath);
      const expectedViewPath = TOOLS_DIR + '/widget-embed';
      assert.strictEqual(viewPath, expectedViewPath, where + '：dv.view 路径应是 ' + expectedViewPath + '：' + viewPath);
      const target = file.includes('/') ? file : WIDGETS_DIR + '/' + file;
      assert.ok(fs.existsSync(path.join(VAULT, target)), where + '：组件文件不存在：' + target);
      assert.ok(fs.existsSync(path.join(VAULT, TOOLS_DIR, 'widget-embed/view.js')), 'view.js 不存在');
      assert.ok(fs.existsSync(path.join(VAULT, TOOLS_DIR, 'widget-embed/view.css')), 'view.css 不存在');
      if (height) assert.ok(Number(height) >= 120 && Number(height) <= 2000, where + '：高度越界：' + height);
    }
    assert.ok(!/^```interactive/m.test(note), path.relative(VAULT, notePath) + ' 里不该再残留 interactive 代码块');
  }
  assert.ok(calls >= Math.max(1, REAL_WIDGET_PATHS.length), '项目入口应至少有 ' + Math.max(1, REAL_WIDGET_PATHS.length) + ' 个 dv.view 调用，实际 ' + calls);
  console.log('✓ 7. ' + notesWithEmbeds + ' 篇笔记里共 ' + calls + ' 个 dv.view 调用全部指向存在的组件文件');
  } else {
    skip('7-全 vault 笔记扫描（每篇笔记的 dv.view 都指向存在的组件文件）');
  }

  // ---- 8) 读取抛错时要在笔记里显示 [错误]，不能静默（Dataview 不会替我们等/报错）----
  readFails = true;
  const c8 = await runView({ file: FIXTURE_FILE });
  await new Promise((r) => setTimeout(r, 50));         // 视图内部是 async，等它跑完
  const errs8 = c8.findByClass('frm-embed-err');
  assert.strictEqual(errs8.length, 1, '读取失败应显示一个错误框，实际 ' + errs8.length);
  assert.ok(errs8[0].textContent.indexOf('[错误]') === 0, '应以 [错误] 开头：' + errs8[0].textContent);
  readFails = false;
  console.log('✓ 8. 读取失败会显示 [错误]（不静默）：' + errs8[0].textContent.slice(0, 38) + '…');

  // ---- 9) 兜底：注入样式压掉 72vh 截断；宽度变化时把 resize 补送进 iframe ----
  const c9 = await runView({ file: FIXTURE_FILE });
  const f9 = c9.findByClass('frm-embed-frame')[0];
  const inner9 = makeInnerDoc(600);
  const head9 = makeEl('head');
  inner9.head = head9;
  inner9.createElement = (t) => makeEl(t);
  f9.contentDocument = inner9;
  const sent = [];
  f9.contentWindow = { Event: function (t) { this.type = t; }, dispatchEvent: (ev) => { sent.push(ev && ev.type); } };
  f9.offsetWidth = 800;
  f9.fire('load');
  assert.strictEqual(head9.children.length, 1, '应向组件页注入一段兜底样式');
  assert.strictEqual(head9.children[0].tagName, 'STYLE', '兜底样式应是 <style>');
  assert.ok(/\.wg-figure\{max-height:none/.test(head9.children[0].textContent),
    '兜底样式必须关掉 .wg-figure 的 max-height，实际：' + head9.children[0].textContent);
  assert.strictEqual(f9.getAttribute('height'), '602', 'load 后应按内容高度 600 回设（+2），实际 ' + f9.getAttribute('height'));
  f9.offsetWidth = 500;                                   // 笔记栏被拖窄 / 窗口变小
  fireWin('resize');
  assert.ok(sent.length >= 2, '宽度变化后应补送 resize 进 iframe，实际发出 ' + sent.length + ' 次');
  assert.ok(sent.length <= 6, '补送必须是有界的（每次改宽最多 3 次修正 + 首尾同步），实际 ' + sent.length + ' 次');
  assert.strictEqual(sent[sent.length - 1], 'resize', '补送的事件类型应是 resize');
  console.log('✓ 9. 兜底：注入样式压掉 72vh 截断；宽度变化会把 resize 补送进 iframe');

  // ---- 10) 组件页里印的构建标识（用来人眼确认"我看到的是新生成的组件页"；要读真 vault 的三类文件）----
  if (VAULT) {
  const wjs = fs.readFileSync(path.join(VAULT, TOOLS_DIR, 'widgets.js'), 'utf8');
  const build = /var BUILD = '([^']+)'/.exec(wjs);
  assert.ok(build, 'widgets.js 里应有 BUILD 常量');
  const allW = JSON.parse(fs.readFileSync(path.join(VAULT, TOOLS_DIR, 'widgets-index.json'), 'utf8')).widgets;
  for (const w of allW) {
    const html = fs.readFileSync(path.join(VAULT, w.html), 'utf8');
    assert.ok(html.indexOf("var BUILD = '" + build[1] + "'") >= 0,
      w.html + ' 里的 widgets.js 副本不是当前构建（应含 BUILD=' + build[1] + '）');
  }
  console.log('✓ 10. ' + allW.length + ' 个组件页内联的 widgets.js 都是当前构建：' + build[1]);
  } else {
    skip('10-组件页内联 widgets.js 的构建版本一致', '要读 vault 的 Maps/_tools/widgets.js、widgets-index.json 与各组件页');
  }
  // ---- 11) 精简模式（默认）与 chrome:"full" 开关 ----
  const c11 = await runView({ file: FIXTURE_FILE });
  const f11 = c11.findByClass('frm-embed-frame')[0];
  const d11 = f11.getAttribute('srcdoc');
  assert.ok(/<meta name="wg-chrome" content="slim">/.test(d11), '默认注入 slim');
  const inner11 = makeInnerDoc(600);
  const head11 = makeEl('head');
  inner11.head = head11;
  inner11.createElement = (t) => makeEl(t);
  f11.contentDocument = inner11;
  f11.fire('load');
  assert.ok(/\.wg-sub,\.wg-src,\.wg-notes,\.wg-foot\{display:none/.test(head11.children[0].textContent),
    '精简模式要注入兜底样式把解释性文字收起来，实际：' + head11.children[0].textContent);
  const c12 = await runView({ file: FIXTURE_FILE, chrome: 'full' });
  const f12 = c12.findByClass('frm-embed-frame')[0];
  assert.ok(/<meta name="wg-chrome" content="full">/.test(f12.getAttribute('srcdoc')), 'chrome:"full" 应注入 full');
  const inner12 = makeInnerDoc(600);
  const head12 = makeEl('head');
  inner12.head = head12;
  inner12.createElement = (t) => makeEl(t);
  f12.contentDocument = inner12;
  f12.fire('load');
  assert.ok(head12.children[0].textContent.indexOf('display:none') < 0,
    '完整模式不该藏任何东西：' + head12.children[0].textContent);
  console.log('✓ 11. 精简模式：默认注入 wg-chrome=slim + 兜底隐藏样式；chrome:"full" 可要回完整版');

  if (skipped.length) {
    console.log('（本次跳过 ' + skipped.length + ' 项依赖 vault 内容的断言：' + skipped.join('、') + '）');
  }
  console.log('\nDataview 视图测试全部通过。');
  process.exit(0);
})().catch((e) => {
  console.error('\nDataview 视图测试失败：', e && e.message);
  process.exit(1);
});
