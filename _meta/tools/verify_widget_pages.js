#!/usr/bin/env node
'use strict';
/**
 * verify_widget_pages.js —— 交互组件质量门禁。
 *
 * 这里不是浏览器截图测试：用 widgets.js 的真运行时和最小 DOM 重绘同一张 spec，
 * 对 semantic controls 的默认/边界值比较主图指纹。指纹只取图区的绘制几何、
 * canvas 调用和 renderer 的绘制状态；readouts、markers、标题与说明文字不能单独
 * 让一个只读数的控件通过。
 *
 * 用法：
 *   node _meta/tools/verify_widget_pages.js
 *   node _meta/tools/verify_widget_pages.js _widgets/a.json ...
 *   node _meta/tools/verify_widget_pages.js --root /别的/项目
 *   node _meta/tools/verify_widget_pages.js --quiet
 *   node _meta/tools/verify_widget_pages.js --strict       # Plotly 也必须真实浏览器验收
 *   node _meta/tools/verify_widget_pages.js --allow-plotly-skip
 *
 * 默认从 <root>/widgets.config.json 读取 widgetsDir（默认 _widgets）。Plotly 不在
 * 假 DOM 中冒充通过：默认输出 SKIP；--strict 下存在 Plotly spec 就失败，提示用
 * 真实浏览器验收。退出码 0 = 没有质量门禁失败（允许 Plotly skip）。
 */

const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const HERE = __dirname;
const argv = process.argv.slice(2);
let root = path.dirname(path.dirname(HERE));
let quiet = false;
let strict = false;
let allowPlotlySkip = true;
const explicit = [];
for (let i = 0; i < argv.length; i++) {
  const a = argv[i];
  if (a === '--root') {
    if (i + 1 >= argv.length) { console.error('[错误] --root 需要项目根路径'); process.exit(1); }
    root = path.resolve(argv[++i]);
    continue;
  }
  if (a === '--quiet' || a === '-q') { quiet = true; continue; }
  if (a === '--strict') { strict = true; continue; }
  if (a === '--allow-plotly-skip') { allowPlotlySkip = true; continue; }
  if (a === '--help' || a === '-h') {
    console.log(fs.readFileSync(__filename, 'utf8').split('*/')[0].replace(/^#![^\n]*\n/, ''));
    process.exit(0);
  }
  explicit.push(a);
}
root = path.resolve(root);

function readLayout() {
  const cfgPath = path.join(root, 'widgets.config.json');
  let cfg = {};
  let configError = null;
  if (fs.existsSync(cfgPath)) {
    try {
      cfg = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
      if (!cfg || Array.isArray(cfg) || typeof cfg !== 'object') {
        configError = 'widgets.config.json 顶层必须是对象';
        cfg = {};
      }
    } catch (e) {
      configError = 'widgets.config.json 不是合法 JSON：' + e.message;
      cfg = {};
    }
  }
  const toolsRel = typeof cfg.toolsDir === 'string' && cfg.toolsDir.trim()
    ? cfg.toolsDir : path.relative(root, HERE);
  const widgetsRel = typeof cfg.widgetsDir === 'string' && cfg.widgetsDir.trim()
    ? cfg.widgetsDir : '_widgets';
  return {
    configError,
    toolsDir: path.resolve(root, toolsRel),
    widgetsDir: path.resolve(root, widgetsRel),
  };
}

const lay = readLayout();
let specs;
if (explicit.length) {
  specs = explicit.map((p) => path.resolve(process.cwd(), p));
} else {
  specs = fs.existsSync(lay.widgetsDir)
    ? fs.readdirSync(lay.widgetsDir).filter((f) => f.endsWith('.json')).sort()
      .map((f) => path.join(lay.widgetsDir, f))
    : [];
}
if (!specs.length) {
  if (lay.configError) console.error('[错误] ' + lay.configError);
  console.log('没有找到任何 spec（目录：%s）', lay.widgetsDir);
  process.exit(lay.configError ? 1 : 0);
}

/* ── 最小 DOM 与 canvas 记录器 ─────────────────────────────────────────── */
class Ctx2d {
  constructor() { this.calls = []; }
  _log(name) { this.calls.push([name].concat(Array.prototype.slice.call(arguments, 1))); }
  setTransform() { this._log('setTransform', ...arguments); }
  clearRect() { this._log('clearRect', ...arguments); }
  fillRect() { this._log('fillRect', ...arguments); }
  beginPath() { this._log('beginPath', ...arguments); }
  closePath() { this._log('closePath', ...arguments); }
  moveTo() { this._log('moveTo', ...arguments); }
  lineTo() { this._log('lineTo', ...arguments); }
  arc() { this._log('arc', ...arguments); }
  fill() { this._log('fill', ...arguments); }
  stroke() { this._log('stroke', ...arguments); }
  fillText() { this._log('fillText', ...arguments); }
  save() { this._log('save'); }
  restore() { this._log('restore'); }
  translate() { this._log('translate', ...arguments); }
  scale() { this._log('scale', ...arguments); }
  rect() { this._log('rect', ...arguments); }
  measureText(s) { return { width: String(s).length * 6.6 }; }
}

function classParts(el) {
  return String(el.className || '').split(/\s+/).filter(Boolean);
}
function setClassParts(el, parts) { el.className = Array.from(new Set(parts)).join(' '); }
function selectorMatch(el, selector) {
  selector = String(selector || '').trim();
  if (!selector || !el) return false;
  if (selector.charAt(0) === '#') return el.id === selector.slice(1);
  if (selector.charAt(0) === '.') return classParts(el).indexOf(selector.slice(1)) >= 0;
  const tagAttr = /^([A-Za-z][A-Za-z0-9-]*)?(?:\[([^=\]]+)(?:=["']?([^\]"']+)["']?)?\])?$/.exec(selector);
  if (!tagAttr) return String(el.tagName).toLowerCase() === selector.toLowerCase();
  if (tagAttr[1] && String(el.tagName).toLowerCase() !== tagAttr[1].toLowerCase()) return false;
  if (!tagAttr[2]) return true;
  const v = el.getAttribute(tagAttr[2]);
  return v !== undefined && (tagAttr[3] === undefined || String(v) === tagAttr[3]);
}
function walk(el, fn) {
  if (!el) return null;
  if (fn(el)) return el;
  for (const child of el.children || []) {
    const found = walk(child, fn);
    if (found) return found;
  }
  return null;
}
function walkAll(el, fn, out) {
  if (!el) return out;
  if (fn(el)) out.push(el);
  (el.children || []).forEach((child) => walkAll(child, fn, out));
  return out;
}

class Element {
  constructor(tag) {
    this.tagName = String(tag || 'div').toLowerCase();
    this.children = [];
    this.parentNode = null;
    this.style = {};
    this.attrs = {};
    this.events = {};
    this.className = '';
    this.textContent = '';
    this.hidden = false;
    this.id = '';
    this.type = '';
    this.value = '';
    this.checked = false;
    this.min = '';
    this.max = '';
    this.step = '';
    this.clientWidth = 640;
    this.clientHeight = 360;
    this.isConnected = true;
    this.__contexts = [];
    this.classList = {
      add: (...names) => setClassParts(this, classParts(this).concat(names)),
      remove: (...names) => setClassParts(this, classParts(this).filter((x) => names.indexOf(x) < 0)),
      toggle: (name, force) => {
        const has = classParts(this).indexOf(name) >= 0;
        const on = force === undefined ? !has : !!force;
        if (on && !has) this.classList.add(name);
        if (!on && has) this.classList.remove(name);
        return on;
      },
      contains: (name) => classParts(this).indexOf(name) >= 0,
    };
  }
  appendChild(child) {
    if (!child) return child;
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    child.isConnected = this.isConnected !== false;
    this.children.push(child);
    return child;
  }
  removeChild(child) {
    const i = this.children.indexOf(child);
    if (i >= 0) { this.children.splice(i, 1); child.parentNode = null; child.isConnected = false; }
    return child;
  }
  insertBefore(child, ref) {
    if (!child) return child;
    if (child.parentNode) child.parentNode.removeChild(child);
    const i = ref ? this.children.indexOf(ref) : -1;
    child.parentNode = this;
    child.isConnected = this.isConnected !== false;
    if (i < 0) this.children.push(child); else this.children.splice(i, 0, child);
    return child;
  }
  get firstChild() { return this.children[0]; }
  setAttribute(key, value) {
    this.attrs[key] = String(value);
    if (key === 'class') this.className = String(value);
    if (key === 'id') this.id = String(value);
  }
  getAttribute(key) { return this.attrs[key]; }
  addEventListener(key, fn) {
    if (!this.events[key]) this.events[key] = [];
    this.events[key].push(fn);
  }
  removeEventListener(key, fn) {
    this.events[key] = (this.events[key] || []).filter((x) => x !== fn);
  }
  dispatchEvent(ev) {
    const e = ev || { target: this };
    if (!e.target) e.target = this;
    (this.events[e.type] || []).slice().forEach((fn) => fn.call(this, e));
  }
  querySelector(selector) { return walk(this, (el) => selectorMatch(el, selector)); }
  querySelectorAll(selector) { return walkAll(this, (el) => selectorMatch(el, selector), []); }
  closest(selector) {
    let el = this;
    while (el) { if (selectorMatch(el, selector)) return el; el = el.parentNode; }
    return null;
  }
  getBoundingClientRect() {
    let w = Number(this.clientWidth) || 640;
    let h = Number(this.clientHeight) || 360;
    if (this.style.width && /px$/.test(String(this.style.width))) w = Number.parseFloat(this.style.width) || w;
    if (this.style.height && /px$/.test(String(this.style.height))) h = Number.parseFloat(this.style.height) || h;
    const view = this.getAttribute('viewBox');
    if (this.tagName === 'svg' && view) {
      const p = String(view).trim().split(/[ ,]+/).map(Number);
      if (p.length === 4 && p[2] > 0 && p[3] > 0) h = w * p[3] / p[2];
    }
    return { left: 0, top: 0, right: w, bottom: h, width: w, height: h };
  }
  getContext(kind) {
    if (this.tagName !== 'canvas' || kind !== '2d') return null;
    if (!this.__contexts.length) this.__contexts.push(new Ctx2d());
    return this.__contexts[0];
  }
  focus() { global.document.activeElement = this; }
  select() {}
  setPointerCapture() {}
}

class DocumentStub {
  constructor() {
    this.documentElement = new Element('html');
    this.body = new Element('body');
    this.documentElement.appendChild(this.body);
    this.activeElement = null;
    this.hidden = false;
    this.readyState = 'complete';
    this.events = {};
  }
  createElement(tag) { return new Element(tag); }
  createElementNS(_ns, tag) { return new Element(tag); }
  getElementById(id) { return walk(this.documentElement, (el) => el.id === id); }
  querySelector(selector) { return walk(this.documentElement, (el) => selectorMatch(el, selector)); }
  querySelectorAll(selector) { return walkAll(this.documentElement, (el) => selectorMatch(el, selector), []); }
  addEventListener(key, fn) {
    if (!this.events[key]) this.events[key] = [];
    this.events[key].push(fn);
  }
}

const documentStub = new DocumentStub();
const windowStub = {
  document: documentStub,
  addEventListener() {},
  removeEventListener() {},
  matchMedia() { return { matches: false }; },
  getComputedStyle() { return { getPropertyValue() { return ''; } }; },
  devicePixelRatio: 1,
  self: null,
  top: null,
  parent: null,
};
windowStub.self = windowStub;
windowStub.top = windowStub;
windowStub.parent = windowStub;
global.document = documentStub;
global.window = windowStub;
global.__wgNoAuto = true;
windowStub.__wgNoAuto = true;

const runtimePath = fs.existsSync(path.join(lay.toolsDir, 'widgets.js'))
  ? path.join(lay.toolsDir, 'widgets.js') : path.join(HERE, 'widgets.js');
try { require(runtimePath); } catch (e) {
  console.error('[错误] 无法加载 widgets.js：' + e.message);
  process.exit(1);
}
const WG = windowStub.WG;
if (!WG || typeof WG.render !== 'function') {
  console.error('[错误] widgets.js 没有导出 WG.render');
  process.exit(1);
}

/* ── teaching / 主绘制字段门禁（与 make_widget.py 的静态口径保持一致） ── */
const CAMERA_KEYS = new Set(['azimuth', 'elevation', 'zoom']);
const CONTROL_TYPES = new Set(['slider', 'number', 'toggle', 'select']);
const REQUIRED_TEACHING = ['question', 'sourceSection', 'controlEffect', 'visualEvidence'];
function collectStrings(value, out) {
  if (typeof value === 'string') out.push(value);
  else if (Array.isArray(value)) value.forEach((x) => collectStrings(x, out));
  else if (value && typeof value === 'object') Object.keys(value).forEach((k) => collectStrings(value[k], out));
}
function addFields(holder, fields, out) {
  if (!holder || typeof holder !== 'object' || Array.isArray(holder)) return;
  fields.forEach((field) => { if (Object.prototype.hasOwnProperty.call(holder, field)) collectStrings(holder[field], out); });
}
function mainDrawStrings(spec, kind) {
  const out = [];
  const addSeries = () => (Array.isArray(spec.series) ? spec.series : [])
    .forEach((x) => addFields(x, ['expr', 'points'], out));
  if (kind === 'plot' || kind === 'scatter') addSeries();
  else if (kind === 'bars') (spec.bars || []).forEach((x) => addFields(x, ['value'], out));
  else if (kind === 'histogram') addFields(spec.histogram, ['sample', 'values'], out);
  else if (['box', 'ecdf', 'qq'].indexOf(kind) >= 0) addFields(spec[kind], ['sample', 'values'], out);
  else if (kind === 'heatmap') addFields(spec.heat, ['values'], out);
  else if (kind === 'timeline') (spec.timeline && spec.timeline.items || [])
    .forEach((x) => addFields(x, ['t', 'amount'], out));
  else if (kind === 'tree') {
    const addTree = (node) => {
      if (!node || typeof node !== 'object') return;
      addFields(node, ['value', 't', 'amount'], out);
      (node.children || []).forEach(addTree);
    };
    addTree(spec.tree && spec.tree.root);
  } else if (kind === 'vector') addFields(spec.vector, ['u', 'v'], out);
  else if (kind === 'matrix') addFields(spec.matrix, ['values'], out);
  else if (kind === 'regression') {
    collectStrings(spec.points, out); addFields(spec.data, ['x', 'y', 'cls'], out);
    const f = spec.fit;
    if (f && (f.mode === 'manual' || f.mode === 'compare')) {
      addFields(f, ['slope', 'intercept', 'slopeKey', 'interceptKey'], out);
      if (!Object.prototype.hasOwnProperty.call(f, 'slopeKey')) out.push('slope');
      if (!Object.prototype.hasOwnProperty.call(f, 'interceptKey')) out.push('intercept');
    }
  } else if (kind === 'pca') {
    collectStrings(spec.points, out); addFields(spec.data, ['x', 'y', 'cls'], out);
    if (spec.candidate && typeof spec.candidate === 'object') {
      addFields(spec.candidate, ['angle', 'angleControl'], out);
      if (!Object.prototype.hasOwnProperty.call(spec.candidate, 'angleControl')) out.push('theta');
    }
  } else if (kind === 'descent') {
    addFields(spec.contour, ['expr'], out); addFields(spec.grad, ['dfdx', 'dfdy'], out);
    if (spec.descent && typeof spec.descent === 'object') {
      addFields(spec.descent, ['lr', 'steps', 'lrKey', 'stepsKey'], out);
      if (!Object.prototype.hasOwnProperty.call(spec.descent, 'lrKey')) out.push('lr');
      if (!Object.prototype.hasOwnProperty.call(spec.descent, 'stepsKey')) out.push('steps');
    }
  } else if (kind === 'contour') addFields(spec.contour, ['expr'], out);
  else if (kind === 'surface3d') {
    addFields(spec.surface, ['expr', 'points'], out); collectStrings(spec.path, out);
    ['adaboost', 'boosting'].forEach((name) => {
      if (spec[name] && typeof spec[name] === 'object') addFields(spec[name], ['errorModeKey', 'shrinkageKey', 'nuKey', 'roundsKey', 'targetKey'], out);
    });
  } else if (kind === 'treefit') {
    collectStrings(spec.points, out); addFields(spec.data, ['x', 'y', 'cls'], out);
    addFields(spec.treefit, ['splits'], out);
  } else if (kind === 'custom') addFields(spec.custom, ['js'], out);
  else if (kind === 'plotly') {
    addFields(spec.plotly, ['data'], out);
    if (spec.plotly && spec.plotly.layout && typeof spec.plotly.layout === 'object') {
      Object.keys(spec.plotly.layout).forEach((k) => { if (k !== 'title') collectStrings(spec.plotly.layout[k], out); });
    }
  }
  return out;
}
function hasIdentifier(text, key) {
  return new RegExp('(?<![A-Za-z0-9_$])' + key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '(?![A-Za-z0-9_$])').test(text);
}
function teachingInfo(spec) {
  const errors = [], warnings = [];
  const t = spec && spec.teaching;
  if (!t || typeof t !== 'object' || Array.isArray(t)) {
    REQUIRED_TEACHING.forEach((k) => errors.push('teaching.' + k + ' 必填'));
    return { errors, warnings, semantic: [], dynamic: false };
  }
  REQUIRED_TEACHING.forEach((k) => {
    const v = t[k];
    const ok = k === 'controlEffect'
      ? ((typeof v === 'string' && v.trim()) ||
         (Array.isArray(v) && v.length && v.every((x) => typeof x === 'string' && x.trim())) ||
         (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length &&
          Object.keys(v).every((x) => x.trim() && typeof v[x] === 'string' && v[x].trim())))
      : (typeof v === 'string' && v.trim());
    if (!ok) errors.push(k === 'controlEffect'
      ? 'teaching.controlEffect 必须是非空字符串、字符串数组或“控件 key → 作用”对象' : 'teaching.' + k + ' 必须是非空字符串');
  });
  const controls = Array.isArray(spec.controls) ? spec.controls : [];
  const known = new Set(controls.filter((c) => c && typeof c.key === 'string' && CONTROL_TYPES.has(c.type)).map((c) => c.key));
  function names(field) {
    if (t[field] === undefined) return null;
    if (!Array.isArray(t[field]) || t[field].some((x) => typeof x !== 'string' || !x.trim())) {
      errors.push('teaching.' + field + ' 若给出，必须是字符串数组');
      return [];
    }
    const out = [];
    t[field].forEach((x) => {
      if (out.indexOf(x) >= 0) errors.push('teaching.' + field + ' 不能重复列出控件 key=' + x);
      else out.push(x);
    });
    return out;
  }
  const listed = names('controls');
  const extraCamera = names('cameraControls');
  const camera = new Set(CAMERA_KEYS);
  (extraCamera || []).forEach((x) => camera.add(x));
  (extraCamera || []).forEach((x) => { if (!known.has(x)) errors.push('teaching.cameraControls 列出了不存在的控件 key=' + x); });
  (listed || []).forEach((x) => { if (!known.has(x)) errors.push('teaching.controls 列出了不存在的控件 key=' + x); });
  const semantic = new Set(listed === null ? Array.from(known) : listed.filter((x) => known.has(x)));
  camera.forEach((x) => semantic.delete(x));
  const dynamic = t.dynamicRenderer === true;
  if (t.dynamicRenderer !== undefined && typeof t.dynamicRenderer !== 'boolean') {
    errors.push('teaching.dynamicRenderer 若给出，必须是 true/false');
  }
  const source = mainDrawStrings(spec, spec && spec.kind).join('\n');
  semantic.forEach((key) => {
    if (hasIdentifier(source, key)) return;
    const msg = '控件 ' + key + ' 没有进入主绘制数据（只在 readouts/markers/title 等非主绘制字段中出现）';
    if (dynamic) warnings.push(msg + '，交给动态指纹比较'); else errors.push(msg + '，请修正主绘制表达式');
  });
  return { errors, warnings, semantic: Array.from(semantic), dynamic };
}

/* ── 指纹：只保留主图区证据，不让 marker/readout 代替绘制数据 ────────── */
const EXCLUDED = new Set([
  'wg-marker', 'wg-mk-label', 'wg-cursor', 'wg-cursor-line', 'wg-cursor-dot',
  'wg-tick', 'wg-gline', 'wg-axis', 'wg-zero', 'wg-s3-edge', 'wg-s3-ticks', 'wg-s3-axlabel',
]);
const GEOMETRY_TAGS = new Set(['path', 'circle', 'rect', 'line', 'polyline', 'polygon', 'text', 'canvas']);
function excludedAncestor(el) {
  let p = el;
  while (p) {
    if (classParts(p).some((x) => EXCLUDED.has(x))) return true;
    p = p.parentNode;
  }
  return false;
}
function finiteRepr(v) {
  if (typeof v === 'number' && !Number.isFinite(v)) return String(v);
  if (Array.isArray(v)) return v.map(finiteRepr);
  if (v && typeof v === 'object') {
    const out = {};
    Object.keys(v).sort().forEach((k) => { if (k !== 'parentNode') out[k] = finiteRepr(v[k]); });
    return out;
  }
  if (typeof v === 'function') return '[function]';
  return v;
}
function geometryFingerprint(it) {
  const records = [];
  const figure = it && it.el && it.el.figure;
  function visit(el) {
    if (!el || excludedAncestor(el)) return;
    const classes = classParts(el);
    const tag = String(el.tagName || '').toLowerCase();
    if (GEOMETRY_TAGS.has(tag)) {
      const attrs = {};
      Object.keys(el.attrs || {}).sort().forEach((k) => { attrs[k] = el.attrs[k]; });
      const style = {};
      Object.keys(el.style || {}).sort().forEach((k) => { style[k] = el.style[k]; });
      const rec = { tag, class: classes, attrs, style, text: String(el.textContent || '') };
      if (tag === 'canvas') rec.calls = (el.__contexts || []).map((c) => c.calls);
      records.push(rec);
    }
    (el.children || []).forEach(visit);
  }
  visit(figure);
  const state = {
    renderMode: it && it.renderMode,
    plotGeom: finiteRepr(it && it.plotGeom),
    plotData: finiteRepr(it && it.plotData),
  };
  const raw = JSON.stringify({ records, state });
  return {
    raw,
    hash: crypto.createHash('sha256').update(raw).digest('hex').slice(0, 12),
    evidence: records.filter((x) => x.tag !== 'text' || x.text).length,
    records: records.length,
  };
}
function readoutText(it) {
  const out = [];
  const cards = it && it.el && it.el.readouts && it.el.readouts.children || [];
  cards.forEach((card) => {
    const label = card.children[0] && card.children[0].textContent || '读数';
    const value = card.children[1] && card.children[1].textContent || '';
    out.push(label.replace(/：$/, '') + '=' + value);
  });
  return out;
}

function seededRandom(seed) {
  let x = seed >>> 0;
  return function () {
    x += 0x6D2B79F5;
    let t = x;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function selectValue(c, raw) {
  if (!Array.isArray(c.options)) return raw;
  const numeric = c.options.length > 0 && c.options.every((o) => {
    const v = Array.isArray(o) ? o[0] : o;
    return v !== null && v !== undefined && v !== '' && Number.isFinite(Number(v));
  });
  return numeric ? Number(raw) : String(raw);
}
function renderSnapshot(spec, overrides) {
  const oldRandom = Math.random;
  Math.random = seededRandom(12345);
  let it;
  try {
    const rootEl = new Element('main');
    it = WG.render(rootEl, { spec, meta: {}, noAuto: true });
    Math.random = seededRandom(12345);
    Object.keys(overrides || {}).forEach((key) => {
      const c = (spec.controls || []).find((x) => x && x.key === key) || {};
      let v = overrides[key];
      if (c.type === 'toggle') v = !!v;
      else if (c.type === 'select') v = selectValue(c, v);
      it.values[key] = v;
      it.pendingKey = key;
    });
    it.redraw();
    const fp = geometryFingerprint(it);
    return { it, fp, errors: (it.errors || []).slice(), readouts: readoutText(it) };
  } catch (e) {
    return { it, fp: null, errors: ['渲染抛异常：' + (e && e.message || e)], readouts: [] };
  } finally {
    Math.random = oldRandom;
  }
}
function statesFor(c) {
  if (c.type === 'toggle') return [false, true];
  if (c.type === 'select') return (c.options || []).map((o) => Array.isArray(o) ? o[0] : o);
  const out = [];
  [c.value, c.min, c.max].forEach((v) => {
    if (v !== undefined && out.indexOf(v) < 0) out.push(v);
  });
  return out;
}
function runSpec(spec) {
  const info = teachingInfo(spec);
  if (info.errors.length) return {
    status: 'FAIL', why: info.errors.slice(0, 3).join('；'), details: '静态 teaching/semantic 门禁拒绝', readouts: []
  };
  const base = renderSnapshot(spec, {});
  if (base.errors.length) return {
    status: 'FAIL', why: base.errors.slice(0, 3).join('；'), details: '默认值渲染失败', readouts: base.readouts
  };
  if (!base.fp || base.fp.evidence <= 0) return {
    status: 'FAIL', why: '没有足够可比较的主绘制内容（不能只用节点数或 readouts 判通过）',
    details: base.fp ? ('基线 ' + base.fp.hash + '，主图证据 0') : '无主图指纹', readouts: base.readouts
  };
  const changes = [];
  let failedControl = null;
  for (const c of (spec.controls || [])) {
    if (!c || info.semantic.indexOf(c.key) < 0) continue;
    const states = statesFor(c);
    const hashes = [];
    let stateError = null;
    states.forEach((v) => {
      const snap = renderSnapshot(spec, { [c.key]: v });
      if (snap.errors.length && !stateError) stateError = snap.errors[0];
      if (snap.fp) hashes.push(snap.fp.hash);
    });
    const unique = Array.from(new Set(hashes));
    const changed = hashes.filter((x) => x !== base.fp.hash).length;
    changes.push(c.key + ': ' + changed + '/' + states.length + ' 个状态改变（' + unique.join(',') + '）');
    if (stateError && !failedControl) failedControl = c.key + ' 重绘失败：' + stateError;
    if (unique.length < 2 && !failedControl) {
      failedControl = '控件 ' + c.key + ' 没有改变主图指纹（只改变 readout/marker 或未接入主绘制数据）';
    }
  }
  if (failedControl) return {
    status: 'FAIL', why: failedControl, details: '基线 ' + base.fp.hash + '；' + changes.join(' ｜ '), readouts: base.readouts
  };
  return {
    status: 'ok', why: '', details: '基线 ' + base.fp.hash + (changes.length ? '；' + changes.join(' ｜ ') : '；无 semantic 控件，主图证据 ' + base.fp.evidence),
    readouts: base.readouts
  };
}

/* ── 逐 spec 执行 ──────────────────────────────────────────────────────── */
const rows = [];
let failed = 0;
for (const file of specs) {
  const rel = path.relative(root, file).split(path.sep).join('/');
  let spec;
  try { spec = JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch (e) {
    rows.push({ rel, kind: '?', status: 'FAIL', why: '不是合法 JSON：' + e.message });
    failed++;
    continue;
  }
  if (spec && spec.kind === 'plotly') {
    const strictFailure = strict || !allowPlotlySkip;
    rows.push({ rel, kind: 'plotly', status: strictFailure ? 'FAIL' : 'SKIP',
      why: strictFailure ? 'Plotly 存在：--strict 要求用真实浏览器验收' : 'Plotly 需要真实浏览器验收，假 DOM 不冒充通过' });
    if (strictFailure) failed++;
    continue;
  }
  const result = runSpec(spec || {});
  if (result.status === 'FAIL') failed++;
  rows.push(Object.assign({ rel, kind: spec && spec.kind || '?' }, result));
}

/* ── 输出（保留旧的「自检：N 个 spec｜通过…」格式） ───────────────────── */
const pad = (s, n) => (String(s).length >= n ? String(s).slice(-n) : String(s) + ' '.repeat(n - String(s).length));
const longest = Math.max(...rows.map((r) => r.rel.length), 1);
for (const r of rows) {
  if (quiet && r.status === 'ok') continue;
  const mark = r.status === 'ok' ? '✓' : (r.status === 'SKIP' ? '·' : '✗');
  console.log('%s  %s  %s  %s', mark, pad(r.rel, longest), pad(r.kind, 10),
    r.status === 'ok' ? '通过' : r.why);
  if (!quiet && r.details) console.log('    指纹：%s', r.details);
  if (!quiet && r.readouts && r.readouts.length) console.log('    读数：%s', r.readouts.join(' ｜ '));
}
console.log('');
console.log('自检：%d 个 spec｜通过 %d、失败 %d、跳过 %d',
  rows.length, rows.filter((r) => r.status === 'ok').length, failed,
  rows.filter((r) => r.status === 'SKIP').length);
process.exit(failed ? 1 : 0);
