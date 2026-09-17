'use strict';
// Minimal DOM: execute the real public WG.render path without browser dependencies.
const assert = require('node:assert/strict');
/* 2d 上下文的替身：只记录绘制调用（不真的画）。有它才谈得上"canvas 路径到底画了几笔"——
   断言 arc/fillRect/beginPath 的次数比只断言 renderMode 有说服力得多。 */
class Ctx2d {
  constructor(canvas) { this.canvas = canvas; this.calls = []; }
  _log(name, args) { this.calls.push([name].concat(Array.from(args))); }
  setTransform() { this._log('setTransform', arguments); }
  clearRect() { this._log('clearRect', arguments); }
  fillRect() { this._log('fillRect', arguments); }
  beginPath() { this._log('beginPath', arguments); }
  closePath() { this._log('closePath', arguments); }
  moveTo() { this._log('moveTo', arguments); }
  lineTo() { this._log('lineTo', arguments); }
  arc() { this._log('arc', arguments); }
  fill() { this._log('fill', arguments); }
  stroke() { this._log('stroke', arguments); }
  fillText() { this._log('fillText', arguments); }
  /* 等宽字体下同长度的数字串宽度相同，所以运行时按长度缓存 —— 记调用次数就能验到那份缓存 */
  measureText(s) { this._log('measureText', [s]); return {width: String(s).length * 6.6}; }
  count(name) { return this.calls.filter(c => c[0] === name).length; }
}
class Element {
  constructor(tag) {
    this.tagName = tag; this.children = []; this.style = {}; this.attrs = {}; this.events = {};
    this.className = ''; this.clientWidth = 640; this.clientHeight = 360; this.parentNode = null;
    this.classList = {add() {}, toggle() {}};
  }
  /* parentNode 必须真的维护：渲染器用 `node.parentNode` 判断"这个 DOM 还在不在图区里"
     （就地重绘时靠它决定要不要复用），假 DOM 少了它就会静默走错分支。 */
  appendChild(e) {
    if (e && e.parentNode) e.parentNode.removeChild(e);
    if (e) e.parentNode = this;
    this.children.push(e);
    return e;
  }
  removeChild(e) {
    const i = this.children.indexOf(e);
    if (i >= 0) { this.children.splice(i, 1); if (e) e.parentNode = null; }
    return e;
  }
  /* insertBefore 也要有：canvas 后端就是靠它把画布插到 <svg> 前面（真实 DOM 里必然有） */
  insertBefore(e, ref) {
    const i = ref ? this.children.indexOf(ref) : -1;
    if (e && e.parentNode) e.parentNode.removeChild(e);
    if (i < 0) return this.appendChild(e);
    e.parentNode = this;
    this.children.splice(i, 0, e);
    return e;
  }
  get firstChild() { return this.children[0]; }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(k, fn) { this.events[k] = fn; }
  querySelector() { return null; }
  /* canvas 后端要用的两个接口。矩形默认给一个非零尺寸（真浏览器里它来自排版）：
     这样"canvas 路径"在假 DOM 里真的会被走到，而想验降级就把它们换掉（见 withoutCanvas /
     nullCanvas / withoutRects 三个助手）。 */
  getBoundingClientRect() {
    return {left: 0, top: 0, right: this.clientWidth, bottom: this.clientHeight,
      width: this.clientWidth, height: this.clientHeight};
  }
  getContext(type) {
    if (type !== '2d') return null;
    if (!this.__ctx) this.__ctx = new Ctx2d(this);
    return this.__ctx;
  }
  focus() {}
}
global.document = {createElement: t => new Element(t), createElementNS: (_, t) => new Element(t)};
global.window = {addEventListener() {}};
require('./widgets.js');
const {WG} = global.window;
function render(spec, meta) { return WG.render(new Element('main'), {spec, meta: meta || {}}); }
function draw(spec) { const it = render(spec); assert.deepEqual(it.errors, []); return it; }
/* mock DOM：SVG 节点（sv）把 class 放在 el.attrs.class，HTML 节点（mk）放在 el.className，两边都要认 */
function findAll(el, cls, out) {
  out = out || [];
  const c = (el.attrs && el.attrs.class) ? String(el.attrs.class).split(/\s+/) : [];
  const k = el.className ? String(el.className).split(/\s+/) : [];
  if (c.indexOf(cls) >= 0 || k.indexOf(cls) >= 0) out.push(el);
  (el.children || []).forEach(ch => findAll(ch, cls, out));
  return out;
}
/* 新渲染器把 plotGeom（像素↔数据的线性映射）留在实例上，用它把图上位置反解成数据值 */
function xData(g, px) { return g.xmin + (px - g.l) / g.pw * (g.xmax - g.xmin); }
function yData(g, py) { return g.ymin + (g.t + g.ph - py) / g.ph * (g.ymax - g.ymin); }
function near(actual, expected, tol, what) {
  assert.ok(Math.abs(actual - expected) <= (tol === undefined ? 1e-3 : tol),
    `${what || 'value'}：实际 ${actual}，应≈ ${expected}`);
}
const stat = draw({kind: 'bars', bars: [{label: 'x', value: '1'}]});
const ev = expr => stat.engine.eval(expr);
assert.equal(ev('quantile([5], .5)'), 5);
assert.equal(ev('quantile([1,3], .5)'), 2);
assert.equal(ev('quantile([1,3], 0)'), 1);
assert.equal(ev('quantile([1,3], 1)'), 3);
assert.ok(Number.isNaN(ev('quantile([], .5)')));
assert.equal(ev('mean([null, "", "  ", false, [], 5])'), 5);
assert.ok(Number.isNaN(ev('sd([])')));
assert.ok(Number.isNaN(ev('sd([5])')));
assert.equal(ev('sd([5,5])'), 0);
assert.equal(ev('sd([1,3])'), Math.sqrt(2));
const hist = draw({kind: 'histogram', histogram: {sample: '5', bins: 2}, readouts: [{label: 'q', expr: 'quantile(__samples__, .5)', fmt: '0.00'}]});
assert.equal(hist.el.readouts.children[0].children[1].textContent, '5.00');
hist.spec.histogram.sample = '7'; hist.redraw();
assert.equal(hist.el.readouts.children[0].children[1].textContent, '7.00');
assert.ok(Object.isFrozen(hist.vars.__samples__));
hist.spec.histogram.sample = 'NaN'; hist.redraw();
assert.equal(hist.vars.__samples__.length, 0);
const heat = draw({kind: 'heatmap', heat: {rows: ['a','b'], cols: ['a','b'], values: [[1,.3],[.3,1]], bind: {rho: [0,1]}, editable: true, symmetric: true}, readouts: [{label: 'rho', expr: 'rho', fmt: '0.00'}]});
assert.equal(heat.el.readouts.children[0].children[1].textContent, '0.30');
const grid = heat.el.figure.children[0].children[0];
/* 可编辑格子里的 <input> 是**常驻**的：不需要先点一下 —— 那条"点击后把文本换成输入框 +
   focus()"的路在内嵌 iframe 里会被抢焦点，blur 立刻提交把格子还原，表现就是"改不动"。 */
const cell = grid.children[5];                               // 第 1 行第 2 列 = .3
const input = cell.children[0];
assert.equal(input.tagName, 'input', '可编辑格子里应常驻一个 input');
assert.equal(input.value, '0.30', '初始值按 heat.fmt 显示');
document.activeElement = input;                              // 模拟"这个格子正在被编辑"
input.value = '0.8'; input.events.input();
assert.equal(heat.spec.heat.values[0][1], 0.8, '输入即写回 heat.values');
assert.equal(heat.spec.heat.values[1][0], 0.8, 'symmetric 同步镜像格的值');
assert.equal(heat.el.readouts.children[0].children[1].textContent, '0.80');
assert.equal(input.value, '0.8', '正在编辑的格子不被回写覆盖（否则光标会跳回开头）');
assert.equal(grid.children[7].children[0].value, '0.80', '镜像格显示同步更新');
assert.equal(grid.children[5], cell, '就地重绘不重建格子 DOM');
assert.equal(cell.children[0], input, '输入框节点本身也没被换掉（焦点才留得住）');
assert.deepEqual(heat.errors, [], '正常编辑不该产生 [错误]');
/* 非法/空输入：还原成 store 里的值 + 一条 [错误]，且不写回 */
input.value = ''; input.events.change();
assert.equal(heat.spec.heat.values[0][1], 0.8, '空值不写回');
assert.equal(input.value, '0.80', '空值还原成 store 里的值');
assert.ok(heat.errors.some(e => e.includes('不是有效数字')), '非法输入要留一条 [错误]');
/* Home 键之外的提交路径：Enter 也要能改 */
input.value = '0.25'; input.events.keydown({key: 'Enter', preventDefault() {}});
assert.equal(heat.spec.heat.values[0][1], 0.25, 'Enter 提交同样写回');
assert.equal(heat.el.readouts.children[0].children[1].textContent, '0.25');
/* Esc：不写回，恢复原值 */
input.value = '0.99'; input.events.keydown({key: 'Escape', preventDefault() {}});
assert.equal(heat.spec.heat.values[0][1], 0.25, 'Esc 不写回');
assert.equal(input.value, '0.25', 'Esc 恢复 store 里的值');
document.activeElement = null;
assert.deepEqual(heat.errors, []);
heat.spec.heat.values[0][1] = null; heat.redraw();
assert.ok(Number.isNaN(heat.vars.rho));

/* ---- 正态分位数 z = Φ⁻¹(p)（Q-Q 用，Acklam 近似） ---- */
assert.equal(WG.normQuantile(0.5), 0);                                  // 中段 q=0 ⇒ 精确 0
near(WG.normQuantile(0.975), 1.959963984540054, 1e-6, 'z(0.975)');
near(WG.normQuantile(0.025), -1.959963984540054, 1e-6, 'z(0.025)');
assert.equal(WG.normQuantile(0), -Infinity);                            // 不外推
assert.equal(WG.normQuantile(1), Infinity);
assert.ok(Number.isNaN(WG.normQuantile(NaN)));

/* ---- box：Tukey 口径。values [1..9,100]：Q1=3.25、Q2=5.5、Q3=7.75、IQR=4.5，
   1.5×IQR 围栏 [-3.5, 14.5] ⇒ 须内最远点为 1 与 9，100 是唯一异常点。 ---- */
const boxIt = draw({kind: 'box', box: {values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}});
const bg = boxIt.plotGeom;
const boxRect = findAll(boxIt.el.figure, 'wg-box-q')[0];
const boxMed = findAll(boxIt.el.figure, 'wg-box-med')[0];
const whiskers = findAll(boxIt.el.figure, 'wg-box-whi');
const caps = findAll(boxIt.el.figure, 'wg-box-cap');
const outDots = findAll(boxIt.el.figure, 'wg-box-out');
near(xData(bg, Number(boxRect.attrs.x)), 3.25, 5e-3, '箱左沿 = Q1');
near(xData(bg, Number(boxRect.attrs.x) + Number(boxRect.attrs.width)), 7.75, 5e-3, '箱右沿 = Q3');
near(xData(bg, Number(boxMed.attrs.x1)), 5.5, 5e-3, '中位线 = Q2');
const whiskerVals = [];
whiskers.forEach(w => { whiskerVals.push(xData(bg, Number(w.attrs.x1)), xData(bg, Number(w.attrs.x2))); });
near(Math.min.apply(null, whiskerVals), 1, 5e-3, '下须 = 1.5×IQR 内最小值');
near(Math.max.apply(null, whiskerVals), 9, 5e-3, '上须 = 1.5×IQR 内最大值');
assert.equal(whiskers.length, 2);
assert.equal(caps.length, 2);
assert.equal(outDots.length, 1);
near(xData(bg, Number(outDots[0].attrs.cx)), 100, 5e-2, '异常点位置');
// 中位数不随须/异常点移动：把 100 去掉后 Q1/Q3 都变 → 重新画一次核对插值口径
const box2 = draw({kind: 'box', box: {values: [1, 2, 3, 4]}});
const box2Rect = findAll(box2.el.figure, 'wg-box-q')[0];
near(xData(box2.plotGeom, Number(box2Rect.attrs.x)), 1.75, 5e-3, 'n=4 时 Q1 = 1.75（线性插值）');
near(xData(box2.plotGeom, Number(box2Rect.attrs.x) + Number(box2Rect.attrs.width)), 3.25, 5e-3, 'n=4 时 Q3 = 3.25');

/* ---- ecdf：单调阶梯，起点 (min,0)、终点 (max,1) ---- */
const ecdIt = draw({kind: 'ecdf', ecdf: {values: [1, 2, 3, 4]}});
const eg = ecdIt.plotGeom;
const ecdPath = findAll(ecdIt.el.figure, 'wg-ecd')[0];
const stepPts = String(ecdPath.attrs.d).trim().split(/[ML]/).filter(s => s.trim())
  .map(s => s.trim().split(/\s+/).map(Number));
assert.ok(stepPts.length >= 8, 'ecdf 路径顶点数');
for (let i = 1; i < stepPts.length; i++) {
  assert.ok(stepPts[i][0] >= stepPts[i - 1][0] - 1e-9, 'ecdf 的 x 单调不减');
  assert.ok(stepPts[i][1] <= stepPts[i - 1][1] + 1e-9, 'ecdf 的像素 y 单调不增（累积概率单调不减）');
}
near(xData(eg, stepPts[0][0]), 1, 1e-3, 'ecdf 起点 x=min');
near(yData(eg, stepPts[0][1]), 0, 1e-3, 'ecdf 起点 F=0');
near(xData(eg, stepPts[stepPts.length - 1][0]), 4, 1e-3, 'ecdf 终点 x=max');
near(yData(eg, stepPts[stepPts.length - 1][1]), 1, 1e-3, 'ecdf 终点 F=1');
assert.ok(stepPts.some(p => Math.abs(xData(eg, p[0]) - 2) < 1e-3 && Math.abs(yData(eg, p[1]) - 0.5) < 1e-3),
  'ecdf 在 x=2 处应到达累积概率 0.5');
assert.ok(stepPts.some(p => Math.abs(xData(eg, p[0]) - 3) < 1e-3 && Math.abs(yData(eg, p[1]) - 0.75) < 1e-3),
  'ecdf 在 x=3 处应到达累积概率 0.75');

/* ---- qq：横轴 = z((i−0.5)/n)，参考线过 (z(0.25), Q1) 与 (z(0.75), Q3) ---- */
const qqIt = draw({kind: 'qq', qq: {values: Array.from({length: 20}, (_, i) => i + 1)}});
const qg = qqIt.plotGeom;
const ref = findAll(qqIt.el.figure, 'wg-qq-ref')[0];
const Z25 = -0.6744897501960817, Z75 = 0.6744897501960817;   // 查表真值，独立于 WG.normQuantile
const Q1 = 5.75, Q3 = 15.25;                                  // [1..20] 的线性插值四分位
const slope = (Q3 - Q1) / (Z75 - Z25), icpt = Q1 - slope * Z25;
const refY = z => icpt + slope * z;
const line = [[xData(qg, Number(ref.attrs.x1)), yData(qg, Number(ref.attrs.y1))],
              [xData(qg, Number(ref.attrs.x2)), yData(qg, Number(ref.attrs.y2))]];
near(line[0][0], -1.959963984540054, 1e-3, '参考线左端 z=z(0.025)');
near(line[1][0], 1.959963984540054, 1e-3, '参考线右端 z=z(0.975)');
near(line[0][1], refY(line[0][0]), 5e-3, '参考线左端落在四分位连线上');
near(line[1][1], refY(line[1][0]), 5e-3, '参考线右端落在四分位连线上');
const tMid = (Z25 - line[0][0]) / (line[1][0] - line[0][0]);
near(line[0][1] + (line[1][1] - line[0][1]) * tMid, Q1, 5e-3, '参考线在 z(0.25) 处应过 Q1');
const qqDots = findAll(qqIt.el.figure, 'wg-qq-dot');
assert.equal(qqDots.length, 20);
const dotData = qqDots.map(d => [xData(qg, Number(d.attrs.cx)), yData(qg, Number(d.attrs.cy))])
  .sort((a, b) => a[0] - b[0]);
near(dotData[0][0], -1.959963984540054, 1e-3, '首点横坐标 z(0.025)');
near(dotData[0][1], 1, 1e-3, '首点纵坐标 = 最小样本 1');
near(dotData[19][0], 1.959963984540054, 1e-3, '末点横坐标 z(0.975)');
near(dotData[19][1], 20, 1e-3, '末点纵坐标 = 最大样本 20');

/* 抽稀分支：ecdf 抽稀后终点仍必须是 (max,1)；qq 抽稀后仍保留两端样本点 */
const ecdBig = draw({kind: 'ecdf', ecdf: {values: Array.from({length: 12}, (_, i) => i + 1), maxPoints: 3}});
const ep = String(findAll(ecdBig.el.figure, 'wg-ecd')[0].attrs.d).trim().split(/[ML]/).filter(s => s.trim())
  .map(s => s.trim().split(/\s+/).map(Number));
near(xData(ecdBig.plotGeom, ep[0][0]), 1, 1e-3, '抽稀 ecdf 起点 x=min');
near(yData(ecdBig.plotGeom, ep[0][1]), 0, 1e-3, '抽稀 ecdf 起点 F=0');
near(xData(ecdBig.plotGeom, ep[ep.length - 1][0]), 12, 1e-3, '抽稀 ecdf 终点 x=max');
near(yData(ecdBig.plotGeom, ep[ep.length - 1][1]), 1, 1e-3, '抽稀 ecdf 终点 F=1');
for (let i = 1; i < ep.length; i++) {
  assert.ok(ep[i][0] >= ep[i - 1][0] - 1e-9 && ep[i][1] <= ep[i - 1][1] + 1e-9, '抽稀后的 ecdf 仍单调');
}
const qqBig = draw({kind: 'qq', qq: {values: Array.from({length: 20}, (_, i) => i + 1), maxPoints: 5}});
const bigDots = findAll(qqBig.el.figure, 'wg-qq-dot');
assert.equal(bigDots.length, 6, 'qq maxPoints=5 ⇒ 每 4 个画一个，外加最大样本点');
const bigData = bigDots.map(d => [xData(qqBig.plotGeom, Number(d.attrs.cx)), yData(qqBig.plotGeom, Number(d.attrs.cy))])
  .sort((a, b) => a[0] - b[0]);
near(bigData[0][0], -1.959963984540054, 1e-3, '抽稀 qq 首个点仍是 z(0.025)');
near(bigData[0][1], 1, 1e-3, '抽稀 qq 保留最小样本点');
near(bigData[5][0], 1.959963984540054, 1e-3, '抽稀 qq 末个点仍是 z(0.975)');
near(bigData[5][1], 20, 1e-3, '抽稀 qq 保留最大样本点');

/* ---- 失败分支：样本为空 / 样本量不足 / 非法 dist 必须报 [错误]，不静默画错 ---- */
const boxEmpty = render({kind: 'box', box: {values: []}});
assert.ok(boxEmpty.errors.some(e => e.indexOf('[错误]') === 0), 'box 空样本要报 [错误]');
assert.equal(boxEmpty.el.figure.children.length, 0, 'box 空样本不画图');
const boxTiny = render({kind: 'box', box: {values: [3]}});
assert.ok(boxTiny.errors.some(e => e.includes('样本量不足')), 'box n=1 要报样本量不足');
assert.equal(boxTiny.el.figure.children.length, 0, 'box n=1 不画假箱体');
const ecdEmpty = render({kind: 'ecdf', controls: [{key: 'n', type: 'number', min: 10, max: 100, step: 1, value: 10}], ecdf: {sample: 'NaN'}});
assert.ok(ecdEmpty.errors.some(e => e.includes('样本为空')), 'ecdf 抽样全失败要报样本为空');
assert.equal(ecdEmpty.el.figure.children.length, 0, 'ecdf 空样本不画图');
const qqTiny = render({kind: 'qq', qq: {values: [1]}});
assert.ok(qqTiny.errors.some(e => e.includes('样本量不足')), 'qq n=1 要报样本量不足');
const qqDist = render({kind: 'qq', qq: {values: [1, 2, 3], dist: 'cauchy'}});
assert.ok(qqDist.errors.some(e => e.includes('qq.dist')), '非法 dist 要报 [错误]');
assert.equal(qqDist.el.figure.children.length, 0, '非法 dist 不画图');
const boxNoSource = render({kind: 'box', box: {}});
assert.ok(boxNoSource.errors.length > 0, 'box 缺 sample/values 要报 [错误]');

/* ---- 抽样约定：每次重绘重新抽样，且 readouts 能读到 __samples__ ---- */
const boxDraw = draw({kind: 'box', controls: [{key: 'n', type: 'number', min: 10, max: 100, step: 1, value: 10}],
  box: {sample: 'i'}, readouts: [{label: 'max', expr: 'quantile(__samples__, 1)', fmt: '0'}]});
assert.equal(boxDraw.el.readouts.children[0].children[1].textContent, '9');
assert.equal(boxDraw.vars.__samples__.length, 10);
assert.ok(Object.isFrozen(boxDraw.vars.__samples__));
boxDraw.spec.box.sample = '2*i'; boxDraw.redraw();
assert.equal(boxDraw.el.readouts.children[0].children[1].textContent, '18');
assert.equal(boxDraw.vars.__samples__.length, 10);


/* ---- contour：marching squares。f = x + y 在 [0,1]² 上，levels=1 只取中间那条：x + y = 1 ---- */
const ctIt = draw({kind: 'contour',
  x: {min: 0, max: 1, points: 9}, y: {min: 0, max: 1, points: 9},
  contour: {expr: 'x + y', levels: 1}});
const cg = ctIt.plotGeom;
const cPaths = findAll(ctIt.el.figure, 'wg-contour');
assert.equal(cPaths.length, 1, 'contour levels=1 ⇒ 恰好一条等值线路径');
const cpts = String(cPaths[0].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
assert.ok(cpts.length >= 8, 'contour 至少两段（9×9 网格）');
const cData = [];
for (let i = 0; i < cpts.length; i += 2) cData.push([xData(cg, cpts[i]), yData(cg, cpts[i + 1])]);
cData.forEach(p => {
  near(p[0] + p[1], 1, 2e-3, '等值线经过 x + y = 1');
  assert.ok(p[0] >= -1e-6 && p[0] <= 1 + 1e-6 && p[1] >= -1e-6 && p[1] <= 1 + 1e-6, '等值线落在定义域内');
});
assert.ok(cData.some(p => Math.abs(p[0] - 1) < 2e-3 && Math.abs(p[1]) < 2e-3), '等值线从边界 (1,0) 出发');
assert.ok(cData.some(p => Math.abs(p[0]) < 2e-3 && Math.abs(p[1] - 1) < 2e-3), '等值线终止于边界 (0,1)');

/* 缺失值：f = sqrt(x) 在 x<0 的列是 NaN——含缺失点的格子整体跳过（等值线只出现在 x≥0 一侧） */
const cmIt = render({kind: 'contour',
  x: {min: -1, max: 1, points: 5}, y: {min: -1, max: 1, points: 5},
  contour: {expr: 'sqrt(x)', levels: 1}});
assert.ok(cmIt.errors.some(e => e.includes('网格点') && e.includes('缺失')), '非有限值要按缺失处理并报 [错误]');
const cmXs = [];
findAll(cmIt.el.figure, 'wg-contour').forEach(path => {
  const ns = String(path.attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  for (let i = 0; i < ns.length; i += 2) cmXs.push(xData(cmIt.plotGeom, ns[i]));
});
assert.ok(cmXs.length > 0, '缺失只跳过含缺失点的格子，其余照画');
assert.ok(Math.min.apply(null, cmXs) >= -1e-3, '缺失列（x<0）不画线，不把 NaN 当 0');

/* 网格点数越界：钳到上限 121 并报 [错误]，仍然画图 */
const coIt = render({kind: 'contour',
  x: {min: 0, max: 1, points: 500}, y: {min: 0, max: 1, points: 5},
  contour: {expr: 'x + y', levels: 1}});
assert.ok(coIt.errors.some(e => e.includes('121')), '网格点数越界要报上限');
assert.ok(findAll(coIt.el.figure, 'wg-contour').length >= 1, '越界后按钳后的点数继续画');

/* expr 未填：报 [错误] 且不画图 */
const cNo = render({kind: 'contour',
  x: {min: 0, max: 1, points: 9}, y: {min: 0, max: 1, points: 9}, contour: {}});
assert.ok(cNo.errors.some(e => e.includes('contour.expr')), 'contour 缺 expr 要报 [错误]');
assert.equal(cNo.el.figure.children.length, 0, 'contour 缺 expr 不画图');

/* ---- vector：恒定场 u=1, v=0。auto 口径：最长箭头 = 0.9×min(单元格宽, 单元格高) ---- */
const vcIt = draw({kind: 'vector',
  x: {min: 0, max: 1, points: 21}, y: {min: 0, max: 1, points: 21},
  vector: {u: '1', v: '0'}});
const vg = vcIt.plotGeom;
const vArrows = findAll(vcIt.el.figure, 'wg-vec');
assert.equal(vArrows.length, 400, '21×21 网格 ⇒ 20×20 个单元格中心各一支箭头');
{
  const ns = String(vArrows[0].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  near(ns[3], ns[1], 1e-9, 'v = 0 ⇒ 箭头水平（像素 y 不变）');
  near(ns[2] - ns[0], 0.9 * Math.min(vg.pw / 20, vg.ph / 20), 0.05,
    'auto：全场最大模长的箭头视觉长度 = 0.9×min(cellW, cellH)');
}

/* 方向：终点 = 起点 + k·(u,v)，像素方向 = (u·pxX, −v·pxY)（屏幕 y 向下） */
const vdIt = draw({kind: 'vector',
  x: {min: -1, max: 1, points: 11}, y: {min: -1, max: 1, points: 11},
  vector: {u: '1', v: '2'}});
{
  const ns = String(findAll(vdIt.el.figure, 'wg-vec')[0].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  const pxX = vdIt.plotGeom.pw / 2, pxY = vdIt.plotGeom.ph / 2;
  near((ns[3] - ns[1]) / (ns[2] - ns[0]), -2 * pxY / pxX, 2e-3, '箭头方向 = (u·pxX, −v·pxY)');
}

/* unit 口径：所有箭头等数据长度 = 0.9×min(Δx, Δy)，只表示方向 */
const vuIt = draw({kind: 'vector',
  x: {min: 0, max: 1, points: 11}, y: {min: 0, max: 1, points: 11},
  vector: {u: '0', v: '1', scale: 'unit'}});
{
  const ns = String(findAll(vuIt.el.figure, 'wg-vec')[0].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  near(ns[2], ns[0], 1e-9, 'unit 下 (0,1) 方向箭头竖直');
  near(Math.hypot(ns[2] - ns[0], ns[3] - ns[1]), 0.09 * vuIt.plotGeom.ph, 0.05,
    'unit：数据长度 = 0.9×min(Δx, Δy) = 0.09，竖直方向像素长度 = 0.09×pxY');
}

/* 失败分支：全零场不画箭头；u 未填报错；非有限值按缺失处理 */
const vzIt = render({kind: 'vector',
  x: {min: 0, max: 1, points: 5}, y: {min: 0, max: 1, points: 5}, vector: {u: '0', v: '0'}});
assert.ok(vzIt.errors.some(e => e.includes('箭头') && e.includes('0')), '全零场要报“没有可画的箭头”');
assert.equal(findAll(vzIt.el.figure, 'wg-vec').length, 0, '零向量不画箭头');
const vmIt = render({kind: 'vector',
  x: {min: 0, max: 1, points: 5}, y: {min: 0, max: 1, points: 5}, vector: {u: '', v: 'x'}});
assert.ok(vmIt.errors.some(e => e.includes('vector.u')), 'u 未填要报 [错误]');
const vnIt = render({kind: 'vector',
  x: {min: -1, max: 1, points: 5}, y: {min: -1, max: 1, points: 5}, vector: {u: 'sqrt(x)', v: '0'}});
assert.ok(vnIt.errors.some(e => e.includes('缺失')), 'sqrt 负值要按缺失处理并提示');
assert.ok(findAll(vnIt.el.figure, 'wg-vec').length > 0, '缺失的点跳过，其余箭头照画');

/* ---- matrix：A = [[2,1],[1,2]]。det=3；A·e1=(2,1)、A·e2=(1,2)；特征值 3 与 1，方向不变 ---- */
const mxIt = draw({kind: 'matrix',
  matrix: {values: [[2, 1], [1, 2]], editable: true, samples: [[1, 0.5]]},
  readouts: [
    {label: 'det', expr: 'a*d - b*c', fmt: '0.00'},
    {label: 'tr', expr: 'a + d', fmt: '0.00'},
    {label: 'λ1', expr: 'ifelse((a+d)*(a+d) - 4*(a*d-b*c) >= 0, ((a+d) + sqrt((a+d)*(a+d) - 4*(a*d-b*c)))/2, NaN)', fmt: '0.00'}
  ]});
assert.equal(mxIt.el.readouts.children[0].children[1].textContent, '3.00', 'readout 能算出行列式 det = 3');
assert.equal(mxIt.el.readouts.children[1].children[1].textContent, '4.00', 'readout 迹 tr = 4');
assert.equal(mxIt.el.readouts.children[2].children[1].textContent, '3.00', 'readout 实数特征值 λ1 = 3');
const mg = mxIt.plotGeom;
function poly2data(el) {
  return String(el.attrs.points).trim().split(/\s+/).map(s => {
    const q = s.split(',').map(Number);
    return [xData(mg, q[0]), yData(mg, q[1])];
  });
}
const transData = poly2data(findAll(mxIt.el.figure, 'wg-mx-trans')[0]);
near(transData[0][0], 0, 5e-3, '变换后正方形第 1 角 x=0');
near(transData[0][1], 0, 5e-3, '变换后正方形第 1 角 y=0');
near(transData[1][0], 2, 5e-3, 'A·e1 的 x = a = 2');
near(transData[1][1], 1, 5e-3, 'A·e1 的 y = c = 1');
near(transData[2][0], 3, 5e-3, 'A·(1,1) 的 x = a+b = 3');
near(transData[2][1], 3, 5e-3, 'A·(1,1) 的 y = c+d = 3');
near(transData[3][0], 1, 5e-3, 'A·e2 的 x = b = 1');
near(transData[3][1], 2, 5e-3, 'A·e2 的 y = d = 2');
const bimg = findAll(mxIt.el.figure, 'wg-mx-bimg');
assert.equal(bimg.length, 2, '基向量 e1 / e2 的像各一条箭头');
{
  const b1 = String(bimg[0].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  near(xData(mg, b1[2]), 2, 5e-3, 'A·e1 终点 x');
  near(yData(mg, b1[3]), 1, 5e-3, 'A·e1 终点 y');
  const b2 = String(bimg[1].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  near(xData(mg, b2[2]), 1, 5e-3, 'A·e2 终点 x');
  near(yData(mg, b2[3]), 2, 5e-3, 'A·e2 终点 y');
}
const eLines = findAll(mxIt.el.figure, 'wg-mx-eigen');
const eImgs = findAll(mxIt.el.figure, 'wg-mx-eimg');
assert.equal(eLines.length, 2, '两个实特征值 ⇒ 两条特征方向');
assert.equal(eImgs.length, 2, '每个特征方向画一条 A·v = λv');
eLines.forEach((ln, i) => {
  const vx = xData(mg, Number(ln.attrs.x2)), vy = yData(mg, Number(ln.attrs.y2));
  const ns = String(eImgs[i].attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
  const ix = xData(mg, ns[2]), iy = yData(mg, ns[3]);
  near(Math.hypot(vx, vy), 1, 5e-3, '特征方向按单位向量画');
  near(vx * iy - vy * ix, 0, 5e-3, 'A·v 与 v 共线：特征方向不变');
});
{
  const ratio = eImgs.map(p => {
    const ns = String(p.attrs.d).match(/-?\d+(?:\.\d+)?/g).map(Number);
    return Math.hypot(xData(mg, ns[2]), yData(mg, ns[3]));   // 单位方向 ⇒ |λv| = |λ|
  }).sort((p, q) => p - q);
  near(ratio[0], 1, 5e-3, 'λ = 1 的像长度 = 1');
  near(ratio[1], 3, 5e-3, 'λ = 3 的像长度 = 3（|det| = 面积缩放 = λ1·λ2）');
}
assert.ok(mxIt.el.caps.children.map(c => c.textContent).join(' ').includes('面积缩放倍数'),
  '图注要写清 det = 面积缩放倍数');

/* 可编辑格子：格子里常驻 <input>，改 c 后写回 spec.matrix.values，图与 readouts 同步重算，
   而且**不重建格子 DOM**（内嵌 iframe 里正是重建/focus 那一步导致"数字改不动"） */
const mxGrid = mxIt.el.figure.children[0].children[0];      // DOM grid 在 SVG 之前
const mxCell = mxGrid.children[7];                           // 第 2 行第 1 列 = c
const mxInput = mxCell.children[0];
assert.equal(mxInput.tagName, 'input', '可编辑格子里应常驻一个 input（无需先点击）');
assert.equal(mxInput.value, '1.00', 'A[2][1] = 1 按 patternForValue 显示为 1.00（与图注/readouts 同一口径）');
const mxSvgBefore = mxIt.el.figure.children[1];
document.activeElement = mxInput;
mxInput.value = '0.5';
mxInput.events.input();
assert.equal(mxIt.spec.matrix.values[1][0], 0.5, '输入即写回 spec.matrix.values');
assert.equal(mxIt.el.readouts.children[0].children[1].textContent, '3.50',
  'det 随编辑同步重算：2×2 − 1×0.5 = 3.5');
assert.equal(mxGrid.children[7], mxCell, '就地重绘不重建格子 DOM');
assert.equal(mxCell.children[0], mxInput, '输入框节点保留（焦点/光标才留得住）');
assert.notEqual(mxIt.el.figure.children[1], mxSvgBefore, '图仍然按新值重画（换掉旧 SVG 节点）');
assert.ok(mxIt.el.caps.children.map(c => c.textContent).join(' ').includes('可以直接改'),
  '图注要说明格子可以直接改');
/* Enter 提交与 Esc 取消 */
mxInput.value = '2'; mxInput.events.keydown({key: 'Enter', preventDefault() {}});
assert.equal(mxIt.spec.matrix.values[1][0], 2, 'Enter 提交写回');
assert.equal(mxIt.el.readouts.children[0].children[1].textContent, '2.00');
mxInput.value = '-9'; mxInput.events.keydown({key: 'Escape', preventDefault() {}});
assert.equal(mxIt.spec.matrix.values[1][0], 2, 'Esc 不写回');
assert.equal(mxInput.value, '2.00', 'Esc 恢复 store 里的值（按同一 pattern 显示）');
/* 非法输入：还原 + [错误] */
mxInput.value = ''; mxInput.events.change();
assert.equal(mxIt.spec.matrix.values[1][0], 2, '空值不写回');
assert.ok(mxIt.errors.some(e => e.includes('不是有效数字')), '非法输入要留一条 [错误]');
document.activeElement = null;

/* det = 0：如实标奇异；复特征值：写出 p ± qi，不画特征方向；非 2×2：报 [错误] 不画图 */
const mzIt = render({kind: 'matrix', matrix: {values: [[1, 1], [1, 1]]},
  readouts: [{label: 'det', expr: 'a*d - b*c', fmt: '0.00'}]});
assert.equal(mzIt.el.readouts.children[0].children[1].textContent, '0.00', 'det = 0 如实显示 0.00');
assert.ok(mzIt.el.caps.children.map(c => c.textContent).join(' ').includes('奇异'),
  'det = 0 要标出矩阵奇异（面积缩到 0）而不是静默略过');
const mcIt = draw({kind: 'matrix', matrix: {values: [[0, -1], [1, 0]]}});
const mcap = mcIt.el.caps.children.map(c => c.textContent).join(' ');
assert.ok(mcap.includes('复数') && mcap.includes('0.00 ± 1.00i'), '复特征值要写出虚部 p ± qi');
assert.equal(findAll(mcIt.el.figure, 'wg-mx-eigen').length, 0, '复特征值不画实特征方向');
assert.equal(findAll(mcIt.el.figure, 'wg-mx-eimg').length, 0);
const mbIt = render({kind: 'matrix', matrix: {values: [[1, 2], [3]]}});
assert.ok(mbIt.errors.some(e => e.includes('matrix.values')), '非 2×2 要报 [错误]');
assert.equal(mbIt.el.figure.children.length, 0, '非法矩阵不画图');


/* ---- aspect：等比例坐标轴。默认 matrix=equal、contour/vector=auto；显式覆盖生效；非法值报 [错误] ---- */
function pxPerUnit(g) { return [g.pw / (g.xmax - g.xmin), g.ph / (g.ymax - g.ymin)]; }
function nearPx(a, b, what) { near(a, b, Math.abs(a) * 5e-3, what + '（容差 0.5%）'); }
function numsOf(d) { return String(d).match(/-?\d+(?:\.\d+)?/g).map(Number); }
function vertexPx(el) { return String(el.attrs.points).trim().split(/\s+/).map(s => s.split(',').map(Number)); }
function sideLens(el) {
  const p = vertexPx(el);
  return p.map((q, i) => { const r = p[(i + 1) % p.length]; return Math.hypot(r[0] - q[0], r[1] - q[1]); });
}
function allContourPts(it) {
  const out = [];
  findAll(it.el.figure, 'wg-contour').forEach(path => {
    const n = numsOf(path.attrs.d);
    for (let i = 0; i < n.length; i += 2) out.push([n[i], n[i + 1]]);
  });
  return out;
}
function radiusSpread(pts) {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  const r = pts.map(p => Math.hypot(p[0] - cx, p[1] - cy));
  return Math.max.apply(null, r) / Math.min.apply(null, r);
}
function capsText(it) { return it.el.caps.children.map(c => c.textContent).join(' '); }
/* f = x² + y² 在 [-2,2]² 上：level 取中值 ⇒ 真等值线是半径 2 的圆 */
function circleSpec(extra) {
  return Object.assign({kind: 'contour', x: {min: -2, max: 2, points: 41},
    y: {min: -2, max: 2, points: 41}, contour: {expr: 'x*x + y*y', levels: 1}}, extra || {});
}
function vecSpec(extra) {
  return Object.assign({kind: 'vector', x: {min: -2, max: 2, points: 21},
    y: {min: -2, max: 2, points: 21}, vector: {u: 'y', v: '-x'}}, extra || {});
}

/* (a) matrix 默认 equal：两轴 px/单位相等，单位正方形在屏幕上仍是正方形 */
const mxEqAsp = draw({kind: 'matrix', matrix: {values: [[2, 1], [1, 2]], samples: [[1, 0.5]]}});
{
  const sc = pxPerUnit(mxEqAsp.plotGeom);
  assert.equal(mxEqAsp.plotGeom.aspect, 'equal', 'matrix 默认 aspect=equal');
  assert.equal(mxEqAsp.plotGeom.aspectFallback, null, 'matrix 默认不退化');
  assert.ok(isFinite(sc[0]) && sc[0] > 0 && isFinite(sc[1]) && sc[1] > 0, 'matrix：px/单位 是有限正数');
  nearPx(sc[0], sc[1], 'equal：pw/(xmax−xmin) == ph/(ymax−ymin)');
  const sides = sideLens(findAll(mxEqAsp.el.figure, 'wg-mx-orig')[0]);
  assert.equal(sides.length, 4, '单位正方形 4 条边');
  sides.forEach((L, i) => near(L, sides[0], sides[0] * 5e-3,
    'equal：单位正方形第 ' + (i + 1) + ' 条边与第 1 条在屏幕上等长'));
  assert.ok(capsText(mxEqAsp).includes('等比例'), '矩阵图注要写明当前用的是等比例坐标轴');
}

/* (f) equal 只是缩框 + 居中：网格线、轴、零轴线、基向量像与特征方向的端点都要跟着新框走 */
const mxNegAsp = draw({kind: 'matrix', matrix: {values: [[1, -2], [-2, 1]]}});
{
  const g = mxNegAsp.plotGeom;
  assert.equal(g.aspect, 'equal', 'A 的两轴都跨 0 时仍默认 equal');
  const glines = findAll(mxNegAsp.el.figure, 'wg-gline');
  assert.ok(glines.length > 0, '有网格线');
  glines.forEach(ln => {
    if (Number(ln.attrs.y1) === Number(ln.attrs.y2)) {
      near(Number(ln.attrs.x1), g.l, 1e-9, 'equal：水平网格线左端贴新框');
      near(Number(ln.attrs.x2), g.l + g.pw, 1e-9, 'equal：水平网格线右端贴新框');
    } else {
      near(Number(ln.attrs.y1), g.t, 1e-9, 'equal：竖直网格线上端贴新框');
      near(Number(ln.attrs.y2), g.t + g.ph, 1e-9, 'equal：竖直网格线下端贴新框');
    }
  });
  const zeros = findAll(mxNegAsp.el.figure, 'wg-zero');
  assert.equal(zeros.length, 2, '两轴都跨 0 ⇒ 两条零轴线');
  zeros.forEach(z => {
    assert.ok(Number(z.attrs.x1) >= g.l - 1e-9 && Number(z.attrs.x2) <= g.l + g.pw + 1e-9 &&
      Number(z.attrs.y1) >= g.t - 1e-9 && Number(z.attrs.y2) <= g.t + g.ph + 1e-9,
      '零轴线落在新图框内（不会停在居中前的旧位置）');
  });
  const inFrame = (px, py, what) => assert.ok(
    px >= g.l - 1e-9 && px <= g.l + g.pw + 1e-9 && py >= g.t - 1e-9 && py <= g.t + g.ph + 1e-9, what);
  assert.equal(findAll(mxNegAsp.el.figure, 'wg-mx-eigen').length, 2, '两个实特征值 ⇒ 两条特征方向线（下面不空转）');
  assert.equal(findAll(mxNegAsp.el.figure, 'wg-mx-eimg').length, 2, '两个特征值的像');
  assert.equal(findAll(mxNegAsp.el.figure, 'wg-mx-bimg').length, 2, '两条基向量像');
  findAll(mxNegAsp.el.figure, 'wg-mx-eigen').forEach(ln => inFrame(Number(ln.attrs.x2), Number(ln.attrs.y2), '特征方向端点在新框内'));
  findAll(mxNegAsp.el.figure, 'wg-mx-eimg').forEach(p => { const n = numsOf(p.attrs.d); inFrame(n[2], n[3], 'A·v 的终点在新框内'); });
  findAll(mxNegAsp.el.figure, 'wg-mx-bimg').forEach(p => { const n = numsOf(p.attrs.d); inFrame(n[2], n[3], 'A·e 的终点在新框内'); });
  const tickTexts = findAll(mxNegAsp.el.figure, 'wg-tick');
  assert.ok(tickTexts.length >= 4, '刻度文字存在（下面不空转）');
  tickTexts.forEach(t => {
    if (Number(t.attrs.x) === g.l - 6) {
      assert.ok(Number(t.attrs.y) >= g.t - 1e-9 && Number(t.attrs.y) <= g.t + g.ph + 1e-9, 'y 刻度文字贴在新框左沿');
    } else {
      near(Number(t.attrs.y), g.t + g.ph + 14, 1e-9, 'x 刻度文字贴在新框下沿');
      assert.ok(Number(t.attrs.x) >= g.l - 1e-9 && Number(t.attrs.x) <= g.l + g.pw + 1e-9, 'x 刻度文字落在新框宽度内');
    }
  });
}

/* (b) contour / vector 默认 auto（保持既有行为），图注说明几何会被压扁 */
const ctAutoAsp = draw(circleSpec());
const vcAutoAsp = draw(vecSpec());
[['contour', ctAutoAsp], ['vector', vcAutoAsp]].forEach(pair => {
  const kind = pair[0], it = pair[1], sc = pxPerUnit(it.plotGeom);
  assert.equal(it.plotGeom.aspect, 'auto', kind + ' 默认 aspect=auto');
  assert.ok(Math.abs(sc[0] / sc[1] - 1) > 0.05, kind + ' 默认 auto：两轴 px/单位 不同（几何被拉伸）');
  assert.ok(capsText(it).includes('压扁'), kind + ' 默认 auto 的图注要写明几何形状会被压扁');
});

/* (c) 显式 aspect 覆盖默认：contour / vector 改 equal 后屏幕几何保真 */
const ctEqAsp = draw(circleSpec({aspect: 'equal'}));
{
  const g = ctEqAsp.plotGeom, sc = pxPerUnit(g);
  assert.equal(g.aspect, 'equal', 'contour 显式 aspect=equal 生效');
  nearPx(sc[0], sc[1], 'equal：contour 两轴 px/单位相等');
  const pts = allContourPts(ctEqAsp);
  assert.ok(pts.length > 100, '等值线顶点数（' + pts.length + '）');
  const spreadEq = radiusSpread(pts), spreadAuto = radiusSpread(allContourPts(ctAutoAsp));
  assert.ok(spreadEq < 1.05, 'equal：f = x² + y² 的等值线在屏幕上是圆（半径比 ' + spreadEq.toFixed(4) + ' < 1.05）');
  assert.ok(spreadAuto > 1.3, 'auto：同一个圆被画成椭圆（半径比 ' + spreadAuto.toFixed(4) + '），这正是要修正的误导');
}
const vcEqAsp = draw(vecSpec({aspect: 'equal'}));
{
  const g = vcEqAsp.plotGeom, sc = pxPerUnit(g);
  assert.equal(g.aspect, 'equal', 'vector 显式 aspect=equal 生效');
  nearPx(sc[0], sc[1], 'equal：vector 两轴 px/单位相等');
  /* 旋转场 u = y、v = −x：箭头在数据空间垂直于半径；两轴 px/单位 相等时屏幕上才仍然垂直 */
  function perpCos(it) {
    const gg = it.plotGeom, n = numsOf(findAll(it.el.figure, 'wg-vec')[0].attrs.d);
    const ox = gg.l + gg.pw * (0 - gg.xmin) / (gg.xmax - gg.xmin);
    const oy = gg.t + gg.ph * (1 - (0 - gg.ymin) / (gg.ymax - gg.ymin));
    const dx = n[2] - n[0], dy = n[3] - n[1], rx = n[0] - ox, ry = n[1] - oy;
    return Math.abs(dx * rx + dy * ry) / (Math.hypot(dx, dy) * Math.hypot(rx, ry));
  }
  const cosEq = perpCos(vcEqAsp), cosAuto = perpCos(vcAutoAsp);
  assert.ok(cosEq < 1e-3, 'equal：箭头与半径垂直（屏幕角度保真，cos=' + cosEq.toExponential(2) + '）');
  assert.ok(cosAuto > 0.1, 'auto：同一旋转场的箭头在屏幕上不垂直于半径（cos=' + cosAuto.toFixed(3) + '，方向被压扁）');
}
const mxAutoAsp = draw({kind: 'matrix', aspect: 'auto', matrix: {values: [[2, 1], [1, 2]], samples: [[1, 0.5]]}});
{
  const sc = pxPerUnit(mxAutoAsp.plotGeom);
  assert.equal(mxAutoAsp.plotGeom.aspect, 'auto', 'matrix 可显式改成 aspect=auto');
  assert.ok(Math.abs(sc[0] / sc[1] - 1) > 0.05, 'matrix auto：两轴 px/单位 不相等（几何被拉伸）');
  const sides = sideLens(findAll(mxAutoAsp.el.figure, 'wg-mx-orig')[0]);
  assert.ok(Math.abs(sides[0] / sides[1] - 1) > 0.05, 'matrix auto：单位正方形被画成矩形（' +
    sides[0].toFixed(1) + '×' + sides[1].toFixed(1) + 'px）');
  const caps = capsText(mxAutoAsp);
  assert.ok(caps.includes('压扁') && caps.includes('细长矩形'), 'matrix auto 的图注要特别提醒形状失真');
}

/* (e) 极端长宽比与窄面板：不产生 NaN / 零 / 负尺寸；退化时退回 auto 并在图注说明 */
const exThinAsp = draw({kind: 'contour', aspect: 'equal',
  x: {min: 0, max: 1e6, points: 5}, y: {min: 0, max: 1, points: 5}, contour: {expr: 'x + y', levels: 1}});
{
  const g = exThinAsp.plotGeom;
  assert.ok(isFinite(g.pw) && g.pw > 0 && isFinite(g.ph) && g.ph > 0, '1e6 : 1 的跨度比：pw/ph 是有限正数');
  assert.ok(isFinite(g.l) && isFinite(g.t), '1e6 : 1 的跨度比：图框偏移是有限数');
  assert.equal(g.aspectFallback, 'thin', '极端长宽比：明确标记 equal 退化');
  assert.ok(capsText(exThinAsp).includes('退回两轴独立拉伸'), '退化必须在图注里说明，不能静默');
  assert.ok(allContourPts(exThinAsp).every(q => isFinite(q[0]) && isFinite(q[1])), '退化时等值线像素坐标全是有限数');
}
const exMxAsp = draw({kind: 'matrix', matrix: {values: [[1e6, 0], [0, 1]]}});
{
  const g = exMxAsp.plotGeom;
  assert.ok(isFinite(g.pw) && g.pw > 0 && isFinite(g.ph) && g.ph > 0, '矩阵取极端数值：pw/ph 是有限正数');
  assert.ok(vertexPx(findAll(exMxAsp.el.figure, 'wg-mx-orig')[0]).every(q => isFinite(q[0]) && isFinite(q[1])),
    '矩阵取极端数值：单位正方形顶点坐标全是有限数');
}
const narAsp = draw({kind: 'contour', aspect: 'equal',
  x: {min: 0, max: 2, points: 9}, y: {min: 0, max: 1, points: 9}, contour: {expr: 'x + y', levels: 1}});
{
  narAsp.root.clientWidth = 240; narAsp.el.figure.clientWidth = 240; narAsp.layout();
  const g = narAsp.plotGeom, sc = pxPerUnit(g);
  assert.ok(g.pw >= 12 && g.ph >= 12 && isFinite(sc[0]) && sc[0] > 0 && isFinite(sc[1]) && sc[1] > 0,
    '窄面板（240px）：尺寸与 px/单位 都是正的有限数');
  nearPx(sc[0], sc[1], '窄面板 equal：两轴 px/单位仍相等');
}

/* (d) 非法 aspect：报 [错误]，并按该渲染器的默认模式继续画（不静默忽略） */
const badAspIt = render({kind: 'contour', aspect: 'stretch',
  x: {min: 0, max: 1, points: 9}, y: {min: 0, max: 1, points: 9}, contour: {expr: 'x + y', levels: 1}});
assert.ok(badAspIt.errors.some(e => e.indexOf('[错误]') === 0 && e.includes('aspect')), '非法 aspect 要报 [错误]');
assert.equal(badAspIt.plotGeom.aspect, 'auto', '非法 aspect 按 contour 默认 auto 继续画');
assert.ok(findAll(badAspIt.el.figure, 'wg-contour').length >= 1, '非法 aspect 不影响出图');

/* ============================================================
   新渲染器：regression / pca / descent / surface3d / treefit
   期望值都是手算的，口径写在每段注释里；数字口径变了这里就该红。
   ============================================================ */

/* ---- regression：OLS 口径。手算：x̄=1.5、ȳ=3.75、sxx=5、sxy=9.5 ⇒ 斜率 1.9、截距 0.9；
   残差 −0.9、−0.8、−0.2、1.9 ⇒ SS_res=0.7；SS_tot=(−2.75)²+(−0.75)²+0.25²+3.25²=18.75 ---- */
const regPts = [[0, 1], [1, 3], [2, 4], [3, 7]];
const regIt = draw({kind: 'regression', points: regPts, readouts: [
  {label: 'slope', expr: '__reg__.slope', fmt: '0.0000'},
  {label: 'r2', expr: '__reg__.r2', fmt: '0.0000'},
  {label: 'rmse', expr: '__reg__.rmse', fmt: '0.0000'}]});
near(regIt.vars.__reg__.slope, 1.9, 1e-12, 'regression 斜率 = sxy/sxx = 9.5/5');
near(regIt.vars.__reg__.intercept, 0.9, 1e-12, 'regression 截距 = ȳ − 斜率·x̄');
near(regIt.vars.__reg__.ssRes, 0.7, 1e-12, 'regression SS_res = 0.7');
near(regIt.vars.__reg__.varY, 18.75 / 3, 1e-12, 'regression SS_tot = 18.75 ⇒ 无偏方差 6.25');
near(regIt.vars.__reg__.r2, 1 - 0.7 / 18.75, 1e-12, 'regression R² = 1 − SS_res/SS_tot');
near(regIt.vars.__reg__.rmse, Math.sqrt(0.7 / 4), 1e-12, 'regression RMSE = √(SS_res/n)');
assert.equal(regIt.el.readouts.children[1].children[1].textContent, '0.9627', 'R² 读数是 0.9627');
assert.equal(findAll(regIt.el.figure, 'wg-resid-line').length, 4, 'OLS 下每个点一条残差线');
assert.equal(findAll(regIt.el.figure, 'wg-fit-line').length, 1, 'OLS 只画一条拟合线');

/* compare 模式：手动线 y = x 的残差 1、2、2、4 ⇒ SS_res=25 ⇒ R² = 1 − 25/18.75 = −1/3（负值要如实显示） */
const reg2 = draw({kind: 'regression', points: regPts, fit: {mode: 'compare'},
  controls: [{key: 'slope', label: 'a', min: 0, max: 3, step: 0.1, value: 1},
    {key: 'intercept', label: 'b', min: -2, max: 2, step: 0.1, value: 0}],
  readouts: [{label: 'mR2', expr: '__reg__.mR2', fmt: '0.0000'}]});
assert.equal(findAll(reg2.el.figure, 'wg-fit-line').length, 2, 'compare 模式画 OLS + 手动线两条');
near(reg2.vars.__reg__.mSlope, 1, 1e-12, '手动线斜率取控件值');
near(reg2.vars.__reg__.mIntercept, 0, 1e-12, '手动线截距取控件值');
near(reg2.plotData.mInt, 0, 1e-12, 'plotData 里的手动线截距同一口径');
near(reg2.vars.__reg__.mR2, 1 - 25 / 18.75, 1e-12, '手动线 R²（可以比 0 还小）');
assert.equal(reg2.el.readouts.children[0].children[1].textContent, '-0.3333');

/* ---- pca：点 [-1,-1]、[1,1]。sxx=syy=2、sxy=2 ⇒ λ1=4、λ2=0，PC1 角 45°；
   候选轴 θ 上的方差 = cos²θ·sxx + 2sinθcosθ·sxy + sin²θ·syy ⇒ θ=0°→2、45°→4、90°→2 ---- */
const pcaBase = {kind: 'pca', points: [[-1, -1], [1, 1]]};
const pcaFixed = draw(Object.assign({}, pcaBase, {candidate: {angle: 45}}));
near(pcaFixed.vars.__pca__.l1, 4, 1e-12, 'pca λ1');
near(pcaFixed.vars.__pca__.l2, 0, 1e-12, 'pca λ2');
near(pcaFixed.vars.__pca__.angle1, 45, 1e-12, 'pca PC1 角度 45°');
near(pcaFixed.vars.__pca__.ratio, 1, 1e-12, 'pca 解释方差比 λ1/tr = 4/4');
near(pcaFixed.vars.__pca__.candVar, 4, 1e-12, 'candidate.angle=45°（=PC1 方向）上方差 = λ1 = 4');
/* 给了 angleControl 就以控件为准：控件是"读者能拖的那个角度" */
const pcaIt = draw(Object.assign({}, pcaBase, {candidate: {angle: 45, angleControl: 'theta'},
  controls: [{key: 'theta', label: 'θ', min: 0, max: 180, step: 1, value: 0}]}));
near(pcaIt.vars.__pca__.candVar, 2, 1e-12, '控件 θ=0° 时以控件为准：方差 = sxx = 2');
near(pcaIt.vars.__pca__.candShare, 0.5, 1e-12, 'θ=0° 的方差 / λ1 = 0.5');
pcaIt.setControl('theta', 45); pcaIt.redraw();
near(pcaIt.vars.__pca__.candVar, 4, 1e-12, '拖到 45° 时方差达到 λ1 = 4');
near(pcaIt.vars.__pca__.candShare, 1, 1e-12, 'θ=45° 的方差 / λ1 = 1');
pcaIt.setControl('theta', 90); pcaIt.redraw();
near(pcaIt.vars.__pca__.candVar, 2, 1e-12, 'θ=90° 上的方差 = syy = 2');
/* data 生成：种子固定 ⇒ 每次重绘同一批点（图注里的数字才不会每次刷新都变），换 seed 才变 */
const pcaSeed = {kind: 'pca', data: {n: 8, x: 'randn()', y: 'randn()'}, candidate: {angle: 0}};
const seed1 = draw(pcaSeed);
const seed2 = draw(JSON.parse(JSON.stringify(pcaSeed)));
assert.deepEqual(seed1.plotData.pts, seed2.plotData.pts, '同一 seed 必须得到同一批点（可复现）');
const seed3 = draw(Object.assign({}, pcaSeed, {data: {n: 8, x: 'randn()', y: 'randn()', seed: 7}}));
assert.notDeepEqual(seed1.plotData.pts, seed3.plotData.pts, '换 seed 应得到不同的一批点');

/* ---- descent：f = x² + 4y²，∇f = (2x, 8y)，α = 0.1，p0 = (−3, 2)。手算逐步：
   p1 = (−3+0.6, 2−1.6) = (−2.4, 0.4)
   p2 = (−2.4+0.48, 0.4−0.32) = (−1.92, 0.08)
   p3 = (−1.92+0.384, 0.08−0.064) = (−1.536, 0.016)
   p4 = (−1.536+0.3072, 0.016−0.0128) = (−1.2288, 0.0032) ---- */
const descIt = draw({kind: 'descent', x: {min: -3, max: 3, points: 21}, y: {min: -3, max: 3, points: 21},
  contour: {expr: 'x*x + 4*y*y', levels: 6}, grad: {dfdx: '2*x', dfdy: '8*y'}, start: [-3, 2],
  controls: [{key: 'lr', label: 'α', min: 0.01, max: 0.9, step: 0.01, value: 0.1},
    {key: 'steps', label: '步数', type: 'number', min: 1, max: 60, step: 1, value: 4}],
  readouts: [{label: 'f', expr: '__gd__.f', fmt: '0.0000'}, {label: 'done', expr: '__gd__.done'}]});
assert.equal(descIt.plotData.path.length, 5, 'descent 轨迹 = 起点 + 4 步');
near(descIt.plotData.path[1][0], -2.4, 1e-12, '第 1 步 x');
near(descIt.plotData.path[1][1], 0.4, 1e-12, '第 1 步 y');
near(descIt.plotData.path[2][0], -1.92, 1e-12, '第 2 步 x');
near(descIt.plotData.path[2][1], 0.08, 1e-12, '第 2 步 y');
near(descIt.plotData.path[3][0], -1.536, 1e-12, '第 3 步 x');
near(descIt.plotData.path[3][1], 0.016, 1e-12, '第 3 步 y');
near(descIt.plotData.path[4][0], -1.2288, 1e-12, '第 4 步 x');
near(descIt.plotData.path[4][1], 0.0032, 1e-12, '第 4 步 y');
near(descIt.vars.__gd__.f0, 25, 1e-12, 'f(−3, 2) = 9 + 16 = 25');
assert.equal(descIt.vars.__gd__.grad, 'analytic', '给了 dfdx/dfdy 就用解析梯度');
assert.equal(descIt.vars.__gd__.escaped, 0, '定义域内不掉出去');
assert.equal(descIt.vars.__gd__.diverged, 0, '不发散');
assert.ok(descIt.vars.__gd__.f < descIt.vars.__gd__.f0, 'f 必须下降');
assert.equal(descIt.el.readouts.children[1].children[1].textContent, '4.00', 'done 读数 = 走过的步数');
/* α 变大到 0.6：第 1 步就越出定义域，如实标 escaped 而不是把点画到框外 */
descIt.setControl('lr', 0.6); descIt.redraw();
assert.equal(descIt.vars.__gd__.escaped, 1, 'α=0.6 时第 1 步就越出定义域');
descIt.setControl('lr', 0.1); descIt.redraw();
assert.equal(descIt.vars.__gd__.escaped, 0, 'α 调回来后不再越界');
/* 不给 grad：中心差分也要能下降，且第 1 步与解析值同量级 */
const descNum = draw({kind: 'descent', x: {min: -3, max: 3, points: 21}, y: {min: -3, max: 3, points: 21},
  contour: {expr: 'x*x + 4*y*y'}, start: [-2.5, 1.5],
  controls: [{key: 'lr', min: 0.01, max: 0.5, step: 0.01, value: 0.1}]});
assert.equal(descNum.vars.__gd__.grad, 'numeric', '没给 grad 就标 numeric');
assert.ok(descNum.vars.__gd__.f < descNum.vars.__gd__.f0, '数值梯度也应下降');
near(descNum.plotData.path[1][0], -2.5 - 0.1 * 2 * -2.5, 1e-9, '数值梯度第 1 步 x ≈ 解析值');

/* ---- surface3d：f = x² + y² 在 [−2,2]² 上取 9×9 网格 ⇒ 8×8 = 64 个面片、包围盒 12 条棱、
   z 从中心的 0 到角点的 8。正交投影 ⇒ 不暴露二维的像素↔数据映射（plotGeom = null） ---- */
const s3It = draw({kind: 'surface3d', x: {min: -2, max: 2, points: 9}, y: {min: -2, max: 2, points: 9},
  surface: {expr: 'x*x + y*y'}, view: {azimuth: 35, elevation: 25},
  controls: [{key: 'azimuth', label: '方位角', min: 0, max: 360, step: 1, value: 35},
    {key: 'elevation', label: '仰角', min: 0, max: 89, step: 1, value: 25}],
  readouts: [{label: 'zmax', expr: '__s3__.zmax', fmt: '0.000'}]});
near(s3It.vars.__s3__.zmax, 8, 1e-12, '曲面在角点取最大 z = 4 + 4 = 8');
near(s3It.vars.__s3__.zmin, 0, 1e-12, '中心 z = 0');
assert.equal(s3It.vars.__s3__.cells, 64, '8×8 = 64 个面片');
const s3Faces = findAll(s3It.el.figure, 'wg-s3-face');
assert.equal(s3Faces.length, 64, '每个面片一个 path');
s3Faces.forEach(f => assert.equal(numsOf(f.attrs.d).length, 8, '面片路径 = 4 个顶点 + Z（z 为 0 的面片也不能缺）'));
assert.equal(findAll(s3It.el.figure, 'wg-s3-edge').length, 12, '包围盒 12 条棱');
assert.equal(s3It.el.readouts.children[0].children[1].textContent, '8.000');
assert.equal(s3It.plotGeom, null, '三维不暴露二维像素↔数据映射（别让外部按 plotGeom 反解）');
/* 仰角拉到 89°（近乎俯视）：z 几乎不再提供竖直方向，图形退化但不应报错 */
s3It.setControl('elevation', 89); s3It.redraw();
assert.deepEqual(s3It.errors, [], '仰角 89° 仍不报错');
assert.equal(findAll(s3It.el.figure, 'wg-s3-face').length, 64, '换视角后仍是同样多的面片');
/* 点云模式：surface.points 走三维散点，不要求网格 */
const s3Cloud = draw({kind: 'surface3d', x: {min: -2, max: 2}, y: {min: -2, max: 2},
  surface: {points: [[0, 0, 0], [1, 1, 1], [-1, 0.5, 2]]}, view: {azimuth: 0, elevation: 30}});
assert.equal(s3Cloud.vars.__s3__.cells, 3, '点云模式 cells = 点数');
assert.equal(findAll(s3Cloud.el.figure, 'wg-s3-pt').length, 3, '点云 3 个点');

/* ---- treefit 1d：点 (0,1)(1,1)(2,3)(3,3)，切点 x=2。
   depth=0：单一叶子取全均值 2 ⇒ MSE = mean((y−2)²) = (1+1+1+1)/4 = 1；
   depth=1：按 x=2 切开后每个区间内 y 恒定 ⇒ MSE = 0 ---- */
const t1 = draw({kind: 'treefit', points: [[0, 1], [1, 1], [2, 3], [3, 3]],
  treefit: {mode: '1d', splits: [{at: 2}], depth: 0},
  controls: [{key: 'depth', label: '深度', min: 0, max: 1, step: 1, value: 0}],
  readouts: [{label: 'mse', expr: '__tree__.mse', fmt: '0.0000'}, {label: 'leaves', expr: '__tree__.leaves'}]});
near(t1.vars.__tree__.mse, 1, 1e-12, 'depth=0：单一叶子（全均值 2）⇒ MSE = 1');
assert.equal(t1.vars.__tree__.leaves, 1);
t1.setControl('depth', 1); t1.redraw();
near(t1.vars.__tree__.mse, 0, 1e-12, 'depth=1：按 x=2 切开后完全拟合 ⇒ MSE = 0');
assert.equal(t1.vars.__tree__.leaves, 2);
assert.equal(findAll(t1.el.figure, 'wg-tree-cut').length, 1, '画出一条切分线');
assert.equal(t1.el.readouts.children[1].children[1].textContent, '2.00');

/* ---- treefit 2d：5 个点按 x=0 分成两半。
   depth=1：左半全是 A、右半 B 有 2 个 + A 有 1 个 ⇒ 判右半为 B ⇒ 错 1 个、错误率 1/5；
   depth=0：整片多数类 B（3 个）⇒ 2 个 A 错 ---- */
const t2 = draw({kind: 'treefit', x: {min: -2, max: 2}, y: {min: -2, max: 2},
  treefit: {mode: '2d', splits: [{axis: 'x', at: 0}], depth: 1},
  points: [[-1, -1, 'A'], [-1.5, 1, 'A'], [1, -0.5, 'B'], [1.5, 1, 'B'], [1.2, 0.5, 'A']],
  controls: [{key: 'depth', min: 0, max: 1, step: 1, value: 1}],
  readouts: [{label: 'err', expr: '__tree__.err', fmt: '0.0000'}, {label: 'regions', expr: '__tree__.regions'}]});
assert.equal(t2.vars.__tree__.regions, 2, '一条 x 切分 ⇒ 2 个区域');
assert.equal(t2.vars.__tree__.wrong, 1, 'x>0 区域里 3 个点：2 个 B + 1 个 A ⇒ 错 1 个');
near(t2.vars.__tree__.err, 1 / 5, 1e-12, '训练错误率 = 1/5');
assert.equal(findAll(t2.el.figure, 'wg-tree-region').length, 2, '每个区域画一个底色框');
t2.setControl('depth', 0); t2.redraw();
assert.equal(t2.vars.__tree__.regions, 1);
assert.equal(t2.vars.__tree__.wrong, 2, '不切分时多数类 B（3 个）⇒ A 的 2 个点错');
/* 切分点落在所有区域之外：如实报 [错误]，不静默 */
const t2bad = render({kind: 'treefit', x: {min: -2, max: 2}, y: {min: -2, max: 2},
  treefit: {mode: '2d', splits: [{axis: 'x', at: 9}], depth: 1}, points: [[0, 0, 'A'], [1, 1, 'B']],
  controls: [{key: 'depth', min: 0, max: 1, step: 1, value: 1}]});
assert.ok(t2bad.errors.some(e => e.indexOf('没有落在任何区域') >= 0), '切分点落空必须报 [错误]');

/* ============================================================
   自动播放与内嵌精简模式（slim）+ `^` 守卫
   ============================================================ */

/* ---- 自动播放：animate:true ⇒ from=min、to=max、6 秒一趟、来回；时间→值的映射是纯函数式的，
   测试直接喂时刻（autoTick）就行 ---- */
const anim = draw({kind: 'plot', x: {min: 0, max: 10, points: 21},
  series: [{label: 'k', expr: 'k*x'}],
  controls: [{key: 'k', label: 'k', min: 1, max: 5, step: 0.5, value: 1, animate: true}]});
assert.equal(anim.autoList.length, 1, '有 animate 的控件进 autoList');
assert.equal(anim.autoList[0].from, 1);
assert.equal(anim.autoList[0].to, 5);
assert.equal(anim.autoList[0].seconds, 6);
assert.equal(anim.autoList[0].pingpong, true);
/* 自动开始只在真浏览器里发生：Node 假 DOM 没有 boolean 的 document.hidden */
assert.equal(anim.autoStart(), false, 'Node 假 DOM 里不该自动开始');
assert.equal(anim.autoPlaying(), false, '没开始就没有 auto 状态');
assert.equal(anim.autoStart({force: true}), true, 'force 下可以开始（测试/静态截图用）');
assert.equal(anim.autoPlaying(), true);
assert.equal(anim.auto.t0, anim.auto.last, 't0 与 last 同起点');
const animT0 = anim.auto.t0;
anim.autoTick(animT0);
assert.equal(anim.values.k, 1, '起点 = from');
anim.autoTick(animT0 + 3000);
assert.equal(anim.values.k, 3, '半程（3s/6s）= 中点 3');
assert.equal(anim.ctlRefs.k.node.value, '3', '滑杆位置跟着走');
assert.equal(anim.ctlRefs.k.valEl.textContent, '3.00', '标签文字跟着走（与手动拖动同一套格式化）');
anim.autoTick(animT0 + 6000);
assert.equal(anim.values.k, 5, '一趟走完 = to（pingpong 的折返点）');
anim.autoTick(animT0 + 9000);
assert.equal(anim.values.k, 3, '回程半程 = 中点');
anim.autoTick(animT0 + 12000);
assert.equal(anim.values.k, 1, '两趟回到起点（来回）');
assert.equal(anim.autoStop(), true);
assert.equal(anim.autoPlaying(), false);
assert.equal(anim.autoStop(), false, '重复 stop 不再返回 true');
/* 单向（pingpong:false）走完从头开始，且 autostart:false 要传到配置里 */
const anim2 = draw({kind: 'plot', x: {min: 0, max: 10, points: 21}, series: [{label: 'k', expr: 'k*x'}],
  controls: [{key: 'k', min: 0, max: 4, step: 1, value: 0,
    animate: {from: 0, to: 4, seconds: 4, pingpong: false, autostart: false}}]});
assert.equal(anim2.autoList[0].autostart, false, 'autostart:false 要传到配置里');
assert.equal(anim2.autoStart(), false, 'autostart:false 的控件不该自己开始');
assert.equal(anim2.autoStart({force: true}), true);
anim2.autoTick(anim2.auto.t0 + 2000);
assert.equal(anim2.values.k, 2, '单向：半程 = 2');
anim2.autoTick(anim2.auto.t0 + 5000);
assert.equal(anim2.values.k, 1, '单向：超过一趟后回到起点重来（5s → 相位 0.25 → 1）');
assert.equal(anim2.autoStop(), true);

/* setControl 的边界：不存在的 key 不写、非有限数不写、数值按 step 取整（否则标签会出现 3.7000000000000006） */
assert.equal(anim.setControl('nope', 1), false, '未知 key 返回 false');
assert.equal(anim.setControl('k', NaN), false, 'NaN 返回 false');
assert.equal(anim.setControl('k', Infinity), false, 'Infinity 返回 false');
anim.setControl('k', 2.26);
assert.equal(anim.values.k, 2.5, '按 step=0.5 取整');
assert.equal(anim.ctlRefs.k.node.value, '2.5', '滑杆位置同步');

/* ---- `^` 与 `**` 现在都是乘方（旧的"`^` 是位异或"守卫已删除）：contour 的 x^2 + y^2 直接能画 ---- */
const caretIt = draw({kind: 'contour', x: {min: -1, max: 1}, y: {min: -1, max: 1},
  contour: {expr: 'x^2 + y^2'}, aspect: 'equal'});
assert.deepEqual(caretIt.errors, [], 'x^2 现在就是乘方，不该再报 [错误]');
const noCaret = draw({kind: 'contour', x: {min: -1, max: 1}, y: {min: -1, max: 1}, contour: {expr: 'x*x + y*y'}});
assert.deepEqual(noCaret.errors, [], 'x*x 写法照旧可用');
/* x^2 与 pow(x, 2) 必须逐点完全等价（同一条引擎路径，不是"看着差不多"） */
[-2, -0.5, 0, 1.5, 3].forEach(xv => {
  assert.equal(caretIt.engine.eval('x^2', {x: xv}), caretIt.engine.eval('pow(x, 2)', {x: xv}), `x=${xv} 时 x^2 = pow(x,2)`);
  assert.equal(caretIt.engine.eval('x^2 + y^2', {x: xv, y: -xv}),
    caretIt.engine.eval('x*x + y*y', {x: xv, y: -xv}), `x=${xv} 时 x^2 + y^2 = x*x + y*y`);
});
/* 说明文字里提到 x^2 照旧不该报错（它本来就不是表达式字段） */
const caretProse = draw({kind: 'box', box: {values: [1, 2, 3]}, notes: ['原文里写 x^2，这里用 x*x 实现']});
assert.deepEqual(caretProse.errors, [], 'notes 里的 x^2 是说明文字，不该报表达式错误');

/* ---- 内嵌精简模式（slim）：只留标题 + 控件 + 图 + 图例 + 图注 + 读数；
   副标题、来源、说明清单、页脚都不建（它们应该写进源笔记） ---- */
const chromeFull = render({kind: 'box', box: {values: [1, 2, 3]}, subtitle: '副标题', notes: ['说明一条']});
const chromeSlim = render({kind: 'box', box: {values: [1, 2, 3]}, subtitle: '副标题', notes: ['说明一条']},
  {chrome: 'slim', note: 'x.md'});
assert.equal(chromeFull.slim, false);
assert.equal(chromeSlim.slim, true);
assert.equal(findAll(chromeFull.root, 'wg-notes').length, 1, '完整版有说明区');
assert.equal(findAll(chromeSlim.root, 'wg-notes').length, 0, '精简版没有说明区');
assert.equal(findAll(chromeFull.root, 'wg-sub').length, 1, '完整版有副标题');
assert.equal(findAll(chromeSlim.root, 'wg-sub').length, 0, '精简版没有副标题');
assert.equal(findAll(chromeFull.root, 'wg-foot').length, 1, '完整版有页脚');
assert.equal(findAll(chromeSlim.root, 'wg-foot').length, 0, '精简版没有页脚');
assert.equal(findAll(chromeSlim.root, 'wg-src').length, 0, '精简版没有来源链接');
assert.equal(findAll(chromeSlim.root, 'wg-title').length, 1, '精简版保留标题');
assert.equal(findAll(chromeSlim.root, 'wg-readouts').length, 1, '精简版保留读数区');
assert.equal(findAll(chromeSlim.root, 'wg-figure').length, 1, '精简版保留图区');
assert.equal(findAll(chromeSlim.root, 'wg-caps').length, 1, '精简版保留图注');
assert.equal(findAll(chromeSlim.root, 'wg-legend').length, 1, '精简版保留图例');

/* ---- det(…) 标记的“口径”行：完整版显示、精简版丢掉；活数字行与可操作提示两版都留 ---- */
const capTexts = it => findAll(it.root, 'wg-cap').map(el => el.textContent);
const capStarts = (texts, prefix) => texts.some(t => String(t).indexOf(prefix) === 0);
/* setCaption 本体的三类条目：普通字符串照旧、null/undefined 忽略、det(…) 的每一段都算口径行 */
const capsFull = render({kind: 'bars', bars: [{label: 'x', value: 1}]});
const capsSlim = render({kind: 'bars', bars: [{label: 'x', value: 1}]}, {chrome: 'slim', note: 'x.md'});
capsFull.setCaption(['a', null, undefined, 'b']);
assert.deepEqual(capTexts(capsFull), ['a', 'b'], '普通字符串数组照旧渲染，null/undefined 条目忽略');
capsFull.setCaption([WG.det('口径一', ['口径二', null]), '活着']);
assert.deepEqual(capTexts(capsFull), ['口径一', '口径二', '活着'], 'det(…) 支持多段/数组，完整版全渲染');
capsSlim.setCaption([WG.det('口径一', ['口径二', null]), '活着']);
assert.deepEqual(capTexts(capsSlim), ['活着'], '精简版只丢 det(…) 的口径行，其余条目照旧');
/* box：活数字行两版都在，Tukey 口径行只在完整版 */
const boxCapSpec = () => ({kind: 'box', box: {values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]}});
const boxCapsFull = capTexts(render(boxCapSpec()));
const boxCapsSlim = capTexts(render(boxCapSpec(), {chrome: 'slim', note: 'x.md'}));
assert.ok(capStarts(boxCapsFull, 'n='), '完整版有 box 的活数字行');
assert.ok(capStarts(boxCapsSlim, 'n='), '精简版保留 box 的活数字行');
assert.ok(capStarts(boxCapsFull, '箱 = Q1..Q3'), '完整版有 box 的 Tukey 口径行');
assert.ok(!capStarts(boxCapsSlim, '箱 = Q1..Q3'), '精简版丢掉 box 的 Tukey 口径行');
/* aspect 的口径行由 aspectCaption 统一标 det(…)：等比例说明只在完整版，f(x, y) … 活数字行两版都留 */
const aspCapSpec = () => ({kind: 'contour', contour: {expr: 'x*x + y*y'}, x: {min: -1, max: 1}, y: {min: -1, max: 1},
  aspect: 'equal'});
const aspCapFull = render(aspCapSpec()), aspCapSlim = render(aspCapSpec(), {chrome: 'slim', note: 'x.md'});
assert.deepEqual(aspCapFull.errors, [], 'aspect=equal 的 contour 不该有 [错误]（否则图注断言测的不是想测的东西）');
assert.ok(capStarts(capTexts(aspCapFull), '坐标轴：等比例'), '完整版有坐标轴等比例口径（aspectCaption 的 det）');
assert.ok(!capStarts(capTexts(aspCapSlim), '坐标轴：等比例'), '精简版丢掉坐标轴口径');
assert.ok(capStarts(capTexts(aspCapFull), 'f(x, y) =') && capStarts(capTexts(aspCapSlim), 'f(x, y) ='),
  'contour 的活数字行两版都在');

/* ---- canvas 后端（scatter / heatmap / surface3d 的密集标记）----
   假 DOM 现在有 canvas（Element.getContext 返回只记录调用的 2d 替身）与非零矩形，所以两条路都能
   真跑：canvas 路径断言"画了几笔"，SVG 路径断言"该有的 DOM 都在"，降级路径断言"静默退回 SVG"。
   阈值由 WG.canvasPolicy 公开，先把它钉住，免得以后有人悄悄改口径而测试还绿。 */
assert.deepEqual(WG.canvasPolicy.kinds, ['heatmap', 'scatter', 'surface3d'], '只有这三个 kind 有 canvas 后端');
assert.equal(WG.canvasPolicy.threshold.scatter, 1000);
assert.equal(WG.canvasPolicy.threshold.heatmap, 3600);
assert.equal(WG.canvasPolicy.threshold.surface3d, 1000);
assert.equal(WG.canvasPolicy.arcChunk, 256, '一条 canvas 路径里的最大圆数（超线性代价的护栏）');
const canvasEl = it => findAll(it.el.figure, 'wg-canvas')[0] || null;
const ctxOf = it => { const c = canvasEl(it); return c ? c.__ctx : null; };
const roText = it => it.el.readouts.children.map(card => card.children[1].textContent);

/* scatter：3 个点 auto 留 SVG（逐点 <title> 悬停提示还在），2001 个点 auto 换 canvas */
const scSmallSpec = () => ({kind: 'scatter', series: [{points: [[0, 1], [1, 2], [2, 3]]}],
  readouts: [{label: 'k', expr: '1 + 2'}]});
const scSmall = draw(scSmallSpec());
const scPts = Array.from({length: 2001}, (_, i) => [i, (i * 7) % 13]);
const scBigSpec = () => ({kind: 'scatter', series: [{points: scPts, label: 's'}],
  readouts: [{label: 'k', expr: '1 + 2'}]});
const scBig = draw(scBigSpec());
assert.equal(scBig.renderMode, 'canvas', '2001 个点 > 阈值 1000 ⇒ auto 换 canvas');
/* 小数据留在 SVG：阈值以下 canvas 没有速度优势，却要拿逐点悬停去换 —— 所以 600 点仍走 SVG */
assert.equal(draw({kind: 'scatter', series: [{points: scPts.slice(0, 600)}]}).renderMode, 'svg',
  '600 点 < 阈值 1000：宁可用 SVG 换逐点悬停');
const scSmall2 = draw(scSmallSpec());
assert.equal(scSmall2.renderMode, 'svg', '3 个点的散点图 auto 走 SVG');
assert.equal(canvasEl(scSmall2), null, 'SVG 路径不建 canvas');
assert.equal(findAll(scSmall2.el.figure, 'wg-dot').length, 3, 'SVG 路径逐点一个 <circle>');
assert.ok(ctxOf(scBig), 'canvas 路径要建 .wg-canvas 并拿到 2d 上下文');
assert.equal(scBig.el.figure.children[0], canvasEl(scBig), 'canvas 挂在 <svg> 之前（CSS 里 SVG 压在上面）');
assert.equal(String(scBig.el.figure.children[1].attrs.class), 'wg-svg', 'SVG 仍入图区、仍是排版元素');
assert.equal(findAll(scBig.el.figure, 'wg-dot').length, 0, 'canvas 路径不再逐点建 DOM');
/* 分块提交：**一条路径里的弧数不许超过 CANVAS_ARC_CHUNK**。
   这不是"实现细节"：单路径里塞满 arc 会让光栅化器把整条路径按统一高精度规则扁平化，
   代价随弧数超线性增长（实测 2 万个圆的单路径 247 秒，按 256 个一批 2.7 ms），
   曾经真的这么写过，所以这里用断言把它钉死。 */
const arcsPerPath = [];
{
  let pending = 0;
  ctxOf(scBig).calls.forEach(c => {
    if (c[0] === 'arc') pending++;
    else if (c[0] === 'fill') { arcsPerPath.push(pending); pending = 0; }
  });
}
assert.equal(arcsPerPath.length, Math.ceil(2001 / WG.canvasPolicy.arcChunk),
  '2001 个点按 arcChunk 分批，每批一次 fill');
assert.equal(Math.max.apply(null, arcsPerPath), WG.canvasPolicy.arcChunk,
  '每条路径的圆数不超过 arcChunk（' + WG.canvasPolicy.arcChunk + '）——不许写成"整张图一条路径"');
assert.ok(Math.max.apply(null, arcsPerPath) < 2001, '确实没有再出现"整张图一条路径"的写法');
assert.equal(ctxOf(scBig).count('moveTo'), 2001, '每点一次 moveTo：把子路径挪到圆弧起点，避免多余连线');
assert.ok(ctxOf(scBig).count('fill') < 2001 / 8, 'fill 次数远少于点数（批内合并提交）');
assert.equal(WG.canvasPolicy.dotR, 2.6, '圆点半径与 SVG 的 .wg-dot 一致');
assert.equal(ctxOf(scBig).count('arc'), 2001, '2001 个点 ⇒ 2001 次 arc');
assert.ok(ctxOf(scBig).count('setTransform') >= 1, '画之前要把 viewBox 坐标映射到设备像素');
assert.ok(canvasEl(scBig).width > 0 && canvasEl(scBig).height > 0, '后备存储按实测矩形×dpr 设置');
/* 同样的数据强行留 SVG：DOM 节点数回到 2001 个 <circle>，readouts 必须一模一样 */
const scBigSvg = draw(Object.assign(scBigSpec(), {renderer: 'svg'}));
assert.equal(scBigSvg.renderMode, 'svg', 'renderer:"svg" 强留 SVG（点数多也听）');
assert.equal(canvasEl(scBigSvg), null);
assert.equal(findAll(scBigSvg.el.figure, 'wg-dot').length, 2001, '强留 SVG 时逐点 DOM 照建');
assert.deepEqual(roText(scBig), roText(scBigSvg), '换后端不改 readouts');
assert.deepEqual(scBig.errors, [], '走 canvas 不该产生 [错误]');
/* 反方向：renderer:"canvas" 在小数据上也要真的走 canvas */
const scForced = draw(Object.assign(scSmallSpec(), {renderer: 'canvas'}));
assert.equal(scForced.renderMode, 'canvas', 'renderer:"canvas" 对小数据照样生效');
assert.equal(findAll(scForced.el.figure, 'wg-dot').length, 0);
assert.equal(ctxOf(scForced).count('arc'), 3);
assert.ok(capStarts(capTexts(scForced), '散点图：点画在 canvas 上'), 'canvas 路径的图注说明代价与回退办法');
assert.ok(capStarts(capTexts(scForced), '散点图：点画在 canvas 上') &&
  capStarts(capTexts(render(Object.assign(scSmallSpec(), {renderer: 'canvas'}), {chrome: 'slim', note: 'x.md'})),
    '散点图：点画在 canvas 上'), '这句是普通字符串（当前状态），内嵌精简模式也要看得见');

/* heatmap：格子数 > 3600 才换 canvas；可编辑一律留 DOM（常驻 <input> 只能活在 DOM 里） */
const heatSpec = (R, C, extra) => ({kind: 'heatmap', heat: Object.assign({
  rows: Array.from({length: R}, (_, i) => 'r' + i),
  cols: Array.from({length: C}, (_, i) => 'c' + i),
  values: Array.from({length: R}, (_, r) => Array.from({length: C}, (_, c) => r + c / 100)),
  bind: {rho: [0, 1]}}, extra || {}), readouts: [{label: 'rho', expr: 'rho', fmt: '0.00'}]});
assert.equal(draw(heatSpec(20, 20)).renderMode, 'svg', '400 格 auto 走 DOM 格子');
const hmBig = draw(heatSpec(101, 101));
assert.equal(hmBig.renderMode, 'canvas', '101×101 = 10201 格 > 阈值 3600 ⇒ canvas');
assert.ok(ctxOf(hmBig), 'heatmap 的 canvas 是 .wg-heat 网格里的一项');
assert.equal(findAll(hmBig.el.figure, 'wg-heat-cell').length, 0, 'canvas 路径不建格子 DOM');
assert.equal(findAll(hmBig.el.figure, 'wg-heat-lab').length, 1 + 101 + 101, '行/列标签仍是 DOM 文本');
assert.equal(ctxOf(hmBig).count('fillRect'), 10201, '每格一次 fillRect');
assert.ok(ctxOf(hmBig).count('measureText') <= 4, '按长度缓存测量：不为每个格子调 measureText');
assert.equal(hmBig.el.readouts.children[0].children[1].textContent, '0.01', 'heat.bind 的 readouts 照常生效');
const hmBigSvg = draw(Object.assign(heatSpec(101, 101), {renderer: 'svg'}));
assert.equal(hmBigSvg.renderMode, 'svg');
assert.equal(findAll(hmBigSvg.el.figure, 'wg-heat-cell').length, 10201, '强留 SVG 时逐格 DOM 照建');
assert.deepEqual(roText(hmBig), roText(hmBigSvg), '换后端不改 readouts');
/* 可编辑 + canvas：优先 DOM（不报错），并在图注里说明为什么 */
const hmEditable = draw(Object.assign(heatSpec(101, 101, {editable: true}), {renderer: 'canvas'}));
assert.equal(hmEditable.renderMode, 'svg', 'heat.editable 时 renderer:"canvas" 被忽略（编辑只能活在 DOM 里）');
assert.equal(canvasEl(hmEditable), null);
assert.equal(findAll(hmEditable.el.figure, 'wg-heat-cell').length, 10201);
assert.deepEqual(hmEditable.errors, [], '优先 DOM 是静默的，不报 [错误]');
assert.ok(capStarts(capTexts(hmEditable), 'heat.editable 为 true'), '为什么忽略 canvas：图注里有一条 det 说明');
assert.ok(capStarts(capTexts(draw(heatSpec(101, 101))), '格子画在 canvas 上'),
  '走 canvas 的 heatmap 图注写明没有逐格悬停 + renderer:"svg" 可以回去');

/* surface3d：面片数 > 1000 才换 canvas（33×33 ⇒ 32² = 1024 > 1000 换 canvas；32×32 ⇒ 31² = 961 留 SVG） */
const s3Spec = (n, extra) => Object.assign({kind: 'surface3d', x: {min: -1, max: 1, points: n},
  y: {min: -1, max: 1, points: n}, surface: {expr: 'x*x + y*y'},
  readouts: [{label: 'cells', expr: '__s3__.cells'}, {label: 'zmax', expr: '__s3__.zmax', fmt: '0.000'}]}, extra || {});
const s3Small = draw(s3Spec(32));
assert.equal(s3Small.renderMode, 'svg', '961 个面片 ≤ 1000 走 SVG');
assert.equal(findAll(s3Small.el.figure, 'wg-s3-face').length, 961);
const s3Big = draw(s3Spec(33));
assert.equal(s3Big.renderMode, 'canvas', '1024 个面片 > 1000 ⇒ canvas');
assert.equal(findAll(s3Big.el.figure, 'wg-s3-face').length, 0, 'canvas 路径不建逐面 <path>');
assert.equal(findAll(s3Big.el.figure, 'wg-s3-edge').length, 12, '包围盒 12 条棱仍在 SVG');
assert.ok(findAll(s3Big.el.figure, 'wg-s3-axlabel').length === 2 && findAll(s3Big.el.figure, 'wg-tick').length > 0,
  '轴标签与刻度文字仍在 SVG（不搬去 canvas）');
assert.equal(ctxOf(s3Big).count('beginPath'), 1024, '1024 个面片 ⇒ 1024 次 beginPath（每面一填一描）');
assert.equal(ctxOf(s3Big).count('stroke'), 1024);
const s3BigSvg = draw(s3Spec(33, {renderer: 'svg'}));
assert.deepEqual(roText(s3Big), roText(s3BigSvg), '换后端不改 __s3__ 的 readouts');
assert.deepEqual(s3Big.vars.__s3__, s3BigSvg.vars.__s3__, '换后端不改 __s3__ 本身');
/* 点云模式按点数算密集度：3 个点走 SVG，3000 个点换 canvas */
const cloudPts = Array.from({length: 3000}, (_, i) => [Math.cos(i) , Math.sin(i), i / 3000]);
const s3CloudBig = draw({kind: 'surface3d', x: {min: -2, max: 2}, y: {min: -2, max: 2},
  surface: {points: cloudPts}, readouts: [{label: 'cells', expr: '__s3__.cells'}]});
assert.equal(s3CloudBig.renderMode, 'canvas', '点云按点数判密集度（3000 > 1000）');
assert.equal(findAll(s3CloudBig.el.figure, 'wg-s3-pt').length, 0);
assert.equal(ctxOf(s3CloudBig).count('arc'), 3000);
assert.ok(capStarts(capTexts(s3CloudBig), '点云画在 canvas 上'), '点云模式的图注用词也跟着变');

/* 降级：没有 getContext / 上下文返回 null / 图区量不到尺寸 —— 三条都必须静默回 SVG */
function withProto(key, val, fn) {
  const saved = Element.prototype[key];
  Element.prototype[key] = val;
  try { return fn(); } finally { Element.prototype[key] = saved; }
}
const bigCanvasSpec = () => Object.assign(scBigSpec(), {renderer: 'canvas'});
[[ '老宿主没有 getContext', 'getContext', undefined],
 [ 'getContext 返回 null', 'getContext', function () { return null; }],
 [ '图区量不到尺寸（未排版/隐藏）', 'getBoundingClientRect',
   function () { return {left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0}; }]
].forEach(([what, key, val]) => {
  const it = withProto(key, val, () => draw(bigCanvasSpec()));
  assert.equal(it.renderMode, 'svg', what + '：静默退回 SVG（it.renderMode 记成 svg）');
  assert.equal(canvasEl(it), null, what + '：不留半张 canvas 在 DOM 里');
  assert.equal(findAll(it.el.figure, 'wg-dot').length, 2001, what + '：退回后逐点 DOM 照画');
  assert.deepEqual(it.errors, [], what + '：降级不报 [错误]，也不该白屏');
});

/* renderer 字段的口径：缺省 = auto，认不出的值按 auto 处理并说一声，其余 kind 一律 SVG */
assert.equal(draw(scSmallSpec()).renderMode, 'svg', '没有 renderer 字段 = auto');
assert.equal(draw(Object.assign(scSmallSpec(), {renderer: 'auto'})).renderMode, 'svg', 'renderer:"auto" 与不写等价');
const scUnknown = render(Object.assign(scBigSpec(), {renderer: 'CANVAS'}));
assert.equal(scUnknown.renderMode, scBig.renderMode, '认不出的 renderer 取值按 auto（大图照样换 canvas）');
assert.ok(scUnknown.errors.some(e => e.indexOf('renderer') >= 0), '但要说出来：值与 aspect / surface.mode 的口径一致');
const plainPlot = draw({kind: 'plot', x: {min: 0, max: 1, points: 5}, series: [{expr: 'x'}], renderer: 'canvas'});
assert.equal(plainPlot.renderMode, 'svg', '只有这三个 kind 认 renderer，其余一律 SVG');
assert.equal(canvasEl(plainPlot), null);

/* canvas 的 repaint 挂在正常重绘路径上（全量重绘 / 宽度变化），不另开定时器 */
const scRepaint = draw(bigCanvasSpec());
const scCtxOld = ctxOf(scRepaint);
assert.equal(scCtxOld.count('arc'), 2001, '首次绘制：2001 个点');
scRepaint.spec.series[0].points = Array.from({length: 2500}, (_, i) => [i, (i * 7) % 13]);
scRepaint.redraw();
assert.notEqual(ctxOf(scRepaint), scCtxOld, '全量重绘会重建 canvas 节点（图区整体清空重建）');
assert.equal(ctxOf(scRepaint).count('clearRect'), 1, '重绘时先清屏再画（不叠着上一次的图）');
assert.equal(ctxOf(scRepaint).count('arc'), 2500, '重绘用新的点数（canvas 与 readouts 走同一份数据）');
scRepaint.root.clientWidth = 300; scRepaint.layout();          // 宽度变化走正常重排
assert.equal(scRepaint.renderMode, 'canvas', '窄屏重排后仍在 canvas 路径上');
assert.equal(ctxOf(scRepaint).count('arc'), 2500, '重排后的重绘也把点画全（宽度变了就重画）');


/* ============================================================================
   表达式引擎（widgets.js 第 3 节）：自己分词 → 自己建 AST → 编译成闭包树。
   这里把"语言语义"钉死：优先级/结合性/乘方两种写法/除零/错误列号/宿主名隔离/缓存命中。
   这些断言是给改引擎的人看的护栏——它们一旦变红，说明"语言"变了，不是实现细节变了。
   ============================================================================ */
const eng = stat.engine;
const evx = e => eng.eval(e);
const errOf = e => { try { evx(e); return null; } catch (err) { return err.message; } };

/* 优先级与结合性 */
assert.equal(evx('1 + 2 * 3'), 7, '乘除先于加减');
assert.equal(evx('(1 + 2) * 3'), 9, '括号改变优先级');
assert.equal(evx('2 + 3 * 4 ^ 2'), 50, '乘方比乘除更紧：3*16');
assert.equal(evx('10 - 3 - 2'), 5, '减法左结合');
assert.equal(evx('100 / 10 / 2'), 5, '除法左结合');
assert.equal(evx('1 < 2 && 3 > 2'), true, '比较比 && 更紧');
assert.equal(evx('1 < 2 ? 7 : 8'), 7, '?: 优先级最低');
assert.equal(evx('!!0'), false, '一元 ! 可叠加');

/* 乘方：`^` 与 `**` 等价，都比一元负号紧，右结合（旧的"`^` 是位异或"守卫已撤） */
assert.equal(evx('2 ^ 3 ^ 2'), 512, '乘方右结合：2^(3^2)');
assert.equal(evx('2 ** 3 ** 2'), 512, '`**` 与 `^` 同义、同样右结合');
assert.equal(evx('-3 ^ 2'), -9, '乘方比一元负号紧：-(3^2)');
assert.equal(evx('-3 ** 2'), -9, '`**` 也是 -(3^2)');
assert.equal(evx('2 ^ -1'), 0.5, '指数位允许一元负号');
assert.equal(evx('x^2'), 0, '`x^2` 在 x=0 处求值为 0（不是位异或）');

/* 算术边界：与 JavaScript 一致，除零不抛异常 */
assert.equal(evx('1 / 0'), Infinity, '/0 ⇒ Infinity');
assert.equal(evx('-1 / 0'), -Infinity, '符号跟着分子');
assert.ok(Number.isNaN(evx('0 / 0')), '0/0 ⇒ NaN');
assert.ok(Number.isNaN(evx('1 % 0')), '1%0 ⇒ NaN');
/* 读数框把非有限值如实标出来（而不是显示 Infinity 或造假数字） */
const roNonFin = render({kind: 'plot', x: {min: 0, max: 1, points: 3}, series: [{expr: 'x'}],
  readouts: [{label: 'a', expr: '1 / 0'}, {label: 'b', expr: 'exp(0)'}]});
assert.deepEqual(roText(roNonFin), ['[错误] 结果不是有限数字', '1.00'],
  'readouts：非有限值报 [错误]，有限值照常格式化');

/* 数组字面量与 quantile：数组可以进函数，也可以参与算术 */
assert.equal(evx('quantile([1,2,3,4,5], 0.5)'), 3, '数组字面量 + 分位数');
assert.equal(evx('quantile([5], 0.5)'), 5, '单元素数组');
assert.equal(evx('quantile([1,3], 0.5) + 1'), 3, '数组调用的结果可以继续算');

/* 语法错误与未知名字：一律带 1 基列号，并且分得清"语法错"和"名字不存在" */
assert.match(errOf('1 +'), /第 4 列：表达式到这里就结束了/, '表达式截断：指出缺值的位置');
assert.match(errOf('(1 + 2'), /第 7 列：括号没闭合/, '括号没闭合');
assert.match(errOf('"abc'), /第 1 列：字符串少了收尾/, '字符串没收尾');
assert.match(errOf('2 + @'), /第 5 列：认不出的字符「@」/, '认不出的字符带列号');
assert.match(errOf('nosuchfn(1)'), /第 1 列：未知函数「nosuchfn」/, '未知函数在编译期就报');
assert.match(errOf('nosuchvar + 1'), /第 1 列：未知变量「nosuchvar」/, '未知变量在编译期就报');
assert.match(errOf('a.b'), /第 1 列：未知变量「a」/, '属性读取也要先有名字');

/* 宿主名到不了求值环境：不是"运行时报错"，而是编译期就拒绝，且只把它当普通标识符 */
assert.match(errOf('window.open(1)'), /第 12 列：多余的记号「\(」/, 'window.open(1)：window 不是宿主对象');
assert.match(errOf('Math.max(1,2)'), /第 9 列：多余的记号「\(」/, 'Math 也一样，必须先走白名单函数');
assert.match(errOf('ifelse(true, 1, nosuchfn(2))'), /第 17 列：未知函数「nosuchfn」/,
  'ifelse 的参数照样先编译：没被选中的分支也要求存在（急切求值）');

/* AST 本身是公开可见、可 JSON 化的结构（列号 1 基，供工具定位） */
assert.deepEqual(WG.parse('1 + 2 * 3'), {t: 'bin', op: '+', col: 3,
  l: {t: 'num', v: 1, col: 1},
  r: {t: 'bin', op: '*', col: 7, l: {t: 'num', v: 2, col: 5}, r: {t: 'num', v: 3, col: 9}}},
  'AST 结构：+ 在根上，右侧是 *（优先级进结构，不靠括号）');
assert.deepEqual(WG.parse('1 + 2'), WG.parse('1 + 2'), 'parse 是纯函数（同样输入同样输出）');
assert.deepEqual(WG.analyze('a + b + sin(c) + sin(d)'), {vars: ['a', 'b', 'c', 'd'], calls: ['sin']},
  'analyze：读到的变量与被调函数（排序去重）');
const lint2 = WG.lintSpec({kind: 'plot', x: {min: 0, max: 1, points: 3}, series: [{expr: 'x'}],
  readouts: [{label: 'a', expr: 'nosuchfn(1)'}, {label: 'b', expr: 'nosuchvar'}]});
assert.equal(lint2.errors.length, 2, 'lintSpec 逐条列出错误');
assert.deepEqual(lint2.unknownFns, ['nosuchfn'], 'lintSpec 另外给出未知函数清单');
assert.deepEqual(lint2.unknownVars, ['nosuchvar'], '以及未知变量清单');

/* 编译缓存：同一份"名字表 + 表达式"只编译一次，拖动滑杆与反复重绘都不再花编译时间 */
const snapStats = () => [WG.__stats.compiles, WG.__stats.cacheHits];
const dStats = a => { const b = snapStats(); return {c: b[0] - a[0], h: b[1] - a[1]}; };
const sameSpec = () => ({kind: 'plot', x: {min: 0, max: 1, points: 3}, series: [{expr: 'x'}],
  readouts: [{label: 'a', expr: 'x * 2 + 1'}]});
let sstat = snapStats(); render(sameSpec()); const cFirst = dStats(sstat);
sstat = snapStats(); render(sameSpec()); const cSecond = dStats(sstat);
assert.ok(cFirst.c > 0, '首次渲染要编译（series.expr 与 readouts.expr 各一次）');
assert.equal(cSecond.c, 0, '同样的 spec 再渲染一次：一次编译都不做，全走缓存');
assert.ok(cSecond.h > 0, '并且记到了缓存命中（WG.__stats.cacheHits）');
/* 网格同理：整张 121×121 的等高线只编译 1 次表达式，之后反复重绘 0 次 —— 不是"每个格子编译一次"。
   先渲染一遍当热身，第二遍的计数才干净（测试文件前面已经渲染过别的 contour，缓存是全局的，
   所以这里断言的是"第二次 0 编译 + 大量命中"，而不是"第一次一定编译 1 次"）。 */
const gridSpec = {kind: 'contour', x: {min: -2, max: 2, points: 121}, y: {min: -2, max: 2, points: 121},
  contour: {expr: 'x^2 + y^2'}};
sstat = snapStats(); const gridWarm = render(gridSpec); const gWarm = dStats(sstat);
sstat = snapStats(); const gridIt = render(gridSpec); const gSecond = dStats(sstat);
assert.deepEqual(gridIt.errors, [], '121² 网格能画（AST 引擎下无 [错误]）');
assert.ok(gWarm.c <= 1, '整张 121² 网格最多编译一次表达式（实际 ' + gWarm.c + ' 次）');
assert.equal(gSecond.c, 0, '同规格再渲染：0 次编译（全走缓存）');
assert.ok(gSecond.h > 20000, '并且是两万次以上的缓存命中（14641 个格子 × 每格多次求值）');
assert.deepEqual(roText(gridIt), roText(gridWarm), '缓存不改变结果（两次渲染读数一致）');
sstat = snapStats();
for (let i = 0; i < 5; i++) gridIt.redraw();
assert.equal(dStats(sstat).c, 0, '连续 5 次重绘仍然 0 次编译');

/* canvas 后端的公开口径：阈值、分块、像素上限、点半径全部钉死（渲染路径不读这里，改了要连测试一起改） */
assert.deepEqual(WG.canvasPolicy, {kinds: ['heatmap', 'scatter', 'surface3d'],
  threshold: {scatter: 1000, heatmap: 3600, surface3d: 1000}, maxPx: 8e6, arcChunk: 256, dotR: 2.6},
  'WG.canvasPolicy：哪些 kind 有 canvas、auto 阈值、单路径最大圆数、后备存储像素上限、点半径');


/* ---- 修复回归：treefit 2d 的 split 边界左闭右开、外边界闭右 ---- */
function regionPointTotal(it) {
  return it.vars.__tree__.regions ? it.plotData.regions.reduce((s, r) => s + r.n, 0) : 0;
}
const t2xBoundary = draw({kind: 'treefit', x: {min: -1, max: 1}, y: {min: -1, max: 1},
  treefit: {mode: '2d', splits: [{axis: 'x', at: 0}], depth: 1},
  points: [[0, 0.2, 'B'], [-0.5, 0.2, 'A'], [0.5, -0.2, 'B']],
  controls: [{key: 'depth', min: 0, max: 1, step: 1, value: 1}]});
assert.equal(t2xBoundary.vars.__tree__.wrong, 0, 'x=split 的点只进右侧区域，不重复计错');
assert.equal(t2xBoundary.vars.__tree__.err, 0, 'x=split 的点只计一次错误率');
assert.equal(regionPointTotal(t2xBoundary), 3, 'x split 后各区域样本数之和仍等于 n');
assert.equal(t2xBoundary.plotData.regions[0].n, 1, 'x<split 的左区域不含边界点');
assert.equal(t2xBoundary.plotData.regions[1].n, 2, 'x=split 的点归到右区域');
const t2yBoundary = draw({kind: 'treefit', x: {min: -1, max: 1}, y: {min: -1, max: 1},
  treefit: {mode: '2d', splits: [{axis: 'y', at: 0}], depth: 1},
  points: [[-0.2, 0, 'B'], [-0.2, -0.5, 'A'], [0.2, 0.5, 'B']],
  controls: [{key: 'depth', min: 0, max: 1, step: 1, value: 1}]});
assert.equal(t2yBoundary.vars.__tree__.wrong, 0, 'y=split 的点只进上侧区域，不重复计错');
assert.equal(t2yBoundary.vars.__tree__.err, 0, 'y=split 的点只计一次错误率');
assert.equal(regionPointTotal(t2yBoundary), 3, 'y split 后各区域样本数之和仍等于 n');

/* ---- 修复回归：surface.expr + mode=points 只画网格点；三维 marker 支持数值/表达式 ---- */
const s3ExprPoints = draw({kind: 'surface3d', x: {min: -1, max: 1, points: 4}, y: {min: -1, max: 1, points: 4},
  surface: {expr: 'x + y', mode: 'points'},
  vars: {mx: 0}, controls: [{key: 'shift', type: 'number', value: 0}],
  markers: [{x: 0, y: 0, z: 0, label: 'origin'}, {x: 'shift + 1', y: '-1', z: '0', label: 'edge'}]});
assert.equal(findAll(s3ExprPoints.el.figure, 'wg-s3-face').length, 0, 'surface.mode=points 不建面片');
assert.equal(findAll(s3ExprPoints.el.figure, 'wg-s3-pt').length, 16, 'surface.expr 的 points 模式逐网格点投影');
assert.equal(findAll(s3ExprPoints.el.figure, 'wg-s3-marker').length, 2, '三维 marker 投影为 .wg-s3-marker');
assert.equal(findAll(s3ExprPoints.el.figure, 'wg-s3-marker-label').length, 2, '有 label 的 marker 画出文字');
assert.deepEqual(s3ExprPoints.errors, [], '合法三维 marker 不应报“二维竖线标记”错误');
function s3Fingerprint(it) {
  return JSON.stringify(findAll(it.el.figure, 'wg-s3-face').concat(findAll(it.el.figure, 'wg-s3-pt'))
    .map(e => [e.attrs.d, e.attrs.cx, e.attrs.cy, e.attrs.r, e.attrs.fill]));
}

/* ---- AdaBoost 动态 surface：权重面由 renderer 按笔记三轮递推生成 D1–D4 ---- */
const adaSpec = () => ({kind: 'surface3d', title: 'AdaBoost weight surface',
  x: {min: 1, max: 4, points: 4}, y: {min: 1, max: 4, points: 4},
  surface: {mode: 'surface'},
  controls: [
    {key: 'errorMode', type: 'select', options: [['note', '笔记误差序列'], ['fixed', '固定 e=0.40']], value: 'note'},
    {key: 'shrinkage', type: 'slider', min: 0.1, max: 1, step: 0.1, value: 1}
  ],
  adaboost: {rounds: 3, errorModeKey: 'errorMode', shrinkageKey: 'shrinkage',
    data: {labels: [1, -1, -1, 1], learners: [
      {predictions: [1, -1, 1, 1]},
      {predictions: [-1, -1, -1, 1]},
      {predictions: [1, -1, 1, 1]}
    ], sampleIndex: [1, 2, 3, 4]},
    errorModes: {note: {errors: [0.25, 1 / 6, 0.30]}, fixed: {error: 0.40}}},
  readouts: [{label: 'alpha', expr: '__ada__.alphaLast', fmt: '0.0000'},
    {label: 'upper', expr: '__ada__.upperBoundLast', fmt: '0.0000'},
    {label: 'error', expr: '__ada__.error01', fmt: '0.00%'}]});
const adaIt = draw(adaSpec());
const ada = adaIt.vars.__ada__;
near(ada.weights[0][0], 0.25, 1e-12, 'Ada D1[1] = 1/4');
near(ada.weights[1][2], 0.5, 1e-12, 'Ada D2[3] = 1/2');
near(ada.weights[3][0], 5 / 14, 1e-12, 'Ada D4[1] = 5/14');
near(ada.weights[3][1], 1 / 14, 1e-12, 'Ada D4[2] = 1/14');
assert.equal(ada.weights.length, 4, 'Ada 曲面包含初始 D1 与三轮更新后的 D2–D4');
assert.equal(ada.weights[3].length, 4, 'Ada D4 包含四个样本权重');
near(ada.rawAlpha[0], 0.5 * Math.log(3), 1e-12, 'Ada α1 = 1/2 ln 3');
near(ada.rawAlpha[1], 0.5 * Math.log(5), 1e-12, 'Ada α2 = 1/2 ln 5');
near(ada.rawAlpha[2], 0.5 * Math.log(7 / 3), 1e-12, 'Ada α3 = 1/2 ln(7/3)');
near(ada.boundFactor[0], 2 * Math.sqrt(0.25 * 0.75), 1e-12, 'Ada Z1 = 2√(e1(1−e1))');
near(ada.upperBoundLast, 0.8660254037844386 * 0.7453559924999299 * 0.9165151398480992, 1e-9, 'Ada 三轮上界');
assert.deepEqual(ada.error01Path, [0.25, 0.25, 0.25], 'Ada 三轮实际 0/1 误差都为 1/4');
assert.equal(adaIt.vars.__ada__.error01, 0.25, 'Ada 末轮错误率读数是标量');
assert.equal(findAll(adaIt.el.figure, 'wg-s3-face').length, 9, 'Ada 4×4 网格画 9 个面片');
const adaFp0 = s3Fingerprint(adaIt);
adaIt.values.errorMode = 'fixed'; adaIt.redraw();
assert.equal(adaIt.vars.__ada__.errorMode, 'fixed', 'errorMode 控件进入 Ada 实际计算');
assert.notEqual(s3Fingerprint(adaIt), adaFp0, 'errorMode 改变主图状态');
const adaFp1 = s3Fingerprint(adaIt);
adaIt.values.shrinkage = 0.5; adaIt.redraw();
near(adaIt.vars.__ada__.appliedAlpha[0], adaIt.vars.__ada__.rawAlpha[0] * 0.5, 1e-12, 'shrinkage 乘入 applied alpha');
assert.notEqual(s3Fingerprint(adaIt), adaFp1, 'shrinkage 改变主图状态');
assert.ok(adaIt.el.caps.children.map(c => c.textContent).join(' ').includes('按笔记算例确定性重算'), 'Ada 图注明确确定性重算');

/* ---- Gradient Boosting 动态 surface：四点树桩递推，ν 改变整张预测面 ---- */
const gbSpec = () => ({kind: 'surface3d', title: 'Gradient boosting prediction surface',
  x: {min: 1, max: 4, points: 4}, y: {min: 0, max: 143, points: 121}, surface: {mode: 'surface'},
  controls: [{key: 'nu', type: 'select', options: [['0.1', 'ν=0.1'], ['0.5', 'ν=0.5']], value: '0.1'}],
  boosting: {rounds: 143, nuKey: 'nu', data: {response: [5, 7, 6, 10]},
    metrics: {targetMSE: [1, 0.5, 0.1, 0.01]}},
  readouts: [{label: 'firstMSE', expr: '__gb__.firstMSE', fmt: '0.0000'},
    {label: 'mse', expr: '__gb__.mse', fmt: '0.0000'},
    {label: 'rounds001', expr: '__gb__.rounds001', fmt: '0'}]});
const gbIt = draw(gbSpec());
const gb01 = gbIt.vars.__gb__;
assert.deepEqual(gb01.F[0], [7, 7, 7, 7], 'GBM F0 = 7');
near(gb01.F[1][0], 6.9, 1e-12, 'GBM ν=.1 第 1 个预测 = 6.9');
near(gb01.F[1][3], 7.3, 1e-12, 'GBM ν=.1 第 4 个预测 = 7.3');
near(gb01.firstMSE, 2.93, 1e-12, 'GBM ν=.1 首轮 MSE = 2.93');
assert.deepEqual(gb01.targetRounds, [9, 15, 51, 143], 'GBM ν=.1 目标轮数');
assert.equal(gb01.split[0], 3, 'GBM 首轮最小 SSE 切分是 x=3 之后');
const gbFp01 = s3Fingerprint(gbIt);
gbIt.values.nu = '0.5'; gbIt.redraw();
const gb05 = gbIt.vars.__gb__;
near(gb05.F[1][0], 6.5, 1e-12, 'GBM ν=.5 第 1 个预测 = 6.5');
near(gb05.F[1][3], 8.5, 1e-12, 'GBM ν=.5 第 4 个预测 = 8.5');
near(gb05.firstMSE, 1.25, 1e-12, 'GBM ν=.5 首轮 MSE = 1.25');
assert.deepEqual(gb05.targetRounds, [2, 3, 8, 23], 'GBM ν=.5 目标轮数');
assert.notEqual(s3Fingerprint(gbIt), gbFp01, 'nu 改变主图状态并重算整张面');
assert.ok(gbIt.el.caps.children.map(c => c.textContent).join(' ').includes('按笔记算例确定性重算'), 'GBM 图注明确确定性重算');

console.log('Runtime regression tests passed (statistics, histogram redraw, heat edit/readouts, box/ecdf/qq, ' +
  'contour/vector/matrix, aspect, regression/pca/descent/surface3d/treefit, control auto-play, ' +
  'slim chrome, caption det detail, canvas backend, expression AST, treefit boundary, surface points/markers, ' +
  'AdaBoost dynamic surface, Gradient Boosting dynamic surface).');
