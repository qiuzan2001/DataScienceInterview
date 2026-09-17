'use strict';

/* Plotly adapter tests use a tiny DOM and fake Promise-returning Plotly API.
   They intentionally do not load either vendored bundle or make network requests. */
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ADAPTER_PATH = path.join(__dirname, 'plotly-adapter.js');
const ADAPTER_SOURCE = fs.readFileSync(ADAPTER_PATH, 'utf8');
let testCount = 0;
let assertionCount = 0;

function check(condition, message) {
  assertionCount += 1;
  assert.ok(condition, message);
}

function equal(actual, expected, message) {
  assertionCount += 1;
  assert.strictEqual(actual, expected, message);
}

function element(tag) {
  const node = { tagName: String(tag).toUpperCase(), className: '', parentNode: null, children: [] };
  Object.defineProperty(node, 'firstChild', {
    enumerable: true,
    get: function () { return node.children[0] || null; }
  });
  node.appendChild = function (child) {
    if (child.parentNode && child.parentNode.removeChild) child.parentNode.removeChild(child);
    node.children.push(child);
    child.parentNode = node;
    return child;
  };
  node.removeChild = function (child) {
    const i = node.children.indexOf(child);
    if (i >= 0) node.children.splice(i, 1);
    child.parentNode = null;
    return child;
  };
  return node;
}

function harness(options) {
  options = options || {};
  const calls = [];
  const registered = {};
  const errors = [];
  const visibleErrors = [];
  const seen = Object.create(null);
  const captions = [];
  let flushCount = 0;
  let layoutCount = 0;
  const figure = element('div');
  figure.clientWidth = options.width || 640;

  const meta = options.vendorPath ? {
    getAttribute: function (name) {
      return name === 'content' ? options.vendorPath : null;
    }
  } : null;
  const document = {
    createElement: element,
    querySelector: function (selector) {
      return selector === 'meta[name="wg-vendor-path"]' ? meta : null;
    }
  };

  const WG = {
    registerKind: function (name, fn) { registered[name] = fn; },
    det: function () {
      return { wgCaptionDetail: Array.prototype.slice.call(arguments) };
    }
  };

  const defaultNewPlot = function (gd) { return Promise.resolve(gd); };
  const defaultReact = function (gd) { return Promise.resolve(gd); };
  const plotly = Object.prototype.hasOwnProperty.call(options, 'plotly') ? options.plotly : {
    version: '3.7.0',
    newPlot: function (gd, data, layout, config) {
      calls.push({ method: 'newPlot', gd: gd, data: data, layout: layout, config: config });
      return (options.newPlot || defaultNewPlot)(gd, data, layout, config);
    },
    react: function (gd, data, layout, config) {
      calls.push({ method: 'react', gd: gd, data: data, layout: layout, config: config });
      return (options.react || defaultReact)(gd, data, layout, config);
    },
    purge: function (gd) { calls.push({ method: 'purge', gd: gd }); }
  };

  const context = {
    console: console,
    Promise: Promise,
    WeakMap: WeakMap,
    document: document,
    WG: WG,
    Plotly: plotly,
    setTimeout: setTimeout,
    clearTimeout: clearTimeout
  };
  context.window = context;
  vm.runInNewContext(ADAPTER_SOURCE, context, { filename: ADAPTER_PATH });

  const it = {
    el: { figure: figure },
    vars: {},
    theme: { c: [] },
    ev: function (expr) { return expr; },
    setCaption: function (parts) { captions.push(parts); },
    figureWidth: function () { return figure.clientWidth; },
    layout: function () { layoutCount += 1; },
    err: function (message) {
      const full = '[错误] ' + String(message);
      if (!seen[full]) {
        seen[full] = true;
        errors.push(full);
      }
    },
    flushErrors: function () {
      flushCount += 1;
      visibleErrors.splice(0, visibleErrors.length);
      Array.prototype.push.apply(visibleErrors, errors);
    }
  };

  return {
    draw: registered.plotly,
    calls: calls,
    errors: errors,
    visibleErrors: visibleErrors,
    captions: captions,
    figure: figure,
    it: it,
    get flushCount() { return flushCount; },
    get layoutCount() { return layoutCount; }
  };
}

function spec(type, bundle) {
  const is3d = type === 'surface';
  return {
    schema: 'widget/v1',
    kind: 'plotly',
    uid: '',
    vars: {},
    plotly: {
      data: is3d ? [{ type: 'surface', x: [0, 1], y: [0, 1], z: [[0, 1], [1, 2]] }] :
        [{ type: 'scatter', x: [0, 1], y: [1, 2] }],
      layout: { title: 'adapter must own the page title', xaxis: { title: 'x' } },
      config: { displayModeBar: false },
      bundle: bundle
    }
  };
}

function draw(h, s) {
  h.draw({ it: h.it, spec: s, narrow: false, W: h.it.figureWidth() });
}

function captionText(h) {
  return h.captions.map(function (parts) {
    return parts.map(function (part) {
      if (part && typeof part === 'object' && part.wgCaptionDetail) {
        return part.wgCaptionDetail.join(' ');
      }
      return String(part);
    }).join(' ');
  }).join(' ');
}

async function settle() {
  await Promise.resolve();
  await new Promise(function (resolve) { setImmediate(resolve); });
  await Promise.resolve();
}

async function test(name, fn) {
  await fn();
  testCount += 1;
  console.log('ok - ' + name);
}

(async function () {
  await test('registers the plotly renderer', function () {
    const h = harness();
    check(typeof h.draw === 'function', 'WG.registerKind should expose drawPlotly');
  });

  await test('successful newPlot preserves layout/config and has no error flush', async function () {
    const h = harness({ vendorPath: '_meta/tools/vendor/plotly.min.js' });
    draw(h, spec('scatter', 'full'));
    equal(h.calls.length, 1, 'success should call Plotly once');
    equal(h.calls[0].method, 'newPlot', 'first draw should use newPlot');
    equal(h.calls[0].config.displayModeBar, false, 'author config should survive');
    equal(h.calls[0].config.responsive, false, 'adapter should disable Plotly internal resize and let widgets.js own reflow');
    equal(h.calls[0].layout.title, undefined, 'page title should own the title');
    equal(h.calls[0].layout.height, 461, 'default height should follow figure width');
    h.it.flushErrors();
    equal(h.flushCount, 1, 'the simulated widgets.js flush should run once');
    await settle();
    equal(h.layoutCount, 1, 'resolved Plotly promise should refresh layout once');
    equal(h.flushCount, 1, 'success should not schedule a second error flush');
    equal(h.visibleErrors.length, 0, 'successful rendering should remain error-free');
  });

  await test('react reuses the existing graph div', async function () {
    const h = harness();
    const s = spec('scatter', 'full');
    draw(h, s);
    await settle();
    draw(h, s);
    await settle();
    equal(h.calls[0].method, 'newPlot', 'initial render should use newPlot');
    equal(h.calls[1].method, 'react', 'second render should use react');
    check(h.calls[0].gd === h.calls[1].gd, 'react should receive the same graph div');
    equal(h.figure.children.length, 1, 'react reuse should not duplicate graph divs');
    equal(h.errors.length, 0, 'react success should not report errors');
  });

  await test('gl3d keeps 3D mode and bundle labeling', async function () {
    const h = harness({ vendorPath: '_meta/tools/vendor/plotly-gl3d.min.js' });
    draw(h, spec('surface', 'gl3d'));
    await settle();
    equal(h.calls[0].data[0].type, 'surface', '3D trace should pass through unchanged');
    equal(h.it.vars.__plotly__.mode, '3d', '3D trace should publish 3d mode');
    check(captionText(h).indexOf('plotly-gl3d.min.js') >= 0, 'gl3d page metadata should be reflected in caption');
  });

  await test('full keeps 2D mode and bundle labeling', async function () {
    const h = harness({ vendorPath: '_meta/tools/vendor/plotly.min.js' });
    draw(h, spec('scatter', 'full'));
    await settle();
    equal(h.calls[0].data[0].type, 'scatter', '2D trace should pass through unchanged');
    equal(h.it.vars.__plotly__.mode, '2d', '2D trace should publish 2d mode');
    check(captionText(h).indexOf('plotly.min.js') >= 0, 'full page metadata should be reflected in caption');
  });

  await test('async Plotly rejection is visible after the initial flush', async function () {
    const h = harness({
      newPlot: function () { return Promise.reject(new Error('simulated Plotly failure')); }
    });
    draw(h, spec('surface', 'gl3d'));
    h.it.flushErrors();
    equal(h.flushCount, 1, 'widgets.js should have flushed before the Promise settles');
    equal(h.visibleErrors.length, 0, 'the rejection is not synchronous');
    await settle();
    equal(h.layoutCount, 1, 'rejection should still refresh layout');
    equal(h.flushCount, 2, 'rejection should schedule a safe secondary flush');
    equal(h.visibleErrors.length, 1, 'one visible error should be published');
    equal(h.visibleErrors[0], '[错误] Plotly 渲染失败：simulated Plotly failure',
      'visible output should preserve the rejection message');
    console.log('async rejection visible output: ' + h.visibleErrors.join(' | '));
  });

  await test('async react rejection is visible after graph reuse', async function () {
    const h = harness({
      react: function () { return Promise.reject(new Error('simulated react failure')); }
    });
    const s = spec('scatter', 'full');
    draw(h, s);
    await settle();
    h.it.flushErrors();
    draw(h, s);
    equal(h.calls.length, 2, 'react rejection should follow the initial successful render');
    equal(h.calls[1].method, 'react', 'rejection should exercise the react path');
    h.it.flushErrors();
    equal(h.flushCount, 2, 'widgets.js should flush before the react Promise settles');
    equal(h.visibleErrors.length, 0, 'the react rejection is not synchronous');
    await settle();
    equal(h.flushCount, 3, 'react rejection should schedule a safe secondary flush');
    equal(h.visibleErrors.length, 1, 'one react error should be published');
    equal(h.visibleErrors[0], '[错误] Plotly 渲染失败：simulated react failure',
      'react rejection should preserve its message');
    console.log('async react rejection visible output: ' + h.visibleErrors.join(' | '));
  });

  await test('StateFarm path diagnostics do not mention Maps/_tools', function () {
    check(ADAPTER_SOURCE.indexOf('Maps/_tools') < 0, 'adapter source must not hardcode the old tool path');
    const h = harness({ plotly: null });
    draw(h, spec('scatter', 'full'));
    h.it.flushErrors();
    check(h.visibleErrors.some(function (message) {
      return message.indexOf('_meta/tools/vendor') >= 0;
    }), 'missing Plotly error should name the StateFarm tool directory');
    check(h.visibleErrors.every(function (message) {
      return message.indexOf('Maps/_tools') < 0;
    }), 'visible diagnostics must not name Maps/_tools');
  });

  console.log(testCount + ' tests passed (' + assertionCount + ' assertions)');
})().catch(function (error) {
  console.error(error && error.stack ? error.stack : error);
  process.exitCode = 1;
});
