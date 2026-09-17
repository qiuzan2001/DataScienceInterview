# widget-embed —— 把交互组件内嵌进笔记的 Dataview 视图

笔记里这样用（`dataviewjs` 代码块）：

````markdown
```dataviewjs
await dv.view("Maps/_tools/widget-embed", { file: "交互组件示例-箱线图.html" })
```
````

- `file`：`Maps/_widgets/` 下的组件文件名；写 vault 相对完整路径也行。写完整 URL 或含 `..` 会被拒绝并显示 `[错误]`。
- `height`：可选。不写 = **自适应**（高度随组件内容，图不会被裁、也没有上下滚动条）；写数字 = 固定高度。
- `minHeight` / `maxHeight`：可选，自适应模式的上下限（默认 140 / 不限）。
- `resizable`：设为 `false` 去掉右下角拖拽把手（默认显示）。
- `chrome`：可选，`"slim"`（**默认**）或 `"full"`。`slim` = **内嵌只留交互图**：往组件页注入
  `<meta name="wg-chrome" content="slim">`，组件就不再渲染副标题、来源链接、`notes` 说明清单和页脚，
  只留标题、控件、图、图例、图注与读数；同时注入一段兜底样式，把旧组件页里的那几块也一起隐藏。
  口径与出处应当写在笔记正文里；`notes` 仍必须写全，因为**单独打开组件页时看到的是完整版**（不注入 meta）。
 想在笔记里要回完整版：`{ file: "…", chrome: "full" }`。

## 尺寸怎么定的

**自适应高度**由三条一起保证：

1. 本视图量 iframe 内部的内容高度：`body` 盒高 / `body.scrollHeight` / `.wg-wrap` 盒高取最大。
   **特意不看 `documentElement.scrollHeight`** —— 它是 `max(内容, 视口高)`，iframe 一旦比内容高，
   这个值就等于视口高，加上 2px 余量后每次重算都会把 iframe 撑高 2px，无界递增。
2. 组件自己 `postMessage({wg:"widget-height"})` 当**触发器**（`widgets.js` 每次重绘/重排后都发）。
   为什么需要它：宿主窗口被遮挡时 Chromium 会推迟渲染生命周期（`rAF` / `ResizeObserver` 都不投递），
   本视图就收不到「该重新量了」的信号；`postMessage` 由事件循环投递，什么时候都到。
   被沙箱化成不同源时读不到 iframe 内部文档，它更是唯一的高度来源。
   同源时仍以现场实测为准，消息只负责「叫醒」宿主。
3. 改高度后再量一轮（最多连量 3 轮）：高度一变，内部可能多出/少掉一条滚动条，宽度跟着变、组件重排，
   内容高度又变（实测能差 60px，表现为底部一大片空白），再量一轮即收敛。

**右下角把手**（默认显示）：拖动改宽度、改高度；拖过之后就不再被内容高度牵着走（手动尺寸优先），
双击回到「满宽 + 自适应」；把手 `title` 会写出「当前 W × H px」和双击回到哪种尺寸。
宽度拖到超过笔记栏宽时，外层 `.frm-embed-shell` 自己横向滚动，不会把笔记布局撑破；
把手由 JS 按 iframe 实际尺寸摆放——挂在容器的 `right/bottom` 上会飘到栏边。

**换笔记栏宽 / 拖窗口时图要跟着变**（四条兜底，专门让「旧组件页 + 新宿主」也不卡在旧尺寸）：

1. 盯 iframe 宽度，一变就把 `resize` 事件补送进 iframe，再重测高度。组件**只**在自己收到 `resize` /
   自己的 `ResizeObserver` 被投递时才按新宽度重排；那个帧的生命周期没跑时它会停在旧宽度（表现为
   「窗口变宽了图不变宽」甚至内部冒滚动条）。主文档的生命周期总是活的，所以这条可靠。
2. 往组件页注入一段兜底样式，强制 `.wg-figure` 不被 `max-height:72vh` 截断 —— 旧版组件页
   （没有 `html[data-wg-embed]` 那条规则）也一样整张图撑开、不内部滚动。
3. 组件内交互（点击 / 拖动 / 按键 / 改输入）之后补测一次高度，覆盖旧组件不 `postMessage` 的情况。
4. 组件若按「带瞬时滚动条的窄宽度」画过，发现它用的宽度 ≠ iframe 宽度就再补一次 `resize`（每次改宽最多 3 次）。

**怎么确认自己看的是最新构建**：组件页脚印着 `widget/v1 · build <日期>`。改了 `widgets.js` / `widgets.css`
后必须重建所有已登记组件（`make_widget.py new … --force`），否则 `--check` 会报「派生 HTML 与源 spec 不同步」。

## 为什么是「目录 + view.js + view.css」

这是 **Dataview 上游文档**规定的写法（*Code Reference → `dv.view(path, input)`*）：
传目录时它会载入 `view.js`，并**自动注入同目录的 `view.css`**。
自动注入 CSS 这点很关键——Obsidian 的清洗器会删掉笔记里手写的 `<style>`，但 JS 创建的不会被删。

## 它做了什么

1. 用 `app.vault.adapter.read()` 读出组件 HTML（组件是自包含单文件，见 `Maps/_widgets/`）；
2. 设成 `<iframe srcdoc="…">`，按上面的规则决定高度；
3. 给 iframe 的 document 挂一个点击捕获，把组件里那条 `obsidian://` 链接转成 `workspace.openLinkText()`
   （否则会被 Obsidian 的导航闸拦下，因为它只放行 http(s)）。源笔记不在了就什么都不做，不误建空笔记。

## 为什么不用 `<iframe src="…">`（四条都查证过）

| 事实 | 出处 |
|---|---|
| 官方嵌入语法是 `<iframe src="URL">`，需要一个 URL | 官方帮助 *Embed web pages* |
| 本地文件没有可写死的 URL：`file://` 被拦；桌面端资源前缀是 `app://random-id/`（每次启动随机） | 官方 API `Platform.resourcePathPrefix`；应用内 `se = oe + qe(36) + "/"` |
| 笔记里手写 `<style>` / `<script>` 会被清洗删掉（`iframe` 本身是放行的） | 应用内清洗器配置 `FORBID_TAGS:["style"]`、`ADD_TAGS:["iframe"]` |
| `<iframe srcdoc>` 直接写在笔记里不行：`srcdoc` 不在白名单 | `obsidian.asar` 属性白名单里 `srcdoc` 命中 0 次 |

所以必须由**脚本在运行时**产出内容；`srcdoc` 由脚本设置就绕过了清洗器。

## 已知限制

Dataview 在**索引版本变化**时（新增/修改文件）会重渲染 `dataviewjs` 块，那一刻组件会被重建、
滑块回到初值、手动拖过的窗口尺寸也回默认。这是 Dataview 的既定行为（按 `index.revision` 重渲染），
本视图控制不了。想在拖动过程中完全不受影响，就单独在浏览器里打开 `Maps/_widgets/<组件>.html`
（自包含单文件，双击即看）。

## 测试

```bash
node Maps/_tools/test_widget_embed_view.js
```

按 Dataview 的真实调用方式（`new Function("dv", "input", 源码)`）执行本目录的 `view.js`，覆盖：
正常内嵌、只写文件名、五类非法输入、六个示例组件、`obsidian://` 链接接线、
**笔记里的 `dv.view` 调用是否都指向存在的组件文件**，以及尺寸相关的一整套断言：
自适应量高、`minHeight`/`maxHeight` 边界、读不到内部文档时退回固定高度、
把手拖宽拖高 + 双击回自适应 + `resizable:false`、
`postMessage` 报高度（认来源 / 只在自适应模式生效 / 可被手动尺寸压住）、
**注入的兜底样式（压掉 `72vh` 截断）+ 宽度变化补送 `resize`（且补送次数有界）**、
**精简模式默认开启 + `chrome:"full"` 可回退**、
**注册表里每个组件页内联的 `widgets.js` 是否都是当前 `BUILD`**。
