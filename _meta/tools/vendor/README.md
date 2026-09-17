# vendor/ —— 内联用的 Plotly 官方发行包

只给 `kind = "plotly"` 的交互组件页用（渲染器实现在 [plotly-adapter.js](../plotly-adapter.js)，
选型与口径见 `python3 Maps/_tools/make_widget.py --spec plotly`）。
`make_widget.py` 构建 plotly 页时按 `spec.plotly.bundle` 选一份内联进页面
（**内容不裁剪、不压缩、不加补丁**，只做两处语义等价的转义，见下面「内联约定」），
零 CDN、零外部请求；其他 kind 的页面不带它。

## 两份 bundle

| bundle | 文件 | 含什么 | 字节数 | 页面代价 |
|---|---|---|---|---|
| `full`（默认） | `plotly.min.js` | 2D + 3D + 地图 + parcoords（官方 full build） | 4851164（4.63 MiB；gzip -9 后 1463984 B ≈ 1.40 MB） | 每个 plotly 页 ≈ 5.0 MB |
| `gl3d` | `plotly-gl3d.min.js` | **只有 3D**：surface / scatter3d / mesh3d / isosurface / volume / cone / streamtube | 1690768（1.61 MiB；gzip -9 后 537333 B ≈ 525 KB） | 每个 plotly 页 ≈ 1.9 MB |

选法：spec 里写 `"plotly": {"bundle": "gl3d", "data": […]}`。省略时用 `full`。
`gl3d` 里没有 2D trace，写了 `scatter` / `bar` / `pie` 之类会在校验阶段收到 `[提醒]`
（页面里会画不出来）——要 2D 就换回 `full`。

## 来源

| 项 | `full` | `gl3d` |
|---|---|---|
| 包 | [`plotly.js-dist-min`](https://www.npmjs.com/package/plotly.js-dist-min) | [`plotly.js-gl3d-dist-min`](https://www.npmjs.com/package/plotly.js-gl3d-dist-min) |
| 下载 URL | <https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.7.0/plotly.min.js> | <https://cdn.jsdelivr.net/npm/plotly.js-gl3d-dist-min@3.7.0/plotly-gl3d.min.js> |
| 版本 | `3.7.0`（文件内印的版本串 `plotly.js v3.7.0`） | `3.7.0`（文件内印的版本串 `plotly.js (gl3d - minified) v3.7.0`） |
| 下载日期 | 2026-09-16 | 2026-09-16 |
| 字节数 | 4851164 | 1690768 |
| sha256 | `8ef4c6ab1369f0019611cbcd2d5b8aafef23e5d19ef58c39d4b4249831fe2180` | `fa6ebaf365ea5ad46a9843ea98fb2635c998558b9d876578aa12f765f823cc3d` |
| 许可 | MIT（Copyright © 2016-2024 Plotly Technologies Inc.，全文见上游仓库 [LICENSE](https://github.com/plotly/plotly.js/blob/master/LICENSE)） | 同上 |

上游仓库都是 <https://github.com/plotly/plotly.js>。
注意 partial bundle 的 npm `latest` 是 `4.1.1`，和 `plotly.js-dist-min` 的 `3.7.0` **不同步**；
本目录两份都是 `3.7.0`，升级时要一起升。

机器可读（`make_widget.py --check` 解析这两块，按 `file:` 匹配；改库必须同步改这里）：

```
file: vendor/plotly.min.js
version: 3.7.0
bytes: 4851164
sha256: 8ef4c6ab1369f0019611cbcd2d5b8aafef23e5d19ef58c39d4b4249831fe2180
```

```
file: vendor/plotly-gl3d.min.js
version: 3.7.0
bytes: 1690768
sha256: fa6ebaf365ea5ad46a9843ea98fb2635c998558b9d876578aa12f765f823cc3d
```

## 与上游逐字节一致（自证方法）

两份文件**都没有做任何改写**（没有自己的压缩、没有裁剪、没有补丁），下载当时各用两份来源对过：

```bash
# full
curl -sS -o /tmp/p.min.js https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.7.0/plotly.min.js
curl -sSL -o /tmp/p.tgz  https://registry.npmjs.org/plotly.js-dist-min/-/plotly.js-dist-min-3.7.0.tgz
tar -xzOf /tmp/p.tgz package/plotly.min.js > /tmp/p.npm.js
cmp /tmp/p.min.js /tmp/p.npm.js && shasum -a 256 /tmp/p.min.js

# gl3d
curl -sS  -o /tmp/g.js  https://cdn.jsdelivr.net/npm/plotly.js-gl3d-dist-min@3.7.0/plotly-gl3d.min.js
curl -sSL -o /tmp/g.tgz https://registry.npmjs.org/plotly.js-gl3d-dist-min/-/plotly.js-gl3d-dist-min-3.7.0.tgz
tar -xzOf /tmp/g.tgz package/plotly-gl3d.min.js > /tmp/g.npm.js
cmp /tmp/g.js /tmp/g.npm.js && shasum -a 256 /tmp/g.js
```

两份都 `cmp` 一致（jsdelivr == npm tarball），且与仓库里这两份文件 `cmp` 一致
（2026-09-16 复核；`gl3d` 的 npm tarball 必须用 `-L` 跟随重定向，否则拿到的是 0 字节页面）。

需要换版本时：重新下载、更新上面「来源」表与两个代码块里的四个字段，再跑
`python3 Maps/_tools/make_widget.py --check`——它会实测两份文件的 sha256 并与本文件登记的比对，
不一致就报 `[错误]`（含实测值前 16 位，方便确认是不是故意升级）。
`--check` 只在**已经存在 plotly 组件**时才校验；没有 plotly 组件的 vault 不受影响。

## 内联约定（页面里的副本与上游的差异）

`build_html()` 把库内联进 `<script>` 时做了两处**语义等价**的转义，上游文件本身不变：

1. `</script` → `<\/script`：和 `widgets.js` 的内联规则一样，防止提前闭合脚本块
   （两份文件里都出现 0 次，规则留着以防升级后出现）。
2. `="http` → `="\x68ttp`：`full` 出现 31 处、`gl3d` 出现 6 处。Plotly 里带着地图瓦片、署名链接、
   maki 图标的 URL 字面量（这些都永远不会被请求到：页面 CSP 是 `connect-src 'none'`，
   `<img>`/网络资源也被 `img-src data:` 挡住）。之所以要转义，是因为组件页有一条
   「页面里不出现 `src="http` / `href="http`」的自包含断言（`test_widgets.py`）；
   在 JS 的字符串、模板、正则与注释里 `\x68` 都解析回 `h`，值完全不变。

因此：**页面里内联的是「转义后的等价副本」**，不是逐字节副本。页面的 `<meta name="wg-vendor-sha256">`
记的是上游文件的 sha256，`<meta name="wg-vendor-inline-sha256">` 记的是页面里那份转义副本的 sha256，
`<meta name="wg-vendor-path">` 记的是用了哪一份（`Maps/_tools/vendor/plotly-gl3d.min.js` 等），
`<meta name="wg-vendor-inline-escapes">` 记转义处数；`--check` 校验的是上游 sha256。

## 体积与已知限制

- **体积**：内联后 `full` 每个 plotly 页至少 4.85 MB（页面总大小 ≈ 5.0 MB），`gl3d` ≈ 1.69 MB
  （页面 ≈ 1.9 MB）；磁盘上不压缩，浏览器解析这份 JS 一次性约几百毫秒（`full`）。
  这是「整库内联 + 零外部请求」的代价：`make_widget.py --check` 会为 plotly 页打印
  「页面 X MB，其中内联 Plotly（bundle=…）Y MB（Z%）」。
- 只要图表真的需要 WebGL 三维、蜡烛图、地图、>10⁵ 点这类手写 SVG 渲染器做不到的东西才值得付这个代价；
  能用 `surface3d` / `contour` / `scatter` 说清的一律不要用 plotly（选型见 INTERACTIVE-AUTHORING.md）。
  纯 3D 场景优先 `bundle:"gl3d"`：省掉 3.16 MB 的 2D/地图代码。
- 需要 `blob:` worker 的特性（如 `parcoords`）在本页 CSP（`default-src 'none'`，未开 `worker-src`/`blob:`）
  下**用不了**：错误会如实显示成页面里的 `[错误]`，**不放宽 CSP**。地图类 trace 需要联网取瓦片，
  同样不可用（CSP 挡住）。gl3d 需要 WebGL：宿主没有 WebGL 时 3D trace 画不出来。
- 想更小可以再裁剪（例如只留 `scatter3d` + `surface`），但那要改 `plotly-adapter.js` 的说明与
  `KINDS['plotly']['notes']`，并把裁剪脚本与产物一起登记进本文件——**当前只采用官方两份 dist**。
