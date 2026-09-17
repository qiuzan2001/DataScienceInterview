# 交互组件（Knowledge Widgets）使用说明

这个库里有一批**可交互的知识图**：拖一个滑块，立刻看到某个机制或数值后果。它们不是为了好看——每个都对应某篇笔记里「已经讲清但很难凭文字记住」的那条结论（通常是「四、算一遍」的条件检验）。
**质量门禁不是“有 slider 就合格”。** 新 spec 必须填写 `teaching.question`、`sourceSection`、`controlEffect`、`visualEvidence`；每个 semantic slider/number/toggle/select 都要进入主绘制数据并改变可比较的主图证据。只改变 marker、readout、标题或说明的控件会被拒绝；`azimuth` / `elevation` / `zoom` 仅作为三维相机视角例外。

## 东西在哪

```
_widgets/                     ← 全部产物
  看板.html                    ← 想一次看完全部图：双击它（浏览器里打开，卡片 + 就地预览）
  Index.md                     ← 纯文本清单（表格 + 链接）
  <笔记名>-<slug>.html         ← 单张图（自包含、离线可用、双击即开）
  <笔记名>-<slug>.json         ← 源 spec（要改就改它，然后用下面的命令重建）
_meta/tools/                  ← 工具链（生成器、运行时、渲染器、内嵌视图）
widgets.config.json           ← 声明"工具链在 _meta/tools、产物在 _widgets"
```

## 怎么用

1. **在笔记里**：相关段落下面有一个 `> [!example] 交互图 · …` 提示块 + 一张内嵌的图（由 Dataview 渲染）。
   - 图没显示时：那一段里有「在浏览器里打开」的兜底链接；或者直接打开 `_widgets/看板.html`。
   - **前提**：Obsidian 设置 → Dataview → 打开 **Enable JavaScript Queries**（`dataviewjs` 默认是关的）。
2. **看板**：`_widgets/看板.html` —— 一次性看全部图，支持搜索、排序、隐藏预览。

## 怎么改 / 怎么加

改图：编辑 `_widgets/<笔记名>-<slug>.json`（那是唯一可编辑的源），然后重建：

```bash
cd "/Users/qiuzan/Desktop/StateFarm Interview"

# 1) 先看有哪些渲染器、某渲染器的骨架长什么样
python3 _meta/tools/make_widget.py --root . --list
python3 _meta/tools/make_widget.py --root . --spec plot        # 骨架含四项必填 teaching 元数据

# 2) 改完 spec 后重建那一张（--force 覆盖派生产物）
python3 _meta/tools/make_widget.py --root . new \
  --note "<该图对应的笔记>" --spec "_widgets/<名字>.json" --slug "<slug>" --force

# 3) 一致性 + 语义两层体检（两个都要过）
python3 _meta/tools/make_widget.py --root . --check        # 注册表 ↔ 磁盘 ↔ 源笔记
node _meta/tools/verify_widget_pages.js                     # 主图指纹：默认/边界值实际重绘比较
node _meta/tools/verify_widget_pages.js --strict            # Plotly 也必须用真实浏览器验收
```

> `.html` 与 `Index.md`、`看板.html` 都是**派生产物**，不要手改（改了下次重建就没了；`--check` 也会报"不同步"）。

## 数字从哪来（重要）

每张图的数字只有两个来源，都在 spec 的 `notes` 里写明：

1. **笔记原文数字**——例如 6.2 的 `bias²=0.0622`、`‖w‖²=0.1557`，或 6.3 的 `TP=30/FP=50`、`Precision 37.50%`。
2. **按笔记口径复算**——用同一套设定（同一随机种子、同一函数、同样本量）重算，并在 `notes` 记录复算命令。凡是笔记没算、由我插值或构造的部分，都明确标成「本节构造」。

如果某张图的数字和笔记对不上，先怀疑图（改 spec 重跑），不要改笔记。

## 这套东西的边界

- 组件是**派生层**：原文与笔记永远是唯一事实来源，图不产生新结论、也不改笔记的正文。
- 一个组件只回答一个问题；控件不超过 3 个；同一批数据不做两张图。
- 图是单文件、零外部依赖、离线可开；`theme:"system"` 才会跟随系统深色（默认白底）。
- 数据量大时渲染器会自己换 canvas 后端（散点 >1000 点 / 热力图 >3600 格 / 三维 >1000 面），代价是没有逐点悬停。
