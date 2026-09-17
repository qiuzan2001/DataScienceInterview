# 交互式知识组件作者规范（v1）

> 这是**派生层**的作者规范：交互组件只让读者**动手验证**已经讲过的话。
> 它不改变知识地图、`Books/` 原文和学习条目深度讲解的任何事实，也不替代推导、数字检验和自测。
> 组件打不开、被删掉、在手机上崩掉时，笔记必须仍然能独立读懂。
> **质量门禁**：有 slider/number/toggle/select 不等于合格。每个 semantic 控件必须进入主绘制数据并改变主图证据；只改变 marker、readout、标题或说明文字的控件会被静态与动态校验拒绝。相机控件（`azimuth` / `elevation` / `zoom`）是三维视角例外，不算教学控件。

---

## 1. 产物与边界

三个文件，角色不能混：

| 路径 | 角色 | 怎么处理 |
|---|---|---|
| `Maps/_widgets/<源笔记文件基名>-<slug>.json` | **源**：人或 agent 可编辑的 spec | 只改这一个 |
| `Maps/_widgets/<源笔记文件基名>-<slug>.html` | **派生产物** | 绝不手改，由 `make_widget.py` 重建 |
| `Maps/_widgets/Index.md` | 脚本生成的组件清单 | 绝不手改，用 `--index` 重建 |

- `<widgetsDir>/看板.html`（默认 `Maps/_widgets/看板.html`）是**可选**的派生索引：卡片 + 就地预览
  （同目录相对路径 `iframe loading="lazy"`），零外部依赖，浏览器直接打开，不依赖 Obsidian 与 PI-Desktop；
  同样是派生产物、**绝不手改**，用 `--board`（只重建看板）或 `--index`（Index.md + 看板）重建。
- 注册表 `Maps/_tools/widgets-index.json` 是「笔记 ↔ 组件」的**唯一映射源**；不要靠文件名猜关系。
- 组件不修改 `FRM-知识地图.md`、不改 `Books/` 原文，也不把交互结果反写成地图事实。常规情况下只新建 `.json` / `.html` / 清单；**只有源笔记已经讲清同一结论时**，才可在相应段落后新增一个 callout + 普通 Markdown 链接，作为入口。它既不是新来源，也不是新结论，只是**验证已有结论**的工具。
- 新建前先列出 `Maps/_widgets/` 和目标笔记所在目录，检查同名冲突；已有同名文件时补充它或换 slug，**绝不覆盖**。
- 临时 spec、试验用 HTML 只写会话的 `$PI_SCRATCH_DIR`，不落进 Vault。

## 2. 六条硬约束

1. **不能把交互塞进 Markdown。** Obsidian 阅读视图会剥掉 `<script>` 和 `<iframe>`，`<style>` 也不可靠。所以正文只放**一个普通 Markdown 相对链接**指向 `.html`；绝不在笔记里写 JS、iframe 或内联 style 交互。**动机**：写进笔记的脚本不会执行，读者只会看到一段意义不明的代码，而且它会被 Git 当成笔记内容长期维护。
2. **笔记必须仍然能独立读懂（渐进增强）。** 凡是组件里出现的结论，笔记正文必须已经讲清；组件不得引入原文或笔记里没有的数字与结论。**动机**：组件是"让你动手验证"，不是"替你讲"——去掉组件后笔记必须依然完整。
3. **数值必须来自源笔记或 `Books/` 原文并注明出处。** 不能随手编一个参数（例如凭空给一个 σ）。**动机**：组件里的数字会被读者当成事实记住，来源不实的数字比不给组件更糟。
4. **一个组件只回答一个问题；控件不超过 4 个；标题写"能看出什么"，不写"某图"。** **动机**：控件一多，读者变成在玩面板，而不是在验证一个反直觉的结论。
5. **不确定的一律如实标注。** 交互可行性、浏览器差异、内嵌方式、未经核对的凸性口径，全部标"未核实"。**动机**：本 Vault 的规则是如实标未核实，而不是看起来完整。
6. **先问"Obsidian 原生能不能做"。** 能用 `> [!question]-` 折叠、表格、mermaid、Bases/Dataview 说清的，**就不要生成 HTML**。**动机**：HTML 是唯一需要在 Obsidian 之外打开的产物，能用原生手段表达的，多一个文件就是多一份维护负担。

## 3. 先问：Obsidian 原生能不能做

按"最弱的手段先试"排序：

| 你想要的效果 | 先用什么 | 什么时候才轮到 HTML 组件 |
|---|---|---|
| 收起/展开一段提示或答案 | `> [!question]- 问题` 折叠 callout | 从不 |
| 固定其他维度、只比关键那一维 | 普通 Markdown 表格 | 从不 |
| 流程、因果、状态转移 | Mermaid 流程图 | 从不 |
| 从已有笔记里筛选、排序、汇总 | Bases（`.base`）或 Dataview 查询 | 从不 |
| 公式排版与逐符号解释 | LaTeX `$…$` + 紧贴的表格 | 从不 |
| 不需要拖参数的科学曲线、散点、分布或矩阵 | matplotlib 静态 PNG/SVG；可选 seaborn | 只有确需改变参数时才用 HTML |
| 读者**拖动参数**、自己看出一个反直觉结论 | 只有这一件事原生做不到 | 这时才建交互组件 |

判据一句话：**读者需要"改变一个量并立刻看到结果"吗？** 需要才建组件；只是想看到结论，写进笔记即可。

## 4. 渲染器选型

用 `python3 Maps/_tools/make_widget.py --list` 查看当前渲染器与选型建议。**20 个 `kind`**，按「基础 8 个 → 跨学科补充 6 个 → 机器学习/统计/三维 5 个 → 第三方库 1 个」分四组：

| `kind` | 适合的知识点 | 例子 |
|---|---|---|
| `plot` | 参数滑块驱动的曲线族：结论随参数连续变化 | 价格-利率曲线比较久期线性近似与凸性二阶近似；BSM 期权价 vs 标的价与 σ；CAPM/SML vs β |
| `bars` | 参数改变条形高度或构成 | EWMA/GARCH 权重随 λ/α 变化；现金流现值分解 |
| `scatter` | 数据点与拟合视野 | 回归残差、两资产收益散点 |
| `histogram` | 抽样与分位线 | 蒙特卡洛 VaR/ES、抽样分布与置信区间 |
| `heatmap` | 可编辑矩阵 | 相关系数矩阵 → 组合方差；VIF；希腊字母敏感度表 |
| `timeline` | 事件时间轴 | 现金流与折现；MBS 提前偿还；利率期限结构 |
| `tree` | 可展开的分层结构 | 二项树/决策树；情景分析 |
| `custom` | 逃生口：agent 自带 `html` + `js` | 只在既有渲染器都不合适时用，并在 `notes` 里写明为什么必须自定义 |

后六个是为了**不只服务金融**而加的（数学、统计、机器学习、理科通用）：

| `kind` | 适合的知识点 | 例子 |
|---|---|---|
| `box` | 一组样本的分布概括：中位数、四分位距、异常点（Tukey 口径） | 实验/观测数据的组间对比与离群点筛查；残差、测量值、重复实验的离散度 |
| `ecdf` | 不假设分布形状的累积分布 | 中位数与尾部概率；样本量如何影响经验分布的抖动；两组的 ECDF 对照 |
| `qq` | 样本分位数 vs 理论分位数，看分布形状对不对得上 | 残差/收益率是否近似正态；厚尾与偏斜在两端最明显；离群点会甩出直线 |
| `contour` | 两个自变量的标量场：f(x,y) 的等值线随参数移动、变形 | 损失函数的等高线；效用无差异曲线；概率密度的水平集；组合方差对权重与相关系数 |
| `vector` | 二维向量场：网格上的方向与相对大小 | 动力系统相图；梯度/最速上升方向；经济学的方向场 |
| `matrix` | 2×2 线性变换的向量几何 | 行列式=面积缩放倍数；特征方向与特征值（含复数）；协方差/相关矩阵的几何直观 |

后五个是面向机器学习 / 统计 / 三维的（判据不变：读者得**改一个量并立刻看到结果**）：

| `kind` | 适合的知识点 | 例子 |
|---|---|---|
| `regression` | 散点 + 拟合直线：改斜率/截距或换一批样本，看残差竖线与 R² 怎么变 | 最小二乘回归（CAPM 市场模型 β、久期回归、因子暴露）；手动线对照 OLS；残差结构与 R²/RMSE 口径 |
| `pca` | 主成分方向：转一条候选轴，看它上面的方差什么时候最大（正好等于 PC1） | PCA / 特征分解的几何含义；正交化与降维保留多少方差；相关矩阵的主轴方向 |
| `descent` | 梯度下降的轨迹：在损失曲面的等高线上跑迭代，改学习率 α 看它走稳、来回震荡还是发散 | 梯度下降的步长选择；凸与非凸函数的下降路径；点图框换起点 |
| `surface3d` | 三维曲面 / 点云 / 轨迹：拖着看 z = f(x, y) 的形状（方位角与仰角都是控件） | 损失曲面、效用面、联合分布密度；三维散点云；带一条三维轨迹的寻优路径 |
| `treefit` | 决策树的轴对齐切分：把 depth 拖深，看叶子越切越细、训练误差一路降 | 回归树 / 分类树的划分与叶值；过拟合的来源；切分点怎么选；bagging/boosting 的基学习器 |

最后一个是唯一**带第三方库**的（opt-in，代价写在 spec 里）：

| `kind` | 适合的知识点 | 例子 |
|---|---|---|
| `plotly` | 手写 SVG 渲染器真的做不到的图：真 WebGL 三维（可平滑旋转 / 光照 / 深度）、K 线、地图、10⁵ 以上的点 | 要转视角看的三维损失曲面；蜡烛图 + 成交量；带光照的 mesh3d；十万级散点 |

`surface3d` 适合能用正交投影读懂的中小型曲面、点云或轨迹；`azimuth` / `elevation` / `zoom` 只改变相机视角，属于相机例外，不能拿来充当教学数据控件。若读者需要真实 WebGL 深度、平滑旋转/光照，或数据量已超出内置预算，再选 `plotly`；Plotly 页面必须走真实浏览器验收，不能用假 DOM 自检冒充通过。即使组件有 slider，也只有 slider 真正改变主图数据并留下可见证据才合格。
选它之前先问三遍：`surface3d` / `contour` / `vector` / `scatter` / `heatmap` 能不能说清？能就别用。这条路的代价是**页面体积**——`plotly.bundle` 决定内联哪一份：

| bundle | 文件（`Maps/_tools/vendor/`） | 含什么 | 字节数 | 页面 |
|---|---|---|---|---|
| `full`（默认） | `plotly.min.js` | 2D + 3D + 地图 + parcoords | 4851164 | ≈ 5 MB |
| `gl3d` | `plotly-gl3d.min.js` | **只含 3D** | 1690768 | ≈ 1.9 MB |

纯三维场景一律用 `gl3d`（省 3.16 MB）；`gl3d` 里写 2D trace（`scatter`/`bar`/`pie`…）构建期会给 `[提醒]`，页面里画不出来。两份库都是官方 npm 包的**原样副本**（无裁剪、无补丁），来源 / 版本 / sha256 / MIT 许可 / 内联时的两处语义等价转义都记在 `Maps/_tools/vendor/README.md`，`--check` 会实测 sha256 并与登记值比对、打印每页体积分摊。库是**本地内联**：依然零 CDN、零网络请求，CSP 不放宽（因此需要 `blob:` worker 的 `parcoords`、要联网取瓦片的地图 trace 不可用；没有 WebGL 的宿主里 3D trace 画不出来，会如实报 `[错误]`）。

`custom` 仍然是**单文件内联、无 CDN**：不允许外链脚本、字体或 CSS。

## 5. spec v1 字段

顶层字段：

| 字段 | 必填 | 取值 | 含义 |
|---|---|---|---|
| `schema` | 是 | 固定 `"widget/v1"` | 版本门槛；写错就是非法 spec |
| `kind` | 是 | 见第 4 节 20 选一 | 决定用哪个渲染器 |
| `title` | 是 | 文本 | 写"能看出什么" |
| `teaching` | 是 | `{question,sourceSection,controlEffect,visualEvidence}` | 质量门禁所需的教学元数据：问题、源笔记定位、控件如何改变主图、读者应观察到的图形证据 |
| `subtitle` | 否 | 文本 | 一行写口径与数据出处 |
| `vars` | 否 | 对象（值可含数组） | 常量：`P0`、`D`、`C`、现金流数组等；键必须是合法表达式变量名 |
| `x` | 依 kind 而定 | `{min,max,points,label,fmt}` | 横轴定义域/取样点数：`contour`/`vector`/`descent`/`surface3d` 用它定义 x 网格（`min`/`max` 必填、`points` 可选 3–121）；`plot`/`scatter` 的 `min`/`max`/`points` 必填；`regression`/`pca`/`treefit` 里只用来定范围与刻度格式 |
| `y` | 见下 | `{min,max,points,label,fmt}` | 纵轴定义域与网格点数（与 `x` 同构）：`contour`/`vector`/`descent`/`surface3d` 必填（`min`/`max` 必填、`points` 可选 3–121）；`regression`/`pca`/`treefit` 可选，只用来定范围与刻度格式 |
| `controls` | 否 | 数组，**≤4 个** | 读者能拖动的旋钮 |
| `series` | plot/scatter 时按需 | 数组 | 曲线族或散点组，每条用表达式或 `points` |
| `bars` | bars 时必填 | 数组 | 条形定义 |
| `histogram` | histogram 时必填 | 对象 | 抽样表达式/离散频数与分箱数 |
| `heat` | heatmap 时必填 | 对象 | `rows`/`cols`/`values`；可选 `editable`（格子可直接改）、`symmetric`（改一格同步镜像格）、`bind`、`fmt` |
| `timeline` | timeline 时必填 | 对象 | `items` 事件及时间 |
| `tree` | tree 时必填 | 对象 | 分层节点 |
| `custom` | custom 时必填 | `{html, js}` | 自带标记与脚本；不得联网或加载外部资源 |
| `plotly` | plotly 时必填 | 对象 | `data`（非空数组，元素是 Plotly trace 对象）；可选 `layout`、`config`、`height`（120–2000 px，省略时 `clamp(图区宽×0.72, 260, 520)`）、`bundle`（`full`/`gl3d`）。`data`/`layout` 里以 `=` 开头的字符串按表达式求值一次；要"按控件生成一列数"用生成器 `{"by": 表达式, "n": …}` 或 `{"by": 表达式, "rows": …, "cols": …}`（可带 `"vars": {"名字": 表达式}`）——见 §6。**只有 plotly 页带库**，其他 kind 零负担 |
| `box` | box 时必填 | 对象 | `sample` 表达式或 `values` 数组；可选 `fmt`/`color`/`medianColor` |
| `ecdf` | ecdf 时必填 | 对象 | `sample` 或 `values`；可选 `maxPoints`（2–5000，默认 1200）、`fmt`、`color` |
| `qq` | qq 时必填 | 对象 | `sample` 或 `values`；可选 `dist`、`maxPoints`（默认 2000）、`fmt`、`color` |
| `contour` | contour / descent 时必填 | 对象 | `expr`（f(x,y)；`descent` 里它就是损失函数）；可选整数 `levels`（层数，等分在 zmin–zmax 内部；`contour` 默认 6、`descent` 默认 8，范围都是 1–20） |
| `vector` | vector 时必填 | 对象 | `u`、`v` 两个分量表达式；可选 `scale`（默认 `auto`：按全场最大模长归一化） |
| `matrix` | matrix 时必填 | 对象 | `values` 为 2×2 数字矩阵；可选 `editable`、`samples`（样本点）、`bind` |
| `points` | regression/pca/treefit 必填其一 | 数组 | `[[x, y], …]`；与 `data` 二选一；`treefit` 的 `mode="2d"` 每点写 `[x, y, 类别]` |
| `data` | regression/pca/treefit 必填其一 | 对象 | `{n, x, y[, cls][, seed]}`：逐点求表达式的点生成器（`x`/`y` 是表达式字符串，`n` 是 1–2000 的整数、默认 30，`seed` 默认 12345；`treefit` 的 `mode="2d"` 还要给 `cls` 表达式）；与 `points` 二选一，见 §6 |
| `fit` | regression 否 | 对象 | `mode` ∈ `ols`（默认）/ `manual` / `compare` / `none`；`slopeKey`/`interceptKey`（控件 key，默认 `slope`/`intercept`）；数字常量 `slope`/`intercept`；`residuals`（`false` = 不画残差竖线）。`manual`/`compare` 必须能找到对应控件或用 `slope`/`intercept` 常量，否则报 `[错误]` 并退回 `ols` |
| `meanCenter` | pca 否 | `true`（默认）/ `false` | `false` = 不中心化，算的是绕原点的二阶矩（PC1 会被均值方向拉偏） |
| `candidate` | pca 否 | 对象 | `angle`（数字，单位度，默认 0）；`angleControl`（控件 key，默认 `theta`；不是任何控件的 key 时退回 `angle` 并报 `[错误]`）；`ellipse`（默认 `true`，1σ 椭圆）；`showProjection`（默认 `true`，样本点到候选轴的垂足） |
| `surface` | surface3d 时必填 | 对象 | `expr`（z = f(x, y)）或 `points`（`[[x, y, z], …]`，给了 `points` 就忽略 `expr`）；`mode` ∈ `surface`（默认）/ `wireframe` / `points`；可选 `fmt` |
| `path` | 否 | `[[x, y, z], …]` | 三维轨迹（surface3d 用），画在最上层 |
| `view` | 否 | 对象 | surface3d 的视角：`azimuthKey`/`elevationKey`/`zoomKey`（控件 key，默认 `azimuth`/`elevation`/`zoom`）与常量 `azimuth`/`elevation`/`zoom`（默认 35° / 25° / 1，`zoom` 运行时夹在 0.2–4）。在图上按住拖动会覆盖视角 |
| `mode` | 否 | `1d`/`2d`（treefit）、`surface`/`wireframe`/`points`（surface3d） | `treefit.mode` / `surface.mode` 的等价顶层写法；**只写一处**，就写在子对象里（两处都写又不同值时，校验和运行看的不是同一个，见本节末注） |
| `treefit` | treefit 时必填 | 对象 | `mode` ∈ `1d`（默认）/ `2d`；`splits` 非空（1d：数字阈值或 `{"at": 数字或表达式}`；2d：`{"axis": "x"\|"y", "at": 数字或表达式}`，按"加深一层"的顺序）；可选 `depth`（数字，默认满深度）、`depthKey`（控件 key，默认 `depth`）。`mode="2d"` 的数据必须带类别标签（见 `points`/`data`） |
| `grad` | 否 | 对象 | descent 的解析梯度：`dfdx` + `dfdy` 两个偏导表达式，**必须同时给**（只给一个报 `[错误]`）；不给就用中心差分（h = 定义域宽/1000），图注会写明用的哪一种 |
| `descent` | 否 | 对象 | `lrKey`/`stepsKey`（控件 key，默认 `lr`/`steps`；找不到控件又没写 `descent.lr`/`descent.steps` 常量时按默认 0.1 / 20 走）；`clickToSetStart`（默认 `true`：点图框换起点） |
| `start` | 否 | `[x, y]` 两个数字 | descent 的迭代起点；不写就从网格上 f 最大的格点出发 |
| `markers` | 否 | 数组 | 参考线、关键点 |
| `readouts` | 否 | 数组 | 数字读数 |
| `notes` | 否 | **字符串数组** | 出处、未核实声明、为何必须自定义 |
| `theme` | 否 | `"light"`（默认）/ `"system"` | 白底是默认；只有 `system` 才显式跟随系统黑白 |
| `renderer` | 否 | `"auto"`（默认）/ `"svg"` / `"canvas"` | **只对 mark 密集的 `scatter` / `heatmap` / `surface3d` 有意义**（其余 kind 一律 SVG，写了会被忽略并提醒）：`auto` 按密集标记数自选（散点 >1000 个、热力图 >3600 格、面片 >1000 个就换 canvas），`svg` 保留逐点 `<title>` 悬停，`canvas` 强制走画布（省掉"一个标记一个 DOM 节点"，代价是没有逐点悬停）。可编辑热力图一律留 DOM。实测与阈值见 §13 |
| `aspect` | 否 | `"equal"` / `"auto"` | 仅 contour/vector/matrix/pca/descent：`equal` 两轴同一比例（几何保真），`auto` 各自拉伸填满图框。默认值按渲染器区分：contour/vector 是 `auto`；matrix/pca/descent 是 `equal`（det = 面积缩放倍数、特征方向不变、点到主轴的距离、最陡方向都依赖几何保真） |

### 5.1 教学元数据与主图语义门禁

`teaching` 是新 spec 的必填顶层对象，四个字段都要有真实内容：

- `question`：读者要回答的一个问题；`sourceSection`：源笔记的章节/段落定位；
- `controlEffect`：字符串或字符串数组，逐一说明控件如何改变主绘制数据；
- `visualEvidence`：读者在主图中应看到的证据，而不是只描述一个 readout。

`teaching.controls` 可用字符串数组明确列出 semantic 控件；省略时，所有 slider/number/toggle/select
默认都是 semantic。`teaching.cameraControls` 可额外列出相机控件；`azimuth`、`elevation`、`zoom`
本身在三维图中也按相机控件处理。相机旋转/缩放是视角交互例外，不替代教学数据变化。

静态门禁按 renderer 的主绘制字段检查控件 key：例如曲线看 `series[].expr`，条形看
`bars[].value`，分布看 sample/values，场/矩阵看各自的数值表达式，回归/PCA/树看 data、points
或切分/拟合数据，Plotly 看 `plotly.data` 与有效布局。`readouts`、`markers`、`title`、`subtitle`
和 `notes` 永远不算主绘制依赖；控件只出现在这些字段里会失败。

少数运行时重算数据但没有可写 expr 的 renderer 可写 `teaching.dynamicRenderer: true`，但这不是
豁免：`verify_widget_pages.js` 必须用默认/最小/最大值（select/toggle 轮换选项）比较主图指纹。
没有足够的主图几何、canvas 调用或 renderer 绘制状态，或者指纹只因 readout/marker 改变，仍然失败。

各渲染器条目（`series[]`、`bars[]`、`readouts[]` 等）的**内部键以 `make_widget.py --spec <kind>` 打印的骨架为准**；本规范只冻结顶层字段与语义，示例不代替核对骨架。

> **`mode` 只写一处。** 校验器对 `surface3d` 只看 `surface.mode`、对 `treefit` 优先看顶层 `mode`，而运行时两者都优先看 `treefit.mode` / `surface.mode`；两处同写且不同值时会出现"校验通过、画出来却不是校验的那个模式"，所以统一写在 `treefit.mode` / `surface.mode` 里。

`controls[]` 字段：

| 字段 | 必填 | 取值 | 含义 |
|---|---|---|---|
| `key` | 是 | 标识符 | 在表达式里当变量名用；不要与 `vars` 的键、`x`、`i`、`item` 冲突 |
| `type` | 是 | `slider` / `select` / `toggle` / `number` | 控件种类 |
| `label` | 是 | 文本 | 显示名，写清单位和范围（如"Δy（25bp = 0.0025）"） |
| `min`/`max`/`step` | slider、number 必填 | 数值 | 可动范围；`step` 决定最小刻度 |
| `value` | 否 | 数值 | 初值；给成笔记里那个算例的值 |
| `options` | select 必填 | `[[值,显示文本],…]` | 选项列表 |
| `fmt` | 否 | 见下 | 显示格式 |
| `animate` | 否 | `true` 或 `{from, to, seconds, pingpong, autostart}` | 自动演示（见 §7）：`true` = 从 `min` 到 `max`、6 秒一趟、来回。`seconds` 默认 6（需在 0.5–120，运行时超过 60 秒按 60 秒走）、`pingpong` 默认 `true`、`autostart` 默认 `true`；`from`/`to` 省略时取控件的 `min`/`max`，落在 `[min, max]` 之外只警告并夹到端点，`from` 与 `to` 相同报 `[错误]`。**只对 `slider`/`number` 生效**（其他类型只警告忽略），也**不计入 4 个控件上限** |

`fmt` 至少支持 `0.00` / `0,0.00` / `0.00%` / `0.0000` 四种模式。

## 6. 表达式作用域与辅助函数

表达式里只有这些名字可见，不要在 spec 里发明作用域：

- `vars` 的**全部键**；
- 全部 control 的 `key`；
- `x`（plot/scatter 的横轴自变量）、`i`（当前下标）、`item`（当前数组元素）；
- 辅助函数：`rand, randn, phi, quantile, sum, mean, sd, clamp, min, max, abs, exp, log, sqrt, pow, sin, cos, tan, floor, round, PI, E, ifelse, fmt`。

规则：能用 `vars` 表达的常量不要写死在表达式里；同一个数在 `series` 与 `readouts` 里必须来自同一个 `vars` 键，否则读者拖到边界时会看到两条曲线对不上。

`plotly` 的表达式的口径与上面**不一样**，要单独记：`plotly.data` / `plotly.layout` 里以 `=` 开头的字符串在渲染时**整体求值一次**（作用域里没有逐点 `x` / `i` / `item`，所以 `"=a*x"` 这种写法没有"逐点"的意思）。要按控件生成一串数，用**生成器**：

- 一维：`{"by": "<表达式>", "n": 25}` → `[v₀, …, v₂₄]`；作用域有 `i`（0 基下标）、`n`（长度）、`x = i`。
- 二维：`{"by": "<表达式>", "rows": 25, "cols": 25}` → 25×25 嵌套数组；作用域有 `i`（行）、`j`（列）、`n`（行数）、`m`（列数）、`x = i`、`y = j`。`z[i][j]` 的 `i` 走行、`j` 走列，与 Plotly `surface` 的口径一致。
- `n` / `rows` / `cols` 可以是数字，也可以是 `"=表达式"`（求值一次后取整）。
- 生成器还可以带 `"vars": {"名字": "表达式"}`，在**同一元素作用域**里先求出这些名字再算 `by`——用来把"范围常量只写一遍"放进 `spec.vars`，让 `x`/`y`/`z` 几个生成器共用（否则三处各写一份区间，改一处就静默跑偏）。
- 上限：一维 ≤ 300 个元素、二维 ≤ 40000 格（超了报 `[错误]`，不静默截断）。某一位算不出有限数字时**跳过该位**（二维填 `NaN`）并报一条 `[错误]`，不伪造数据。
- 也可以在 `data` 里直接写数组（Plotly 要什么写什么）；生成器只是"跟着控件变"的那条路。

表达式引擎**没有对象字面量、没有 for/while、没有 `Array.from`/`map`**（这是"不用 `new Function`"的直接后果），所以生成器不是语法糖，而是唯一能"按控件生成一列数"的写法。

- `histogram.sample` 模式下，`readouts` 可读取只读数组 `__samples__`，例如 `-quantile(__samples__, 0.05)`；它只在当前一次抽样/重绘中存在，不能当作跨次可复现数据。
- `heat.bind` 可显式把矩阵格子暴露为表达式变量，例如 `"bind": {"rho12": [0, 1]}`；`"symmetric": true` 时编辑 `[0,1]` 会镜像到 `[1,0]`。不要靠行列标题猜变量，绑定必须写在 spec 里。

`data{n, x, y[, cls][, seed]}` 是 `regression` / `pca` / `treefit` 的**逐点生成器**（与 `points` 二选一，见 §5）：

- 逐点求值的作用域里有行号 `i`（0 起）与总点数 `n`（1–2000 的整数，默认 30；`treefit` 的 `mode="2d"` 默认 60），以及 `vars` 与控件值。
- `rand()` / `randn()` 在 `data` 里是**按 `data.seed` 播种**的伪随机数（默认 12345），不是全局 `Math.random()`：同一个 seed + 同一个 n 一定得到同一批点，**拖其他控件不会重抽样本**——否则"看同一批数据"这个前提就没了。（`histogram` / `box` / `ecdf` / `qq` 的 `sample` 用的是未播种的全局 `randn()`，每次重绘重新抽样，那是故意的：让抽样误差自己抖出来。）
- `x` 表达式算完会**写回作用域**，所以 `y` 里可以直接引用同一个 `x`（例如 `x: "i*0.1"`、`y: "0.8*x + 2*randn()"`）。**例外**：`treefit` 在 `mode="2d"` 下走另一条求值路径（同一次要一起算 `cls`），作用域里的 `x` 始终是行号、不会被写回；要在 `y`/`cls` 里用同一个 x，请给 `points` 或改用 `mode="1d"`。
- 求值失败的、或结果不是有限数的点会被跳过，并在 `[错误]` 里给出"几个点失败 + 第一条原因"。

**表达式引擎是自己的 AST，不是 `new Function`。** 字符串走 tokenizer → parser → AST → 编译成闭包树（按「名字表 + 表达式」缓存），作用域里只可能有 `vars` / `controls` / 逐点变量 / 白名单函数，`window`、`document`、`fetch`、`eval`、`Function`、`constructor`、`__proto__` 这些名字到不了求值环境。因此：

- **`^` 与 `**` 都是乘方**（右结合：`2**3**2 = 512`；`-x^2 = -(x^2)`；`2^-1 = 0.5`）。早先"`^` 是位异或会被拒绝"的临时守卫已经撤掉——现在写 `x^2` 就是乘方，和数学一致。`x*x`、`pow(x, 2)` 照样可用。
- 报错精确到**列号**，并且区分三类：语法错（`表达式「1 +」第 4 列：表达式到这里就结束了，少了一个值`）、未知函数（`第 1 列：未知函数「coqs」`）、未知变量（`第 1 列：未知变量「nosuchvar」`）。
- 想在构建期先扫一遍，用运行时导出的静态检查：`WG.analyze(expr)` 给 `{vars, calls}`，`WG.lintSpec(spec, kind)` 给 `{errors, unknownFns, unknownVars}`（按字段名 `expr/sample/u/v/x/y/dfdx/dfdy/at/amount/t`… 逐个表达式解析；`make_widget.py` 侧尚未接线）。
- 除法仍按 JS 语义：`1/0` 是 `Infinity`（不是抛错），`0/0` 是 `NaN`；`ifelse(cond, a, b)` 的两个分支**都会被求值**（和以前一致，别把它当 `a ? b : c` 用）。

`readouts`（以及 `markers`、`series[].expr`）还能读渲染器写回的**只读状态**（`make_widget.py` 的 `INTERNAL_VARS`，`--spec <kind>` 会打印）：

| kind | 读数变量 |
|---|---|
| `histogram` / `box` / `ecdf` / `qq` | `__samples__`：本次抽样的冻结副本（只在当前一次重绘里有效，不能当跨次可复现数据） |
| `regression` | `__reg__`：n, mx, my, slope, intercept, r2, rmse, ssRes, varY；`compare`/`manual` 时另加 mSlope, mIntercept, mR2, mRmse |
| `pca` | `__pca__`：n, mx, my, l1, l2, ratio, angle1, totalVar, candAngle, candVar, candRatio, candShare, meanCentered |
| `descent` | `__gd__`：steps, lr, done, escaped, diverged, x, y, f, f0, gnorm, lost, grad（`analytic` / `numeric`） |
| `surface3d` | `__s3__`：az, el, zoom, zmin, zmax, zRange, cells, pathPoints, bad |
| `plotly` | `__plotly__`：traces, points, height, width, mode（前四个是数字，可直接进 readouts；`mode` 是 `"2d"`/`"3d"`/`"mixed"` 字符串，读数框只格式化数字，要看它请写进图注） |
| `treefit` | `__tree__`：`mode="1d"` 时 depth, maxDepth, leaves, emptyLeaves, mse, rmse, varY, n；`mode="2d"` 时 depth, maxDepth, regions, emptyRegions, err, wrong, n, classes |

`treefit` 的 `__tree__` 运行时确实写进作用域（名字也在保留名清单里），但**目前漏在 `INTERNAL_VARS` 之外**，所以 `--spec treefit` 不会打印这一行；写 readouts 时以上表为准。

## 7. 内嵌精简模式与自动演示

### 7.1 内嵌默认是精简模式（slim）

`dv.view` 内嵌时，宿主（`Maps/_tools/widget-embed/view.js`）会往组件页 `<head>` 注入 `<meta name="wg-chrome" content="slim">`，`widgets.js` 读到它就只生成**标题 / 控件 / 图 / 图例 / 图注 / 读数**这几块：副标题、来源链接（"在 Obsidian 中打开源笔记 ↗"）、`notes` 说明清单、页脚、控件区的"参数"小标题都不生成，内边距也收紧（`widgets.css` 的 `html[data-wg-chrome="slim"]`）。

- **为什么**：那几个块都是"读一遍就够"的解释性文字，内嵌时它们把图挤下去；它们该写在**源笔记**（md）里，跟推导、出处、上下文放在一起。
- **所以作者规则**：解释性的话写进笔记，别指望组件替你讲；组件里只留"改这个量、看这个图"。
- **`notes` 仍然是照写的必填习惯**：单独打开 `.html`、或内嵌时写 `chrome: "full"`，读者看到的是完整版，`notes` 是那里唯一的出处与口径声明。
- 磁盘上的 `.html` 一直不变（注入的 meta 只存在于 `srcdoc` 里），所以**在浏览器里单独打开组件页永远是完整版**，`--check` 看的是同一个文件。
- 要回完整版：`await dv.view("Maps/_tools/widget-embed", { file: "…html", chrome: "full" })`（想在图边上直接看口径时用）。
- 兜底：宿主还会往组件页注入一段样式，强制隐藏 `.wg-sub` / `.wg-src` / `.wg-notes` / `.wg-foot`——**旧组件页**（它的内联 CSS/JS 还不认识 `wg-chrome`）也能被收起来。
- **图注也分两类，改动图注的代码时要按这条分**：讲**方法 / 口径 / 定义 / 公式 / 几何约定**的行（内容不随当前控件值与数据变化）用 `det('…')` 标成"口径行"——渲染器侧直接用同文件里的助手 `det(t, …)`（可传多段或数组，`null`/`undefined` 忽略），它也在 `window.WG.det` 上暴露了一份供测试与宿主使用。slim 内嵌会**跳过**这些行，单独打开组件页照常显示；而**活数字**（`n=10，Q1 …`、`α=0.25、20 步`、`候选轴上的方差 / 总方差 = 0.729…`）、**可操作提示**（拖… / 点… / 悬停… / 格子里的数字可以直接改…）与**当前状态警告**（`第 k 步已越出定义域`、求值失败）必须保持普通字符串，内嵌也要看得见。判据一句话：**读者读一遍就够的句子标 `det`，"这一眼看出来的数"和"我能做什么"不标。**

### 7.2 自动演示（`animate`）

给控件写 `animate`（字段见 §5 的 `controls[]` 子表）就能让数值在读者不动手时自己走：

- 值按**经过的时间**在 `from` → `to` 之间推进（不是按帧数，换个机器速度一致）：默认 `seconds: 6` 走一趟、默认来回（`pingpong`），`pingpong: false` 就是单向循环。`seconds` 校验器要求 0.5–120，运行时超过 60 秒按 60 秒走。
- 由 `setTimeout` 以 **~12fps**（80ms 一帧）驱动，**不是 `requestAnimationFrame`**：动画期间每帧都要重算重画（等高线、三维曲面都不便宜），12fps 足够看清趋势、数字也来得及读；不追 60fps 就不会在慢机器上卡。
- `document.hidden` 时把时间轴往后推：后台不烧 CPU，切回来也不会突然跳一大段。组件被 Dataview 换掉（`root` 不在文档里）后自动收摊，不留常驻定时器。
- 读者偏好**减少动态效果**（`prefers-reduced-motion: reduce`）时**不自动开始**，但按钮仍可手动播放（读者明确要求才动）；同理，Node 里的假 DOM、或宿主设了 `window.__wgNoAuto = true`（例如做静态截图）时也不自动开始。
- **读者一动手就永久停下**：`pointerdown`/`keydown`/`wheel`/`input` 在捕获阶段就把自动演示停掉——这是"自动演示"，不该跟读者抢控件。停下时**保持当前值、不回到初值**；点"▶ 自动演示"按钮本身不算打断。
- 有可播放控件时才出现 `▶ 自动演示` / `⏸ 停下` 按钮（带 `aria-pressed`）。它是 UI，**不占 spec 的 4 个控件名额**。
- 再点按钮从头开始是**从 `from` 重新走一遍**（`autoStart()` 把相位归零、第一步即落在 `from`），不是从当前值接着走。
- 自动开始只由**第一个**可播放控件的 `autostart` 决定（`autostart: false` = 不自动开始，仍可手动播放）。

## 8. 三段可复制示例

以 `Maps/Notes/有效久期.md` 的计算示例为例（拖动 Δy，比较**线性近似**、**加凸性的二阶近似**与同一现金流的真实重估）。

> **本节示例的现状（2026-09-16）**：源笔记 `Maps/Notes/有效久期.md`（uid `N1104.05`）已删除，该组件也从注册表摘除，两份文件移到了 `Archive/`。下面 (a)(b)(c) 仍然照原样保留——它们是**写法模板**：把 note / slug / uid 换成你自己的即可；示例里的数值出处以归档的那两份文件为准。

**(a) `spec.json`**

```json
{
  "schema": "widget/v1",
  "kind": "plot",
  "title": "久期近似与凸性：拖动 Δy 看差多少",
  "subtitle": "同一只债：15 年、7% 票息、面值 10 万、现价 98,550",
  "vars": {
    "P0": 98550, "D": 9.0563, "C": 114.05,
    "y0": 0.071608, "cpn": 7000, "F": 100000, "n": 15
  },
  "x": { "min": -0.02, "max": 0.02, "points": 121,
           "label": "利率变动 Δy（小数）", "fmt": "0.00%" },
  "controls": [
    { "key": "dy", "type": "slider", "label": "利率变动 Δy",
      "min": -0.02, "max": 0.02, "step": 0.0005, "value": -0.0025, "fmt": "0.00%" }
  ],
  "series": [
    { "label": "只到久期（一阶）", "expr": "P0*(1 - D*x)", "color": "#e0a15a", "dash": true },
    { "label": "加凸性（二阶）", "expr": "P0*(1 - D*x + 0.5*C*x*x)", "color": "#7fb8e6" },
    { "label": "真实重估价格", "expr": "cpn*(1-pow(1+y0+x,-n))/(y0+x)+F*pow(1+y0+x,-n)", "color": "#4ec27a" }
  ],
  "markers": [ { "x": "dy", "label": "当前位置", "color": "#e0a15a" } ],
  "readouts": [
    { "label": "一阶近似：价格变化", "expr": "P0*(-D*dy)", "fmt": "0,0.00" },
    { "label": "二阶近似：价格变化", "expr": "P0*(-D*dy+0.5*C*dy*dy)", "fmt": "0,0.00" },
    { "label": "真实重估：价格变化", "expr": "cpn*(1-pow(1+y0+dy,-n))/(y0+dy)+F*pow(1+y0+dy,-n)-P0", "fmt": "0,0.00" }
  ],
  "notes": [
    "P0、D*、y 与 25bp 的一阶近似来自 Maps/Notes/有效久期.md 的 4.4 复算表，非教材原图。",
    "C=114.05 与真实价格曲线是按同一只债的现金流做的数值演算（本节计算，非原文照抄）。"
  ]
}
```

> **数值出处必须自己核一遍。** 4.4 复算表给出 $y\approx7.16\%$、$D^*\approx9.06$、$P_0=98{,}550$，25bp 时一阶近似约 \$2,231。$C=114.05$ 使用年复利口径 $C=\frac1P\frac{d^2P}{dy^2}$：按笔记里的 15 年、7% 票息、面值 10 万、反解 $y=7.1608\%$ 演算得到；25bp 下二阶近似为约 \$2,266，真实重估为约 \$2,267。它是**本组件/本节的数值演算**，不是教材给出的凸性数；换口径、现金流或频率时必须重算，不能照抄。

**(b) 建档命令**

```bash
python3 Maps/_tools/make_widget.py new \
  --note "Maps/Notes/有效久期.md" \
  --spec "$PI_SCRATCH_DIR/有效久期-久期凸性.json" \
  --slug 久期凸性 \
  --uid N1104.05
```

**(c) 笔记正文里放的那几行**（两种，按是否需要内嵌选一种）

只给入口（不内嵌，最省事）：

```markdown
> [!interactive] 久期近似与凸性：拖动 Δy 看差多少
> [打开交互组件](../_widgets/有效久期-久期凸性.html)
```

要**就地内嵌**（推荐；需要 Dataview，见 §10）：

````markdown
> [!interactive] 久期近似与凸性：拖动 Δy 看差多少

```dataviewjs
await dv.view("Maps/_tools/widget-embed", { file: "有效久期-久期凸性.html" })
```
````

- 链接是**普通 Markdown 相对路径**，基准永远是这篇笔记所在文件夹；上例位于 `Maps/Notes/`，所以要写 `../_widgets/…`，不能写 vault 根路径 `Maps/_widgets/…`。
- `dataviewjs` 块里的 `file` 只写文件名即可（自动到 `Maps/_widgets/` 找）；**不要**写完整 URL。默认高度**自适应**（见 §10.2），不需要写 `height`。
- 位置放在它所验证的推导/数值段落之后，不要集中堆到章末；**不**在笔记里手写 `<iframe>`、`<style>` 或 `<script>`（前两个会被清洗器删，第三个也不该进笔记）。

## 9. 命令

```bash
python3 Maps/_tools/make_widget.py --list                       # 组件渲染器目录 + 选型建议 + 已有组件清单
python3 Maps/_tools/make_widget.py --spec plot                  # 骨架含必填 teaching.question/sourceSection/controlEffect/visualEvidence
python3 Maps/_tools/make_widget.py new --note "Maps/Notes/有效久期.md" --spec /path/spec.json [--slug 久期凸性] [--uid N1104.05] [--force]
python3 Maps/_tools/make_widget.py --check                      # 校验注册表↔磁盘↔源笔记一致性，并检测 HTML 是否已过期
node Maps/_tools/verify_widget_pages.js                    # 默认/边界值重绘并比较主图指纹（Plotly 默认 SKIP）
node Maps/_tools/verify_widget_pages.js --strict           # 要求 Plotly 用真实浏览器验收，存在即失败
python3 Maps/_tools/make_widget.py --index                       # 重建两份派生索引：Maps/_widgets/Index.md + Maps/_widgets/看板.html
python3 Maps/_tools/make_widget.py --board                       # 只重建组件看板（<widgetsDir>/看板.html）
python3 Maps/_tools/make_widget.py --layout                     # 打印已解析布局（一行 JSON：root / toolsDir / widgetsDir / registry / indexMd / notesIndex）
python3 Maps/_tools/make_widget.py --root /path/to/project …    # 指定项目根（默认按脚本位置推导）；<root>/widgets.config.json 可改写目录布局
```

在别的项目里运行：用 `--root <项目根>` 指定根（`--vault` 是同一语义的隐藏别名；不传就按本脚本所在位置推导）。项目根下可放可选的 `widgets.config.json`（JSON 对象，键全部可省）：

| 键 | 默认 | 说明 |
|---|---|---|
| `mapsDir` | `Maps` | 内容目录；`toolsDir` / `widgetsDir` 的默认值都挂在它下面 |
| `toolsDir` | `<mapsDir>/_tools` | 注册表（`widgets-index.json`）所在目录 |
| `widgetsDir` | `<mapsDir>/_widgets` | 源 spec / 派生 HTML / `Index.md` 所在目录 |
| `notesIndex` | `<toolsDir>/notes-index.json` | 与知识地图共用同一份 `notes-index.json` |

取值必须是项目根内的相对路径：绝对路径与含 `..` 的路径一律 `[错误]` 退出（解析结果还要真的落在项目根内，symlink 也拦）；配置文件不是对象或不是合法 JSON → `[错误]`；未知键只 `[提醒]`（写 stderr，好让 `--layout` 的 stdout 只有一行 JSON）、不失败。运行时资产（`widgets.js` / `widgets.css` / `vendor/`）始终从脚本自己所在目录读，与项目根无关。`--check` 结尾会多打一行已解析布局；机器读取用 `--layout`（stdout 只有一行 JSON）。
退出码：

| 码 | 含义 | 该怎么办 |
|---|---|---|
| `0` | 成功 | 继续 |
| `1` | 用法错误或冲突：源笔记不存在、spec 不合法、目标文件已存在且未加 `--force`、`uid` 未找到 | 修输入；**不要**靠 `--force` 硬盖已存在的组件 |
| `2` | **已登记成功，但派生清单 `Index.md` 重建失败** | 这不是登记失败，**不要重跑登记**；单独修派生视图重建 |

错误信息带 `[错误]` 前缀并写 stderr；批量问题清单写 stdout。别只看有没有打印就当作通过。

## 10. 内嵌观看方式（Dataview `dv.view`，零进程 / 零端口 / 零自建插件）

**做法：笔记里写 `dataviewjs` 代码块调用 `dv.view()`**，它把组件 HTML 读出来塞进 iframe 的 `srcdoc`。

### 10.1 为什么只能是这个方案（四条都查证过，出处可复核）

| 事实 | 出处 |
|---|---|
| 官方给的嵌入语法是 `<iframe src="URL">`，**需要一个 URL** | 官方帮助 *Embed web pages*（`obsidian-help/en/Editing and formatting/Embed web pages.md`） |
| 笔记里写 HTML 会被清洗：`<script>` 删、`<style>` 删，但 **`iframe` 是明确放行的** | 应用内清洗器配置：`FORBID_TAGS:["style"], ADD_TAGS:["iframe"]`（`obsidian.asar`） |
| **本地文件没有可写死的 URL**：`file://` 被拦（Electron 未开 `--allow-file-access-from-files`，也没开 `disable-web-security`）；桌面端资源前缀是 **`app://random-id/`**，每次启动随机 | 官方 API 类型定义 `Platform.resourcePathPrefix`（"`app://random-id/` on desktop (Replaces the old format of `app://local/`)"）+ 应用内 `se = oe + qe(36) + "/"`，`qe()` 用 `Math.random()` |
| `<iframe srcdoc="…">` **不能直接写在笔记里**：`srcdoc` 不在白名单（整个 `asar` 里 0 次命中），会被剥掉 | `obsidian.asar` 属性白名单 + `srcdoc` 命中数 = 0 |

结论：静态笔记写不出指向本地文件的 URL，所以必须**由脚本在运行时**产出内容——这正是 `dv.view()` 的用途。
（`srcdoc` 由脚本设置就绕过了清洗器，所以能用；这也是本方案的落点。）

### 10.2 笔记里的写法

````markdown
```dataviewjs
await dv.view("Maps/_tools/widget-embed", { file: "交互组件示例-箱线图.html" })
```
````

- `dv.view(path, input)` 是 **Dataview 上游文档**规定的接口（*Code Reference → `dv.view(path, input)`*）：
  载入指定 JS 执行、传入 `dv` 与 `input`，**必须 `await`**；路径相对 vault 根、不能以 `.` 开头。
  写成「目录」形式时（`Maps/_tools/widget-embed/view.js`），Dataview 会**自动注入同目录的 `view.css`**——
  这很关键：笔记里手写 `<style>` 会被删，而 JS 注入的 `<style>` 不会。
- `file` 只写文件名时自动到 `Maps/_widgets/` 找；写完整 URL 或含 `..` 会被拒绝并显示 `[错误]`。
- 依赖：Dataview 已装且 `enableDataviewJs: true`（本机如此）。
- 内嵌默认是**精简模式**（只留标题 / 控件 / 图 / 图例 / 图注 / 读数），要完整版写 `{ chrome: "full" }`；为什么这么定、`notes` 为什么还得写，见 §7.1。

**尺寸（都可以不写）**

| 写法 | 效果 |
|---|---|
| 不写 / `height: "auto"` | **默认**：高度 = 组件内容的实际高度。图不会被裁，也不会有上下滚动条 |
| `minHeight` / `maxHeight` | 自适应模式的上下限（默认 140 / 不限） |
| `height: 420` | 固定高度；内容超出时窗口内滚动 |
| `resizable: false` | 去掉右下角拖拽把手（默认显示） |

**右下角把手**：拖动改宽度、改高度；拖过之后就不再被内容高度牵着走，双击回到「满宽 + 自适应」。
把手的 `title` 会写出「当前 W × H px」以及双击回到哪种尺寸，悬停即可确认。
宽度拖到超过笔记栏宽时，外层容器自己横向滚动，不会把笔记布局撑破。

自适应高度怎么来的（三条，缺一不可）：

1. 宿主视图量 iframe 内部**内容**高度（`body` 盒高 / `body.scrollHeight` / `.wg-wrap` 盒高取最大）。
   **特意不看 `documentElement.scrollHeight`**：它是 `max(内容, 视口高)`，iframe 一旦比内容高，
   这个值就等于视口高，加上 2px 余量后每次重算都会把 iframe 撑高 2px——无界递增。
2. 组件自己 `postMessage({wg:"widget-height"})` 当**触发器**（`widgets.js` 每次重绘/重排后都发）。
   为什么需要它：宿主窗口被遮挡时 Chromium 会推迟渲染生命周期（`rAF`/`ResizeObserver` 都不投递），
   宿主就收不到「该重新量了」的信号；而 `postMessage` 由事件循环投递，什么时候都到。
   被沙箱化成不同源时宿主读不到 iframe 内部文档，它更是唯一的高度来源。
   同源时仍以现场实测为准，消息只负责「叫醒」宿主。
3. 改高度后再量一轮（最多连量 3 轮）：高度一变，内部可能多出/少掉一条滚动条，宽度跟着变、组件重排，
   内容高度又变（实测能差 60px，表现为底部一大片空白），再量一轮即收敛。
**换笔记栏宽 / 拖窗口时图要跟着变**（四条兜底，专门让「旧组件页 + 新宿主」也不会卡在旧尺寸）：

1. 宿主盯 iframe 宽度，一变就把 `resize` 事件补送进 iframe、再重测高度。组件**只**在自己收到
   `resize` / 自己的 `ResizeObserver` 被投递时才按新宽度重排；那个帧的生命周期没跑时它就停在旧宽度，
   表现为「窗口变宽了图不变宽」甚至内部冒出一条滚动条。主文档的生命周期总是活的，所以这条可靠。
2. 宿主往组件页注入一段兜底样式，强制 `.wg-figure` 不被 `max-height:72vh` 截断——
   旧版组件页（没有 `html[data-wg-embed]` 那条规则）也能整张图撑开、不内部滚动。
3. 组件内交互（点击 / 拖动 / 按键 / 改输入）之后补测一次高度，覆盖旧组件不 `postMessage` 的情况。
4. 组件若按「带瞬时滚动条的窄宽度」画过，宿主发现它用的宽度 ≠ iframe 宽度就再补一次 `resize`
   （每次改宽最多 3 次，有界，不会来回抖）。

**怎么确认自己看的是最新构建**：组件页脚印着 `widget/v1 · build <日期>`（`Maps/_tools/widgets.js` 的
`BUILD` 常量）。改了 `widgets.js` / `widgets.css` 后**必须重建所有已登记组件**（`make_widget.py new …
--force`），否则 `--check` 会报「派生 HTML 与源 spec 不同步」，笔记里看到的还是旧组件。

**已知限制**：Dataview 在**索引版本变化**时（新增/修改文件）会重渲染 dataviewjs 块，那一刻组件会重建、
滑块回到初值、手动拖过的窗口尺寸也回默认。这是它的既定行为（按 `index.revision` 重渲染），不是视图能控制的。

### 10.3 其它通道（都是可选的）

| 通道 | 怎么做 | 状态 |
|---|---|---|
| 系统浏览器直开 | 笔记里放普通 Markdown 相对链接，或直接打开 `Maps/_widgets/*.html` | 单文件自包含、零外部依赖，**最稳的兜底** |
| 官方 `<iframe src="URL">` + 本机服务 | 跑 `python3 Maps/_tools/serve_widgets.py`（默认 8790；8765 是 knowledge-platform 的端口别占），笔记里写 `<iframe src="http://127.0.0.1:8790/...">` | 与官方文档写法一致；代价是看的时候服务要在跑 |
| Agent 预览 | PI-Desktop 的 BrowserPreview 打开 workspace 相对路径 | 可用 |

- **不要**在笔记里写「需要先启动服务」——`dv.view` 那条路不需要任何服务。
- 知识地图是 `build_map.py` 产出的单文件页面，同样用浏览器打开即可。
- 地图当前是**深色**、组件与静态图默认**白底**；要改地图主题得动 `Maps/_tools/template.html` 并重建。


## 11. 完成自检

- [ ] 源笔记存在，且本次没有修改它的任何事实、数字或结论。
- [ ] 组件要回答的那个问题，笔记正文已经讲清楚；去掉组件仍能读懂。
- [ ] 组件里每个数字都能追到源笔记或 `Books/` 原文，并在 `subtitle` 或 `notes` 里注明出处；核不到的已标"未核实"。
- [ ] 控件 ≤ 4 个；每个控件都有写清单位与范围的中文 `label`。
- [ ] `teaching.question`、`sourceSection`、`controlEffect`、`visualEvidence` 都已填写；`teaching.controls` 与 `cameraControls` 口径核对过。
- [ ] 每个 semantic 控件都进入 renderer 的主绘制字段，并能在主图几何或 canvas 调用中留下变化；只动 marker/readout 的控件不算合格。
- [ ] 若写了 `teaching.dynamicRenderer: true`，已运行 `verify_widget_pages.js` 的默认/最小/最大值比较；没有用声明本身替代证据。
- [ ] 窄面板（手机或侧边栏宽度）下标题、控件、读数都还看得见。
- [ ] 默认白底：没有特殊理由就不写 `theme`；确实写了 `"system"` 时，已在深色系统下实测过它真的会切黑。
- [ ] HTML 无外部依赖：无 CDN、无外链脚本/字体/图片。
- [ ] 只用键盘也能操作（Tab 能落到控件上）。
- [ ] `title` 说的是"能看出什么"，不是"某图"。
- [ ] 笔记正文的入口只有：一个 callout、可选的 `dataviewjs` 内嵌块、一条普通 Markdown 兜底链接；**没有**手写 `<iframe>` / `<style>` / `<script>`。
- [ ] 若用了 `dataviewjs` 内嵌：`dv.view` 的路径与 `file` 名字都核对过（`make_widget.py --check` 通过 + 在 Obsidian 里看到图）。
- [ ] 笔记里那条兜底链接已在 Obsidian 里实际点开验证过；没验证的已标"未核实"。
- [ ] `.json` 与 `.html` 同名配对，slug 不与既有组件冲突，没有覆盖任何已有文件。
- [ ] `.html` 没有手工编辑过；改动都发生在 `.json` 上并已重新生成。
- [ ] 密集标记（散点 / 热力图 / 三维面片）**没有**手写死 `renderer`：默认 `auto` 就够；若确实要 `svg`（为了逐点悬停）或 `canvas`（为了省 DOM），在 `notes` 里写清为什么。
- [ ] 若用了 `plotly`：已确认内置渲染器做不到；`bundle` 选了够用的那一份（纯 3D 用 `gl3d`）；生成器的 `by`/`n` 手算核对过一个元素；`--check` 打印的体积分摊可接受。
- [ ] `python3 Maps/_tools/make_widget.py --check` 通过；退出码为 `2` 时只重建派生清单，不重跑登记。
- [ ] `git status` 显示本次只动了 `.json`、派生的 `.html`、`Index.md` 与那一处笔记链接；**没有**往 `Books/` 写入任何内容。

## 12. 静态科学图（matplotlib / 可选 seaborn）

静态图不是交互组件，不登记 `widgets-index.json`。用 `make_figure.py` 的数据 JSON 接口，离线生成 `_attachments/` 下 PNG/SVG；不执行 spec 中的代码，不修改源笔记。必须先有非 `Books/` 的来源笔记、已讲清的机制与数字，图后就近注明来源与计算口径。工具检查路径与 source 非空，**不替代作者核对事实**。

依赖只用于生成，阅读图片无需 Python。需要 Python ≥3.9、matplotlib；seaborn ≥0.12 可选。不要全局 pip 安装；经允许后在隔离环境安装，例如：

```bash
python3 -m venv "$PI_SCRATCH_DIR/figure-venv"
"$PI_SCRATCH_DIR/figure-venv/bin/python" -m pip install matplotlib
# 仅需 seaborn 时：
"$PI_SCRATCH_DIR/figure-venv/bin/python" -m pip install 'seaborn>=0.12'
export MPLCONFIGDIR="$PI_SCRATCH_DIR/matplotlib"
```

将以下接口演示保存为 `$PI_SCRATCH_DIR/figure.json`。这些是 $y=x^2$ 的演示计算，**不是教材数据**；正式使用时换成已在源笔记解释的数据与出处：

```json
{
  "title": "Square grows faster than x",
  "source": "接口演示：y=x^2，本节计算，非原文照抄；正式使用需定位源笔记段落",
  "engine": "matplotlib",
  "kind": "line",
  "x": [0, 1, 2, 3],
  "y": [0, 1, 4, 9],
  "xlabel": "x",
  "ylabel": "y"
}
```

```bash
"$PI_SCRATCH_DIR/figure-venv/bin/python" Maps/_tools/make_figure.py --help
"$PI_SCRATCH_DIR/figure-venv/bin/python" Maps/_tools/make_figure.py \
  --note "Maps/Notes/已有笔记.md" --spec "$PI_SCRATCH_DIR/figure.json" \
  --output "_attachments/figures/square.svg"
PYTHONDONTWRITEBYTECODE=1 "$PI_SCRATCH_DIR/figure-venv/bin/python" Maps/_tools/test_figures.py
```

- `title` / `source` / `kind` 必填；`engine` 默认 `matplotlib`，可改 `seaborn`。不是所有 kind 都有 seaborn 实现：`box` / `ecdf` 走 seaborn，`contour` / `vector` / `qq` 一律由 matplotlib 画。不接受表达式、远程数据或非有限数值。seaborn 条形图会对重复 x 取均值；需要逐项条形时用 matplotlib 或先明确聚合口径。
- 十个 kind 的数据形状（完整表以 `make_figure.py --help` 为准，这里是速查）：

  | kind | 必填键 | 形状与口径 |
  |---|---|---|
  | `line` / `scatter` / `bar` | `x[]` `y[]` | 等长一维有限数值 |
  | `histogram` | `data[]` | 一维；可选整数 `bins`（1–1000，默认 10） |
  | `heatmap` | `data` | 二维等宽矩阵，≤1000 行、≤100000 格 |
  | `contour` | `x[]` `y[]` `z` | `z` 行数 = `len(y)`、列数 = `len(x)`；`z[j][i]` 对应 `y[j]`、`x[i]`；可选整数 `levels`（1–100） |
  | `vector` | `x[]` `y[]` `u` `v` | `u`/`v` 按 `meshgrid(x, y)`（`'xy'`）展平，长度 = `len(x) * len(y)`，下标 = `j*len(x)+i` |
  | `box` | `groups` | 非空的数组的数组（每组一维数值）；可选 `labels`（等长文本） |
  | `ecdf` | `series` | `[{label, values}]`；x 升序，`y = (i+1)/n`，阶梯画在 `where='post'` |
  | `qq` | `data[]` | 至少 2 点；可选 `dist` = `normal`（默认 N(0,1)）/ `uniform` / `exponential`；口径：样本次序统计量对 `p_i = (i-0.5)/n` 的理论分位，参考线过首末两点（斜率只在位置-尺度族里近似 σ） |
- stdout 只输出可粘贴的普通 Markdown 相对嵌入 `![标题](../../_attachments/figures/square.svg)`（特殊路径字符百分号编码）。核对图形后放在源笔记对应讲解段落后，另附可见来源说明；不自动插入、不新增 registry。
- 默认拒绝覆盖；核对同名图后才加 `--force`。输入或依赖错误退出 1，成功 0；CLI 参数语法错误由 argparse 退出 2（不是交互工具的登记状态）。
- PNG/SVG **固定不透明白底**，不会自动跟随系统深色主题；未实现自动双图切换，不提供虚假的 `system` 参数。`title` / `xlabel` / `ylabel` **必须写英文**：绘图默认字体没有中文字形，中文会渲染成方框，工具会直接报错；`source` 可用中文（只进图片元数据，不进图面），确实已配置中文字体时才加 `--allow-cjk`。标签写单位与口径（`Discount rate y (%)`），不要只写 `y`。
- 源 JSON 如需长期复现，可经冲突检查保存在图旁；工具不自动保存副本。测试文件和缓存只放 scratch，图片生成不联网。

### 交互图的主题边界

交互图默认**白底**：`html` / `body` / `.wg-wrap` 三层都是白底，生成页也不再声明 `color-scheme: dark`。跟随系统黑白是**显式 opt-in**：spec 里写 `"theme": "system"`，运行时把它落到 `.wg-wrap[data-theme="system"]` 与 `<html data-wg-theme="system">`，CSS 的 `prefers-color-scheme` 才接管；不写就固定白底。生成器校验 `theme` 只能取 `light` / `system`，其余值报错而不是静默忽略。限制：`custom` 里作者自写的颜色、以及改样式之前生成的旧 HTML 都不保证跟随——改样式后必须由原 spec 重建（`new … --force`）并在浏览器里核对；静态图不使用此机制。

## 13. 数据量、渲染后端与实测基准

这一节是**实测记录**（headless Chromium 软件光栅化，1280×720，渲染总耗时中位数；重跑方法见本节末），
不是理论估计。改阈值、改分块常数、或想给某个 kind 加 canvas 支持之前先看这里。

### 13.1 为什么要有 canvas 后端

成本几乎全在「一个标记一个 DOM 节点」上。同一批数据，SVG 与 canvas 两条路的对照：

| 场景 | SVG 节点数 / 耗时 | canvas 节点数 / 耗时 |
|---|---|---|
| `scatter` 1k 点 | 2048 / 7.9 ms | 46 / 1.8 ms |
| `scatter` 10k 点 | 20048 / 54.9 ms | 46 / 4.1 ms |
| `scatter` 50k 点 | 100048 / 245.1 ms | 46 / **17.2 ms** |
| `heatmap` 120²（14400 格） | 14670 / 55.7 ms | 272 / 37.6 ms |
| `heatmap` 200²（40000 格） | 40430 / 148.0 ms | 432 / 95.4 ms |
| `surface3d` 121²（14400 面） | 14450 / 169.7 ms | 52 / 162.5 ms |

对照：`contour` 121² 的节点数**恒为 82**（成本是 marching squares 的算术，13.2 / 41.3 / 87.0 ms），
所以它不需要 canvas 后端。

交叉点大致在 `scatter` 300 点、`heatmap` 40²、`surface3d` 25²（小数据上 canvas 略慢：建层、量矩形、dpr 都有固定开销）。
阈值取 **`scatter` 1000 / `heatmap` 3600 / `surface3d` 1000**：比交叉点保守，因为 canvas 上会丢掉逐点 `<title>` 悬停，
数据量不大时宁可用 SVG 换悬停。`WG.canvasPolicy` 把这三个数字、单路径最大圆数与后备存储像素上限公开出来，
`test_widget_runtime.js` 会逐项钉住——**改阈值要连测试一起改**。

### 13.2 canvas 上的`arc`：必须分块提交

**这是踩过的坑，不是风格问题。** 把一张散点图的全部圆点塞进**一条** canvas 路径再 `fill()` 一次，
看起来"只提交一次、最省"，实际会让光栅化器把整条路径按统一高精度规则扁平化，代价随弧数**超线性**增长：

| 提交方式（圆，r = 2.6 px） | 20k 点 | 50k 点 |
|---|---|---|
| **单路径 + 一次 `fill`**（最慢，别写） | **247 000 ms** | —— |
| 每 16 个一批 + 每批一次 `fill` | 6.3 ms | —— |
| 每 64 个一批 + 每批一次 `fill` | 4.3 ms | —— |
| **每 256 个一批 + 每批一次 `fill`**（现在用的） | **2.7 ms** | **6.9 ms** |
| 每 1024 个一批 + 每批一次 `fill` | 2.7 ms | —— |
| 逐点 `beginPath` + `arc` + `fill` | 7.7 ms | 33.5 ms |
| 逐点 `fillRect`（方点） | 2.6 ms | 13.4 ms |

（`Path2D` 圆 + 逐点 `setTransform` + `fill` 是 35.5 ms / 20k —— 反而更慢，因为它每次都要换变换矩阵。）

所以 `widgets.js` 用 `CANVAS_ARC_CHUNK = 256` 分批：批内共用一条路径、一次 `fill`，批间重置。
`test_widget_runtime.js` 里有一条断言专门盯这件事（统计"`beginPath`→`fill` 之间出现过几个 `arc`"，不许超过 256）。
**直线、矩形、多边形没有这个病**（2 万个矩形的单路径只要 8.7 ms），别把这条结论套到热力图格子上。

### 13.3 数据量预算（构建期就拦住）

`make_widget.py` 的校验按上限/提醒线拦截，避免 spec 悄悄生成十万个节点：

| 字段 | 上限（硬错误） | 提醒线 | 说明 |
|---|---|---|---|
| `series[].points` / 顶层 `points` | 20000 | 5000 | 抽稀口径可复用 `ecdf`/`qq` 的 `maxPoints`（等间隔保留、端点精确） |
| 同上 + `renderer: "canvas"` | 200000 | 100000 | canvas 上 10 万点 ≈ 24 ms/帧 |
| `heat.rows × heat.cols` | 40000 | 10000 | 200×200 |
| `data.n` | 1–2000 | —— | 逐点表达式生成器（与运行时 clamp 一致） |

### 13.4 重跑基准的方法

测量装置（`measure.html` 一类把 `widgets.js`/`widgets.css` 内联进页面的小文件）是临时件，放在会话 scratch 里，
不登记进仓库。要点：同机 A/B、加一次预热、取 3 次中位数、断言节点数与 `[错误]` 数一起记录（只报 ms 容易被误读）。
`scatter` 用「一个 mark 一个节点」的写法生成点集，`heatmap`/`surface3d` 用规则网格；
测 canvas 时对比同一 spec 的 `renderer: "svg"` 与 `renderer: "canvas"` 两版。
