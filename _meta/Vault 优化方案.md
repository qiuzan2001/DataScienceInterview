---
title: Vault 优化方案
aliases: [Optimization Plan, 优化方案]
tags: [meta, plan, vault-maintenance]
status: 待执行
updated: 2026-09-16
scope: 全库 42 篇笔记 / 约 8150 行
---

# Vault 优化方案

> 本笔记是执行清单。完成后 `status` 改 `已执行`，并在 §9 验收表逐项打勾。
> 基线实测见 §0；体检脚本 `_meta/tools/vault_check.sh`，输出存档 `_meta/baseline_report.txt`。

---

## 0. 基线实测（2026-09-16 首次运行 `bash _meta/tools/vault_check.sh`）

| 指标 | 基线 | Batch 后目标 |
|---|---|---|
| 笔记数 / 行数 | 42 / 8146 | — |
| 空文件 | **1**（`07_Random_Forest/7.2.2.4. Pruning Decision Trees.md`） | 0 |
| 悬空 wikilink | **5**（全部在 `01_LR_GLMs/Logistic Regression Template.canvas`） | 0 |
| AI 对话残留 | **8 处 / 8 个文件** | 0 |
| 缺 frontmatter | **42 / 42（全库）** | 0 |
| 双井号标题 | **2**（`01_LR_GLMs/1.7 Assumptions.md:40,61`） | 0 |
| 转义残留 `\$` | **4 处**（`1.6.3 Regularization.md:6,147,148`、`7.5 CART Algorithm.md:63`） | 0 |
| 一级编号跳号 | 第2章缺 2.1–2.2；第3章缺 3.1–3.3；第5章缺 5.1；第7章缺 7.1、7.3；第8章缺 8.1、8.2、8.5 | 按 §4 规范收敛 |
| 实测硬伤 | SAS 代码 **0 行**；`3.5.2 SAS VARCLUS.md` 名为 SAS 但无 `PROC` 语句 | 09 章 |

> 注：脚本本身按 GNU/BSD 兼容写法（`sed` 不用 `\|` 交替），已在本机 macOS 实测通过。

---

## 1. 目标与原则

**目标**：把现在「8 章 AI 生成的教材式笔记」升级为「可用来自测、能扛住面试追问、结构自洽」的知识库。

**三条原则**

| # | 原则 | 含义 |
|---|---|---|
| P-a | **单一事实来源（SSOT）** | 每个概念只在一处完整推导，其它地方只放结论 + 链接。现状：对数似然出现 4 次、假设清单 5 次、熵/信息增益 4 次、AUC 3 次、IV 阈值 4 次——任一处改错就会分叉 |
| P-b | **面试导向** | 每个知识点最终要能回答「30 秒版 + 追问 + 反例」。现状：库很擅长推导，但**零**保险业务语境、零 SAS 代码、零自测题 |
| P-c | **可校验** | 断链、空文件、编号跳号、AI 对话残留，全部由脚本查，不靠肉眼。§7 提供脚本 |

---

## 2. 目标状态（目录蓝图）

```
_meta/                      ← 新建：规范、模板、校验脚本、本方案
00 Index.md                 ← 新建：唯一入口（四章叙事 + 8 章目录 + 学习路径 + 题库入口）
01_LR_GLMs/                 ← 编号不动，仅修父笔记伪编号（见 P0-6）
02_Transformations/
03_Multicollinearity/
04_Missing_Data/
05_Dimension_Reduction/
06_Model_Assessment/
07_Random_Forest/           ← 扁平化重编号（见 P1-1，可选）
08_ML_Methods/
09_SAS_风控实务/            ← 新建：09 章，全库最大缺口
99_QuestionBank/            ← 新建：自测题 + 答案折叠
Images/
Data Science Interview Knowledge Base.md   ← 降级为「速查卡」，入口交给 00 Index
```

---

## 3. 批次与任务清单

工时按「我执行」估算；每批次一个 git commit，可独立回滚。

### Batch 0 · 准备（30 min）

| ID | 动作 | 验收 |
|---|---|---|
| 0-1 | **先提交当前未提交改动**（`04_Missing_Data/4. Missing Data.md`、`08_ML_Methods/8. Ensemble Learning.md`、根笔记已 M；`8.6 Bagging VS Boosting.md` 未跟踪），保证后续 diff 干净 | `git status` 仅剩 `.DS_Store`/`.obsidian` |
| 0-2 | 建 `_meta/`（规范、模板、脚本、本方案） | 目录存在 |
| 0-3 | 放入 `_meta/tools/vault_check.sh`（§7），跑出**基线报告** | 报告存档 `_meta/baseline_report.txt` |
| 0-4 | 定稿命名规范与模板（§4/§5） | 后续批次照此执行 |

### Batch 1 · P0 硬伤修复（3–4 h）

按「不修就会误导读者」排序。

| ID | 问题 | 动作 | 涉及文件 |
|---|---|---|---|
| 1-1 | **空文件**，却被 7.2.2 标题引用 | 写满内容：预剪枝 vs 后剪枝、代价复杂度 CCP `R_α(T)=R(T)+α\|T\|`、weakest-link 序列、α 的 1-SE 规则、剪枝对偏差方差的影响；与 7.5 CART 去重（CART 只留算法本体，剪枝细节全放这里） | `07_Random_Forest/7.2.2.4. Pruning Decision Trees.md` |
| 1-2 | **WOE 符号三处不一致**（最危险的错误） | 定死 `WOE_k = ln(P(X=k\|Y=0)/P(X=k\|Y=1)) = ln(%Goods/%Bads)`，正 WOE = 低风险；改 `2. Transformations.md` L306 / L320-324 / L331；`2.3.1 WOE & IV.md` 加一句「本笔记的 %non-events/%events 即 Goods/Bads，两者等价」；`5.2 Univariate Selection.md` 对齐；三处互加 cross-ref | 02、05 目录 3 个文件 |
| 1-3 | **AdaBoost Round 2 表格整列错位**（预测列沿用了 Round 1，与 §正文「误分 1 和 2」、ε=0.3324 矛盾），且缺 Step 3 权重更新 | 重算 Round 2 的 stump 预测列、e₂、α₂、D₃，补 Step 3 | `08_ML_Methods/8.4.2.1 AdaBoost (Adaptive Boosting).md` |
| 1-4 | **Boosting 是否降方差三处口径冲突** | 统一为「主要降偏差；方差取决于 shrinkage 与早停，多轮无正则会上升」，`8. Ensemble Learning.md` L107 与对比表 L158 对齐，链向 `8.6` | 08 目录 |
| 1-5 | **断章** 4 处 | ① `4. Missing Data.md` MICE 补第 3 步 Pooling（Rubin 规则：合并估计、within/between/total variance）＋闭合表格；② `3.4.2 VIF.md` 补 §7 Remedies（正文引用了却缺）；③ `2.3.1 WOE & IV.md` 补 §9（IV 的坑：过拟合筛选、样本依赖、多值类别）；④ `1.7 Assumptions.md` 收尾并修 `### ## 2.` / `### ## 3.` 双井号；⑤ `6.4 Diagnostic & Visualization Tools.md` 补承诺过的 ROC/阈值一节 | 01/02/03/04/06 |
| 1-6 | **编号撞名** | `1. Logistic Regression & GLMs.md` 里 `## 1.7 Evaluation via Confusion Matrices` 与子笔记 `1.7 Assumptions` 同号异内容；`## 1.8 GLM vs. GBM` 无子文件。→ 去掉父笔记这两节的数字前缀（改「延伸：…」并链到 `6.3`），**不动子文件名**（零链接风险） | `01_LR_GLMs/1. Logistic Regression & GLMs.md` |
| 1-7 | **AI 对话残留 8 处** | 删除或改写为正文：`1.4:93`、`1.7` 尾、`2.3.1:175`、`3.5.4 PCA:147`、`6.3:111`、`6.4:74`、`7.2.2.3.1.9:1` 与 `:230`、`7.2.3.3:151` | 8 处 |
| 1-8 | **渲染损坏** | ① `2.3.1`、`8.4.2.1` 的 `>[!example]-` 折叠块内多张表格缺 `> ` 前缀（会断出框外）；② `1.6.3:147` 的 `\$\lambda\$` 显示反斜杠；③ 拼写 `leakege`→`leakage`（`6. Model Assessment.md:5`）、`Engineers features`→`Engineer features`（`7.4:6`）；④ `8.4.2.1` L100 折叠块；⑤ `8.6` L23-27 有序列表序号回跳 | 6 处 |
| 1-9 | **canvas 死链 5 个** | `Logistic Regression Template.canvas` 中 `[[1. Regression]]`、`[[Imbalanced Data]]`、`[[Dimensionality Reduction]]`、`[[Univariate Methods]]`、`[[Multivariate Methods]]` → 分别改指 `5. Dimension Reduction`、`1.5 Dealing with Unbalanced Samples`、`5. Dimension Reduction`、`5.2 Univariate Selection`、`5.3 Multivariate Selection` | canvas |

### Batch 2 · P1 结构与规范（3–4 h）

| ID | 动作 | 说明 |
|---|---|---|
| 2-1 | **07 章扁平化重编号**（可选，风险已控） | 深度 `7.2.2.3.1.9` 已达 5 层且跳号。建议映射：<br>`7.2. Building the Forest` → `7.1 建树流程`<br>`7.2.2 Decision Tree` → `7.2 决策树总览`<br>`7.2.2.3.1 ID3 Algorithm` → `7.3 ID3`<br>`7.2.2.3.1.4 ID3 Complete Code` → `7.3.1 ID3 代码`<br>`7.2.2.3.1.9 ID3 Build Example` → `7.3.2 ID3 手算`<br>`7.2.3.3 C4.5 Algorithm` → `7.4 C4.5`<br>`7.5 CART Algorithm` → `7.5 CART`（不变）<br>`7.2.2.4. Pruning` → `7.6 剪枝`<br>`7.4. Quantifying Feature Importance` → `7.7 特征重要性`<br>**用脚本 `git mv` + 全库重写 wikilink/canvas 引用**，不手工逐个改 |
| 2-2 | **全库补 frontmatter** | 42 篇统一模板（§5），`status` 区分 `完成/草稿/待补`，为后续 Dataview/题库筛选铺路 |
| 2-3 | **统一笔记骨架** | 每篇：`🎯 一句话定义` → `📐 公式与推导` → `⚠️ 假设/前提` → `🛠️ 实操要点` → `🗣️ 面试速答（30 秒）` → `🔗 相关`。**新增「面试速答」块是本次最大收益点** |
| 2-4 | **SSOT 去重** | 把重复推导（对数似然、假设清单、熵/IG、AUC、IV 阈值）各自收敛到唯一「主笔记」，其余改为一句结论 + 链接；主笔记在 frontmatter 标 `canonical: true` |
| 2-5 | **新建 `00 Index.md`** | 唯一入口：四章叙事（Intro → LR Overview → Discovery → Measuring Performance）+ 8 章目录表 + 3 条学习路径（速成 3 天 / 系统 3 周 / 面试前夜速览）+ 题库入口 |
| 2-6 | **根笔记降级** | `Data Science Interview Knowledge Base.md` 保留为「一页速查卡」（去掉与子笔记重复的推导），顶部指向 `00 Index` |

### Batch 3 · P2 内容补齐（1–2 天，面试价值最高）

**3-A 新建 09 章：SAS 风控实务**（现状：全库 SAS 代码为零，但 `3.5.2 SAS VARCLUS` 以 SAS 命名——名字与内容严重不符）

| 笔记 | 内容要点 | 对应现有章节 |
|---|---|---|
| `09.1 PROC LOGISTIC 建模全流程` | `model y(event='1')=x1-xk / selection=stepwise slentry=0.05 slstay=0.05 lackfit rsquare;`、`ctable pprob=0.5`、`outroc=`、`plots=roc`、`score` 语句出分 | 01 章 |
| `09.2 PROC VARCLUS 变量聚类` | 完整参数：`maxeigen=1`、`proportion=0.7`、`summary`、`reduce`、输出 `R²` 与 `R²_own/next` 解读；补齐 `3.5.2` 缺失的代码 | 03 章 |
| `09.3 WOE/IV 与分箱` | `PROC HPBIN`/`PROC FREQ` 分箱、WOE/IV 宏骨架、fine→coarse、单调性检查 | 02 章 |
| `09.4 PROC MI / MIANALYZE` | `proc mi nimpute=10 seed=…;` + `proc mianalyze;`——把 04 章的 MICE 落到 SAS | 04 章 |
| `09.5 降维与筛选` | `PROC PRINCOMP`/`PROC FACTOR`、`PROC SURVEYSELECT` 分层抽样 | 05/06 章 |
| `09.6 树模型与评分卡上线` | `PROC HPSPLIT`、`PROC GRADBOOST`、`PROC GENMOD`（poisson/link=log）；评分卡刻度化 `score = A − B·ln(odds)`、PDO 换算 | 07/08 章 |

**3-B 风控/保险面试高频缺口**（现有库完全没有或仅一句带过）

1. **KS 统计量**（`KS = max|CDF_good − CDF_bad|`，>0.3 良好）与**十分位 Lift 表**、Gini/AR
2. **PSI / CSI** 稳定性监控（<0.1 稳定 / 0.1–0.25 关注 / >0.25 重训）与 out-of-time 验证
3. **评分卡刻度化**：`score = A − B·ln(odds)`，PDO、base score/odds 的换算
4. **采样偏差校正**：过采样后截距回填 `logit(p) − ln(w)`；类权重与「风控不用 SMOTE」的取舍
5. **拒绝推断 reject inference**（parcelling / fuzzy augmentation）——面试常问却全库未提
6. **计数/保费模型**：Poisson / NB / 零膨胀、`offset=log(exposure)`；Gamma 严重度；纯保费 = 频率 × 严重度 → **Tweedie**
7. **成本敏感阈值 / profit curve**（1.5 只有一句「connect to real-world costs」）
8. **变量筛选的稳定性**：bootstrap 重抽 + 时间外样本；选择后推断偏差（多重比较）
9. **单调约束 / 可解释约束**（受监管场景）
10. **时间维度 CV**：rolling window / out-of-time 切分（现在只有一句话）

**3-C 业务语境**：为 3-B 每条配「车险/家财险」示例段落（续保流失、理赔欺诈、风险分级定价、索赔频率 vs 严重度），把现在的信贷例子升一层。**不编造任何 State Farm 内部数据或流程**，只用公开的通用保险概念。

### Batch 4 · P3 增强（0.5–1 天）

| ID | 动作 |
|---|---|
| 4-1 | `99_QuestionBank/`：每章 5–8 题，共 50 题；题目 + `> [!success]- 答案` 折叠，含「追问链」与「常见错误答法」 |
| 4-2 | 每章末尾「面试速答卡」（Batch 2-3 已埋骨架，这里批量填实） |
| 4-3 | 可选：装 Dataview 后生成「按 status/tag 的自动待办面板」；对接 spaced repetition 插件做复习队列 |
| 4-4 | 可选：`hugo`/Obsidian Publish 站点结构（`publish.json` 已存在但 `included` 为空） |

---

## 4. 命名与编号规范（定稿）

1. 文件名 = `编号[.子编号] 标题.md`，编号必须与父笔记内的标题编号**逐字一致**。
2. 子编号**不跳号**；最多 4 层；超过 4 层说明应该拆章。
3. 同一编号在库内**全局唯一**（禁止出现两个 `1.7`）。
4. 父笔记（`N. 章名.md`）只做 MOC + 速查卡，**不重复**子笔记的完整推导。
5. 不用 `&` 以外的特殊字符；`_` 与空格保持现有风格，不做无意义改名（改名必须脚本化 + 同步重写链接）。
6. `_meta/`、`00 Index.md`、`99_QuestionBank/` 为保留命名，不参与编号体系。

---

## 5. 笔记模板

**frontmatter**

```yaml
---
title: 7.3 ID3
chapter: 07
tags: [ml/tree, algorithm, interview]
status: 完成        # 完成 | 草稿 | 待补
canonical: false    # 该概念的唯一权威笔记标 true
updated: 2026-09-16
aliases: [ID3 算法]
---
```

**正文骨架**

```markdown
# <标题>

## 🎯 一句话
## 📐 公式与推导
## ⚠️ 假设 / 前提 / 失效场景
## 🛠️ 实操（SAS / Python 代码）
## 🗣️ 面试速答（30 秒 + 3 个追问）
## 🔗 相关笔记
```

---

## 6. 自动化校验脚本

`_meta/tools/vault_check.sh`（只读，不改文件）：

```bash
#!/usr/bin/env bash
# vault 体检：断链 / 空文件 / 跳号 / AI 残留 / frontmatter
cd "$(dirname "$0")/../.." || exit 1
echo "== 空文件 =="; find . -name "*.md" -not -path "./.git/*" -empty
echo "== 未解析 wikilink =="
grep -rhoE '\[\[[^]|#]+' --include="*.md" . | sed 's/^\[\[//' | sort -u | while read -r l; do
  find . -name "${l}.md" -not -path "./.git/*" | grep -q . || echo "  悬空: [[$l]]"
done
echo "== AI 对话残留 =="
grep -rnE "Let me know if|Great idea|Absolutely —|Here is a detailed|I've assumed you know|Got it —" --include="*.md" . \
  | grep -v "^./_meta/"
echo "== 缺 frontmatter =="
for f in $(find . -name "*.md" -not -path "./.git/*" -not -path "./_meta/*"); do
  head -1 "$f" | grep -q '^---$' || echo "  $f"
done
echo "== 双井号标题 =="; grep -rn '^#\+ ## ' --include="*.md" .
echo "== 转义残留 =="; grep -rn '\\\$' --include="*.md" .
```

**验收门槛**：Batch 1 结束时应为「空文件 0 / 悬空链接 0 / AI 残留 0 / 双井号 0」；Batch 2 结束时应为「缺 frontmatter 0」。

---

## 7. 排期与里程碑

| 里程碑 | 内容 | 工时 | 完成后的状态 |
|---|---|---|---|
| M0 | Batch 0 基线 | 0.5 h | 体检脚本 + 基线报告可就绪，git 干净 |
| M1 | Batch 1 硬伤清零 | 3–4 h | 不再有错内容、空文件、断章、死链 |
| M2 | Batch 2 结构规范 | 3–4 h | 编号自洽、全库有 frontmatter、有唯一入口 |
| M3 | Batch 3 内容补齐 | 1–2 d | 有 09 SAS 章 + 风控高频考点，能直接支撑面试 |
| M4 | Batch 4 题库 | 0.5–1 d | 50 题自测 + 每章速答卡 |

**建议**：M1 单独 commit（`fix: 硬伤清零`），M2 一个（`refactor: 结构规范`），M3 按 3-A/3-B/3-C 分三个（`feat: SAS 章` / `feat: 风控考点` / `feat: 保险语境`）。任何一步不满意可 `git revert` 单个 commit。

---

## 8. 风险与回滚

| 风险 | 缓解 |
|---|---|
| 改名断链 | 只走脚本 `git mv` + 批量重写 `.md` 与 `.canvas` 内链接；改完全库 grep 校验；**不在 Obsidian 外手工改名** |
| 与 Obsidian 自动改链冲突 | app.json `alwaysUpdateLinks: true`；批量改名时先关掉 Obsidian 或改名后重启校验 |
| 内容改写引入新错误 | 每批次后跑体检脚本 + 抽查 diff；AdaBoost 表格这类数值改动**重算一遍**再写入 |
| 一次改太多无法定位 | 严格按批次 commit；`.obsidian/workspace.json` 等噪音文件不混入内容 commit |
| 业务语境编造 | 只写公开通用保险概念，不臆造 State Farm 内部流程/数据 |

---

## 9. 验收清单

- [ ] 全库无空文件
- [ ] wikilink 悬空数为 0（含 canvas）
- [ ] WOE 符号全库唯一口径
- [ ] AdaBoost 算例逐轮数值自洽（e_t / α_t / D_t 三列可对上）
- [ ] Boosting 方差表述统一
- [ ] 5 处断章补齐，折叠块渲染正常
- [ ] 42+ 篇均有 frontmatter，编号无撞名、无跳号
- [ ] `00 Index.md` 成为唯一入口，根笔记降级为速查卡
- [ ] 09 章 6 篇 SAS 笔记可运行/可粘贴（关键参数齐全）
- [ ] 10 个风控高频考点落地
- [ ] 题库 ≥50 题，答案折叠
- [ ] 每篇有「面试速答」块

---

## 10. 需要你决策

1. **07 章是否扁平化重编号**（2-1）？收益：编号可读；代价：6 个文件路径变化（链接我会同步重写）。
2. **根笔记**保留为速查卡，还是把内容并入 `00 Index.md` 后删除？
3. **面试侧重**：SAS 风控（09 章优先）还是 Python/ML 深度（07/08 章加深）？决定 M3 的排序。
4. **题库形式**：只要题+答案，还是每题带「追问链 + 常见错误答法」？
5. 是否允许我在库内建 `_meta/` 与 `99_QuestionBank/`（会多两个目录出现在 Obsidian 侧栏）。
