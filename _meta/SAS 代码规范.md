---
title: SAS 代码规范
chapter: meta
tags: [meta, sas, convention]
status: 完成
updated: 2026-09-16
---

# SAS 代码规范（保险风控场景）

> `_meta` 文件不参与学习，只作为**写代码/写笔记时的口径依据**。
> ⚠️ **本文件的 SAS 语法需在实际环境中验证**（版本、组件、发行包不同，选项可用性可能不同）。凡标「需核对」处，不要当成确定事实。

---

## 1. 适用范围

- 第 10 章「保险风控实务」里所有 SAS 片段
- 任何新增的、要在保险风控语境下复用的 SAS 代码（评分卡、频率-严重度、监控）

## 2. 命名与结构

| 对象 | 约定 | 示例 |
|---|---|---|
| 数据集 | 小写下划线，语义前缀 | `claims_raw`、`card_woe`、`scorecard_out` |
| 中间/临时 | `_tmp_` 前缀 | `_tmp_bin` |
| 变量 | 小写下划线，金额带单位后缀 | `loss_amt`、`ln_exposure`、`pct_good` |
| 宏变量 | 大写 + `_` | `&PI_POP`、`&PDO` |
| 成员后缀 | 与语境一致 | `veh_type`、`age_bnd`（`_bnd` = 分带） |

**结构固定为 5 段**（便于他人接手与面试讲解）：

```sas
/* 0) 参数与宏变量 —— 所有可调阈值放这里，不散落在正文 */
/* 1) 数据准备 —— 暴露期、派生变量、粒度对齐 */
/* 2) 分箱 / 特征处理 —— 切点必须固化并落盘 */
/* 3) 建模 / 评估 —— 主模型 + 对照 */
/* 4) 打分 / 上线 —— 输出表结构固定、带版本号 */
```

## 3. 硬性约定（踩过坑的）

1. **`log()` 是自然对数**；`log10()` 才是常用对数。别在 SAS 里写「ln」。
2. **offset 必须是已经取过对数的变量**：`model ... / offset=ln_exposure;`
   SAS 不会替你取 log（与 R 的 `offset=log(x)` 不同）。
3. **`offset` 的系数固定为 1**，是「已知常数」，不是待估参数——不要改成普通自变量。
4. **类别变量一律显式写编码**：`class x (ref='base') / param=ref;`
   默认效应编码下输出的是「与均值之差」，不是相对数。
5. **事件方向显式写**：`model y(event='1') = ...`，不要依赖排序（`event=` 与 `descending` 二选一，写清哪个）。
6. **相对数统一用 `exp(Estimate)`** 表示；写报告时标明「相对数以基准类别 = 1」。
7. **任何随机过程都要有 `seed=`**（`PROC MI` / `PROC HPSPLIT` / `PROC GRADBOOST` / 抽样），保证可复现。
8. **`PROC MI` 之后必须 `by _imputation_;` + `PROC MIANALYZE` 汇总**，禁止只报单份结果。
9. **过散布诊断看 `Pearson Chi-Square/DF`**；quasi-Poisson 不是似然模型，**不要用它比 AIC**。
10. **分箱切点必须落盘**（数据集或 macro 变量），线上打分层严禁重新分箱。
11. **WOE 口径全局唯一**：`WOE = ln(%Good/%Bad)`，Good = 非事件，**正 WOE = 低风险**（与 [[2.3.1 WOE & IV]] 一致）。
12. **`PROC LOGISTIC` 的 `descending` 与 `model ... (event='1')` 不要重复使用**，避免把事件方向搞反两次。
13. **SAS 9.4 与 Viya 要分清**：`PROC GRADBOOST` 属 Viya/CAS 环境；9.4 侧常见 `PROC HPFOREST` / `PROC HPSPLIT`（可用性需核对）。

## 4. 可复用骨架：WOE / IV（PROC SQL 版）

```sas
/* 输入：cell_counts = PROC FREQ 的 tables bin * y / sparse out= 输出；
         列名约定：bin, y, count
   输出：woe 表（bin, n_good, n_bad, woe, iv_contrib） */
proc sql;
  create table totals as
    select sum(y=0) as tot_good, sum(y=1) as tot_bad
    from cell_counts;

  create table woe as
    select c.bin,
           sum(c.count*(c.y=0)) as n_good,
           sum(c.count*(c.y=1)) as n_bad,
           log( ((sum(c.count*(c.y=0))+0.5)/(t.tot_good+1))
              / ((sum(c.count*(c.y=1))+0.5)/(t.tot_bad +1)) ) as woe,
           calculated woe * ( ((sum(c.count*(c.y=0))+0.5)/(t.tot_good+1))
                            - ((sum(c.count*(c.y=1))+0.5)/(t.tot_bad +1)) ) as iv_contrib
    from cell_counts as c, totals as t
    group by c.bin;
quit;
```

要点：
- `+0.5 / +1` 是 0.5 平滑（Laplace/Jeffreys），避免 $\ln(0)$
- `calculated woe` 是 PROC SQL 在同一 SELECT 内复用前面算好的列
- 算完**必须先检查 WOE 是否单调**再进模型（顺序变量上）

## 5. 交付物清单（一个模型上线前）

| 交付物 | 内容 | 校验 |
|---|---|---|
| 建模数据集 | 粒度（保单/赔案）、暴露期定义、时间窗 | 行数、唯一键、暴露期分布 |
| 分箱映射表 | 变量 → 切点 → WOE | 切点覆盖率 100%，缺失单独一箱 |
| 模型系数表 | 变量、系数、标准误、p、相对数 | 符号与业务方向一致 |
| 评分映射表 | 每个箱的整数分值 + 基准分 | 手工验算 1 条样本 |
| 评估报告 | KS / Gini / AUC、十分位 Lift、校准表、OOT | 标注样本期与先验 |
| 监控配置 | 分数 PSI、各变量 CSI、阈值与处置动作 | 切点与基准期固化 |

## 6. 自检清单（提交前逐条打勾）

- [ ] `log()` 用法正确，offset 已取对数且系数固定为 1
- [ ] 类别变量有 `param=ref`，事件方向显式声明
- [ ] 所有随机过程都有 `seed=`
- [ ] 过散布已诊断（`Pearson Chi-Square/DF`），并说明处置
- [ ] WOE 方向为「正 = 低风险」，分箱单调性已检查
- [ ] 多重插补用了 `PROC MIANALYZE` 汇总
- [ ] 评估含 **OOT** 与**校准**，不只报 AUC
- [ ] 阈值与切点已固化，可复现、可审计
- [ ] 不确定的选项已在注释标「需核对」并在真实环境验证

## 7. 备注

- 本规范只为**统一口径、避免重复踩坑**，不代表任何机构的内部流程或要求。
- 与第 10 章正文一致：[[10.7 SAS 实操速查]]、[[10.4 评分卡与 WOE 落地]]、[[10.5 模型评估与监控]]。
