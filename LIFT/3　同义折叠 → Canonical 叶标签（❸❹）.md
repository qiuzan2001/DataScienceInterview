### ❷ 行级标签：只让 LLM 给“一句话”贴英文意图词

> **场景假设**：每次 LLM 只看到 *1 条英文摘要*（≈ 10-25 words），返回一个或多个 **leaf-intent tags**。下面解释如何把这一 seemingly-trivial 步骤做得 **稳定、可控、可扩展**。

---

## 1 目标再细化

| 质量维度     | 理想状态                       | 失败典型                                 | 成功度量               |
| -------- | -------------------------- | ------------------------------------ | ------------------ |
| **完整性**  | 每个 summary 都收到 ≥1 tag      | 空标签 / API 超时                         | 行数恒等 (100 %)       |
| **一致性**  | 类似语义用同一拼写                  | “refund-request” vs “request-refund” | 标签重用率↑ / 新标签增速↓    |
| **可解析性** | 100 % 符合 JSON schema       | LLM 输出多余文本                           | JSON 解析成功率         |
| **可读性**  | tag ≤ 3 words, lower-snake | 冗长句子                                 | 平均 token/label ≤ 3 |

---

## 2 Prompt 设计：让 LLM 只做“填空”

```
System:
You are an intent-labelling assistant. Think silently, then output JSON.

User:
Summaries are short and may contain several intents.  
Rules for tags:  
- English, lower_snake_case, 1-3 words each  
- Use existing tags exactly if possible (list below)  
- Separate multiple tags with a semicolon “;”.  
- Output ONLY a JSON object: {"id": <int>, "label": "<tag1>[;<tag2>...]"}  

Existing tags: [refund_request, delivery_delay, …]     ← 可选白名单  
###  
{id}  {summary text}  
###
```

**关键点**

1. **极强格式**：告诉模型 *只* 输出那一个 JSON。
2. **显式规则**：语言、大小写、分隔符、最少-最多词数。
3. **可选白名单**：给出当前已批准的标签池 → 模型倾向重用而不是发明新词。
4. **隐藏思考**：“Think silently…” 防止 Chain-of-Thought 泄漏到输出。

---

## 3 确定性设置

| 参数                             | 值                           | 原因                         |
| ------------------------------ | --------------------------- | -------------------------- |
| `temperature`                  | **0.0**                     | 最大限度减少随机拼写差异               |
| `top_p`                        | 0.9 或默认                     | 与 temperature=0 搭配无影响，可留默认 |
| `presence / frequency penalty` | 0                           | 避免刻意去重                     |
| **模型**                         | GPT-3.5-Turbo / GPT-4o-mini | 4k 上下文已足够；3.5 足够准确+便宜      |

---

## 4 调用与校验流程

```
FOR each summary IN file:
    1. call LLM(prompt(summary))
    2. TRY json.loads(output)
          ASSERT keys == {"id","label"}
          ASSERT output["id"] == expected id
    3. split label by ";"  → raw_tags
    4. IF any step fails:
          retry same prompt (max 2)
          else log to manual_review.jsonl
```

### 为什么行数恒等能保完整性

* 一行进去一行出；失败会立即抛错并重试→绝不会“悄悄漏掉”一条摘要。

---

## 5 提升标签一致性的四个技巧

| 手段              | 做法                                                    | 作用                   |
| --------------- | ----------------------------------------------------- | -------------------- |
| **[[白名单软约束]]**  | 在 Prompt 给“Existing tags”                             | 90 %+ 命中常用词，长尾仍可自由发挥 |
| **Few-shot 样例** | 在规则后放 3-5 行“示例 → 期望标签”                                | 让模型学格式 + 词汇粒度        |
| **即时相似词提醒**     | 对新标签向量最近邻搜索，若与已存在相似度>0.9，追加到 Prompt：“Did you mean X?” | 降低单复数、词序差异           |
| **后验正则化**       | 用 `re.sub` 统一连字符、空格 → snake\_case                     | 再次收敛拼写               |

---

## 6 多标签的准确性

> 单条摘要往往包含多个意图（“Cancel my order **and** get a refund”）。

1. **Prompt 强调**：“If multiple intents are present, list **all**; max 3.”
2. **示例展示**：

   ```
   42  Please cancel my order and issue a refund.
   → {"id":42, "label":"order_cancellation;refund_request"}
   ```
3. **后处理阈值**（可选）：

   * 若模型经常只给 1 个标签，可在 Prompt 加“common secondary intents: payment\_issue, account\_help…”。

---

## 7 质量监控与持续改进

| 指标                      | 自动化脚本                | 触发动作                   |
| ----------------------- | -------------------- | ---------------------- |
| **新标签增长率 / 周**          | 统计 `set(raw_tags)`   | > 10 % 说明写法漂移→扩白名单或加示例 |
| **解析失败率**               | (# 解析异常) / 总调用       | > 1 % 排查示例与格式是否冲突      |
| **欧几里得距离**（摘要嵌入 ↔ 标签嵌入） | cosine(tag, summary) | < 0.2 可能误标→人工抽检        |

---

## 8 为什么 “一条摘要一次调用” 仍划算？

* **Token 开销**：Prompt ≈ 100 tok + Summary ≤ 30 tok + Output ≤ 10 tok → <150 tok / 调用。
  15 000 条 ×150 tok ≈ 2.25 M tok → GPT-3.5 成本 < \$1.
* **并发**：512 并发线程下几分钟跑完，无长上下文拖慢吞吐。
* **稳定性**：单行输入彻底避免“模型在长列表里数错行”的老问题。

---

## 9 思维核⼼

1. **把难题拆小**：一句一句让模型标，而不是让它做“群分类”。
2. **先收集“词料”**：此阶段不动同义性、不管层次。标签先“粗放生长”。
3. **格式与规则双保险**：Prompt 定死 + 解析验证 → **0 %** 概率漏 ID。
4. **静态词汇 + 样例教化**：引导模型复用已存在写法，实现输出“收敛”。
5. **后处理再治理**：真正的同义折叠、层级聚类放到下一阶段，用算法做，可追溯、可复跑。

理解了这些原则，就能在任何语言（此处是英文）和任何 LLM 引擎上，**高质量、零漏失、低成本** 地生成行级意图标签，为后续 Canonical 化与 Taxonomy 抽象打下稳固地基。
