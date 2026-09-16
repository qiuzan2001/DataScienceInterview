#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""引文保真度校验：验证 定义原文库.md 里的英文引文是否真的逐字出现在教材原文中。

做法：把所有被抓取的教材 .txt 拼成一个大字符串，做「去空白 + 归一化标点」后
逐条检查引文是否为该大字符串的子串。非子串 = 可能被改写 / 或来自未缓存的页面。
"""
import pathlib, re, sys, json

VAULT = pathlib.Path(__file__).resolve().parent.parent.parent
DOC = VAULT / "定义原文库.md"
SRC_DIR = pathlib.Path(__file__).resolve().parent.parent / "sources"


def norm(s: str) -> str:
    """归一化：去 LaTeX 定界符、统一引号/破折号、去掉全部空白与多余标点。"""
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    s = re.sub(r"\\[\[\]\(\)]", " ", s)      # \[ \] \( \)
    s = re.sub(r"\$+", " ", s)               # $ $$
    s = s.replace("\\", " ")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)         # 只留字母数字
    return s


def load_corpus() -> str:
    parts = []
    for p in sorted(SRC_DIR.glob("*.txt")):
        if p.name.startswith(("audit_", "vault_", "old_", "new_", "release")):
            continue
        try:
            parts.append(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return norm("\n".join(parts))


def extract_quotes(text: str):
    """返回 [(entry, quote), ...]：每个 ### 小节里的英文 blockquote。"""
    entries = re.split(r"\n### ", text)
    out = []
    for e in entries[1:]:
        title = e.split("\n", 1)[0].strip()
        # blockquote 行
        for m in re.finditer(r"^>\s*(.+)$", e, re.M):
            q = m.group(1).strip()
            if q.startswith("[!"):          # callout 标记
                continue
            letters = len(re.findall(r"[A-Za-z]", q))
            # 只要「英文句」：字母数 ≥ 40 且含空格
            if letters >= 40 and " " in q:
                out.append((title, q))
    return out


def main():
    corpus = load_corpus()
    text = DOC.read_text(encoding="utf-8")
    quotes = extract_quotes(text)
    print(f"共提取英文引文 {len(quotes)} 条（来自 {len(set(t for t,_ in quotes))} 个小节）\n")

    ok, bad = [], []
    for title, q in quotes:
        nq = norm(q)
        if len(nq) < 25:
            continue
        if nq in corpus:
            ok.append((title, q))
        else:
            bad.append((title, q))

    total = len(ok) + len(bad)
    print(f"✅ 逐字命中教材原文：{len(ok)}/{total}")
    print(f"❌ 未命中（需人工核）：{len(bad)}/{total}\n")
    if bad:
        print("=" * 78)
        print("未命中的引文（可能是：改写 / 碎片化 / 跨行被拆 / 来自未缓存页面）")
        print("=" * 78)
        for title, q in bad:
            print(f"\n[{title}]\n  {q[:200]}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
