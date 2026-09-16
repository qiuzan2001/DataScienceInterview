#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计：每篇笔记的语言构成（中文字符 vs 英文词）。

判定：去掉代码块、frontmatter、wikilink、URL、表格分隔线后，
统计 CJK 字符数与英文单词数，算「中文占比」。
"""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SKIP_DIRS = {"_meta", ".obsidian", ".git", "Images"}

CJK = re.compile(r"[\u4e00-\u9fff]")
WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def strip_noise(t: str) -> str:
    # frontmatter
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end > 0:
            t = t[end + 4:]
    # 代码块
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    # wikilink 目标（保留显示名）
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r" \1 ", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r" \1 ", t)
    # URL / HTML 注释 / 表格分隔
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"^\s*\|?[\s:\-|]+\|?\s*$", " ", t, flags=re.M)
    # LaTeX
    t = re.sub(r"\$\$?.*?\$\$?", " ", t, flags=re.S)
    return t


def analyze(p: pathlib.Path):
    raw = p.read_text(encoding="utf-8", errors="ignore")
    t = strip_noise(raw)
    cjk = len(CJK.findall(t))
    words = len(WORD.findall(t))
    # 英文词折算成「等效中文字符」：1 英文词 ≈ 1.6 个汉字的信息量
    denom = cjk + words * 1.6
    if denom < 80:          # 太短/纯表格，跳过
        return None
    return cjk, words, cjk / denom


def main():
    rows = []
    for p in ROOT.rglob("*.md"):
        if any(s in p.parts for s in SKIP_DIRS):
            continue
        r = analyze(p)
        if r:
            rows.append((r[2], r[0], r[1], str(p.relative_to(ROOT))))
    rows.sort()

    bands = {"❌ 全英文/近全英文 (<10%)": [], "🟠 英文为主 (10–35%)": [],
             "🟡 中英混合 (35–60%)": [], "✅ 中文为主 (>60%)": []}
    for ratio, cjk, w, rel in rows:
        if ratio < 0.10:
            k = "❌ 全英文/近全英文 (<10%)"
        elif ratio < 0.35:
            k = "🟠 英文为主 (10–35%)"
        elif ratio < 0.60:
            k = "🟡 中英混合 (35–60%)"
        else:
            k = "✅ 中文为主 (>60%)"
        bands[k].append((ratio, cjk, w, rel))

    print(f"共审计 {len(rows)} 篇\n")
    for k in ["❌ 全英文/近全英文 (<10%)", "🟠 英文为主 (10–35%)",
              "🟡 中英混合 (35–60%)", "✅ 中文为主 (>60%)"]:
        v = bands[k]
        print(f"{'='*76}\n{k}  —— {len(v)} 篇\n{'='*76}")
        for ratio, cjk, w, rel in v:
            print(f"  {ratio*100:5.1f}%  中{cjk:>5} 英{w:>5}   {rel}")
        print()


if __name__ == "__main__":
    main()
