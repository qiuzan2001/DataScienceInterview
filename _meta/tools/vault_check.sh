#!/usr/bin/env bash
# vault_check.sh — Obsidian vault 只读体检
# 用法: bash _meta/tools/vault_check.sh
# 检查: 空文件 / 悬空 wikilink / AI 对话残留 / 缺 frontmatter / 双井号标题 / 转义残留 / 编号跳号
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

MD=(-name '*.md' -not -path './.git/*' -not -path './.obsidian/*')
SRC=(-not -path './.git/*' -not -path './.obsidian/*' -not -path './_meta/*')

echo "########## vault 体检 @ $(date +%F' '%T) ##########"
echo "笔记数: $(find . "${MD[@]}" -not -path './_meta/*' | wc -l | tr -d ' ')  |  总行数: $(find . "${MD[@]}" -not -path './_meta/*' -exec cat {} + | wc -l | tr -d ' ')"

echo
echo "== 1. 空文件 =="
find . "${MD[@]}" -empty | sed 's/^/  /' || true

echo
echo "== 2. 悬空 wikilink =="
find . \( -name '*.md' -o -name '*.canvas' \) "${SRC[@]}" -print0 \
  | xargs -0 grep -hoE '\[\[[^]|#]+' \
  | sed -e 's/^\[\[//' -e 's/\\*$//' -e 's/ *$//' \
        -e 's/\.png$//' -e 's/\.jpg$//' -e 's/\.jpeg$//' -e 's/\.canvas$//' \
  | sort -u > "$TMP/targets.txt"

find . \( -name '*.md' -o -name '*.png' -o -name '*.jpg' -o -name '*.canvas' \) "${SRC[@]}" -print0 \
  | xargs -0 -n1 basename | sed 's/\.[^.]*$//' | sort -u > "$TMP/names.txt"

comm -23 "$TMP/targets.txt" "$TMP/names.txt" | sed -e 's/^/  悬空: [[/' -e 's/$/]]/'

echo
echo "== 3. AI 对话残留 =="
grep -rnE "Let me know if|Great idea|Absolutely —|Here is a detailed|Here's a detailed|I've assumed you know|Got it —|I hope this helps" \
  --include='*.md' . 2>/dev/null | grep -v '^\./_meta/' | sed 's/^/  /'

echo
echo "== 4. 缺 frontmatter =="
find . "${MD[@]}" -not -path './_meta/*' -print0 | while IFS= read -r -d '' f; do
  head -1 "$f" | grep -q '^---$' || echo "  $f"
done

echo
echo "== 5. 双井号标题 (=# ## x) =="
grep -rn '^#\+ ## ' --include='*.md' . 2>/dev/null | grep -v '^\./_meta/' | sed 's/^/  /'

echo
echo "== 6. 转义残留 (\$ \lambda) =="
grep -rn '\\\$' --include='*.md' . 2>/dev/null | grep -v '^\./_meta/' | sed 's/^/  /'

echo
echo "== 7. 编号跳号 / 同章文件数 =="
find . "${MD[@]}" -not -path './_meta/*' -print0 | xargs -0 -n1 basename \
  | grep -oE '^[0-9]+\.[0-9]+' | sort -u -t. -k1,1n -k2,2n > "$TMP/lvl2.txt"
awk -F'[.]' '
  function report(c,m,s,   i,miss){ miss="";
    for(i=1;i<=m;i++) if(!(i in s)) miss = miss (miss==""?"":", ") c"."i;
    if(miss!="") printf "  第%s章 缺: %s\n", c, miss }
  { ch=$1+0; k=$2+0;
    if(ch!=prev){ if(prev!="") report(prev,maxk,seen); prev=ch; maxk=0; delete seen }
    seen[k]=1; if(k>maxk) maxk=k }
  END{ if(prev!="") report(prev,maxk,seen) }' "$TMP/lvl2.txt"
find . "${MD[@]}" -not -path './_meta/*' -print0 | xargs -0 -n1 basename \
  | grep -oE '^[0-9]+\.' | sort | uniq -c | awk '{ sub(/\.$/,"",$2); printf "  第%s章 %s 个文件\n", $2, $1 }'

echo
echo "== 8. 统计 =="
printf "  悬空链接: %s\n" "$(comm -23 "$TMP/targets.txt" "$TMP/names.txt" | wc -l | tr -d ' ')"
printf "  AI 残留:  %s\n" "$(grep -rlE "Let me know if|Great idea|Absolutely —|Here is a detailed|I've assumed you know|Got it —" --include='*.md' . 2>/dev/null | grep -vc '^\./_meta/')"
echo "########## 完 ##########"
