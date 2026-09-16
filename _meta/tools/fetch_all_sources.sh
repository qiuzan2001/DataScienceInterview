#!/usr/bin/env bash
# fetch_all_sources.sh — 抓取 定义原文库.md 引用过的全部来源页面到 _meta/sources/
#
# 用途：让 `python3 _meta/tools/verify_quotes.py`（引文逐字校验）可复现。
# 注意：_meta/sources/ 已在 .gitignore 中——教材原文不入库（体积 + 版权）。
#
# 用法: bash _meta/tools/fetch_all_sources.sh
set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/_meta/sources"
FETCH="$ROOT/_meta/tools/fetch_source.py"
mkdir -p "$OUT"
cd "$OUT" || exit 1

PSL="https://liangfgithub.github.io/PSL"
LDA="https://openacttexts.github.io/Loss-Data-Analytics"

# 需要缓存页面的清单与对应的输出名
PAGES=(
  "$PSL/w1/w1_1_type_of_learning.html|w1_1_type_of_learning.html.txt"
  "$PSL/w1/w1_2_intro_LS_kNN.html|w1_2_intro_LS_kNN.html.txt"
  "$PSL/w1/w1_2_discussion.html|w1_2_discussion.html.txt"
  "$PSL/w2/w2_1.html|w2_1.html.txt"
  "$PSL/w2/w2_2.html|w2_2.html.txt"
  "$PSL/w2/w2_3.html|w2_3.html.txt"
  "$PSL/w3/w3_1_subset.html|w3_1_subset.html.txt"
  "$PSL/w3/w3_2_regularize.html|w3_2_regularize.html.txt"
  "$PSL/w3/w3_3_ridge.html|w3_3_ridge.html.txt"
  "$PSL/w3/w3_4_lasso.html|w3_4_lasso.html.txt"
  "$PSL/w3/w3_5_discussion.html|w3_5_discussion.html.txt"
  "$PSL/w4/w4_1_reg_tree.html|w4_1_reg_tree.html.txt"
  "$PSL/w4/w4_2_randomforest.html|w4_2_randomforest.html.txt"
  "$PSL/w4/w4_3_gbm.html|w4_3_gbm.html.txt"
  "$PSL/w5/w5_1_poly.html|w5_1_poly.html.txt"
  "$PSL/w5/w5_2_spline.html|w5_2_spline.html.txt"
  "$PSL/w5/w5_4_smoothing_spline.html|w5_4_smoothing_spline.html.txt"
  "$PSL/w9/w9_1_intro_classification.html|w9_1_intro_classification.html.txt"
  "$PSL/w9/w9_2_DA.html|w9_2_DA.html.txt"
  "$PSL/w9/w9_3_QDA.html|w9_3_QDA.html.txt"
  "$PSL/w9/w9_4_LDA.html|w9_4_LDA.html.txt"
  "$PSL/w9/w9_6_NB.html|w9_6_NB.html.txt"
  "$PSL/w10/w10_1_setup.html|w10_1_setup.html.txt"
  "$PSL/w10/w10_2_mle.html|w10_2_mle.html.txt"
  "$PSL/w10/w10_3_seperable.html|w10_3_seperable.html.txt"
  "$PSL/w10/w10_5_sampling.html|w10_5_sampling.html.txt"
  "$PSL/w12/w12_2_measures.html|w12_2_measures.html.txt"
  "$PSL/w12/w12_3_compare.html|w12_3_compare.html.txt"
  "$PSL/w12/w12_4_aAaboost.html|w12_4_aAaboost.html.txt"
  "$PSL/w12/w12_5_boost.html|w12_5_boost.html.txt"
  "$LDA/ChapRiskClass.html|ChapRiskClass.html.txt"
  "$LDA/ChapAggLossModels.html|ChapAggLossModels.html.txt"
  "$LDA/ChapSeverity.html|ChapSeverity.html.txt"
  "$LDA/ChapFrequency-Modeling.html|ChapFrequency-Modeling.html.txt"
)

fail=0
for pair in "${PAGES[@]}"; do
  url="${pair%%|*}"; out="${pair##*|}"
  if [ -s "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    printf "  skip   %s\n" "$out"; continue
  fi
  if python3 "$FETCH" "$url" 0 >/dev/null 2>&1; then
    # fetch_source.py 会写到 "_<basename>.txt"，改名到目标名
    gen="_$(basename "${url%.html}").html.txt"
    [ -f "$gen" ] && mv -f "$gen" "$out"
    printf "  ok     %-40s %s bytes\n" "$out" "$(wc -c < "$out" 2>/dev/null | tr -d ' ')"
  else
    printf "  FAIL   %s\n" "$url"; fail=$((fail+1))
  fi
done

# ESL 全本 PDF（1-SE 原则等引文的出处）
ESL_PDF="${ESL_PDF:-$HOME/Downloads/ESLII_print12_toc.pdf}"
if [ -f "$ESL_PDF" ]; then
  if [ ! -s esl_full.txt ]; then
    if command -v pdftotext >/dev/null 2>&1; then
      pdftotext "$ESL_PDF" esl_full.txt 2>/dev/null && echo "  ok     esl_full.txt ($(wc -c < esl_full.txt | tr -d ' ') bytes)"
    else
      echo "  skip   esl_full.txt（需要 pdftotext；macOS: brew install poppler）"
    fi
  else
    echo "  skip   esl_full.txt"
  fi
else
  echo "  note   ESL PDF 未找到（可用 ESL_PDF=<路径> 指定）。ESL 引文将无法校验，其余不受影响。"
fi

echo
echo "完成。失败 $fail 项。现在可以运行："
echo "  python3 _meta/tools/verify_quotes.py"
