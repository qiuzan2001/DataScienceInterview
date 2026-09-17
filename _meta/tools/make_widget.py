#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""交互式知识组件生成器：spec.json → 单文件 HTML（零依赖、可离线、派生勿手改）。

架构
  源      Maps/_widgets/<笔记基名>-<slug>.json   spec v1，人/agent 可编辑
  派生    Maps/_widgets/<笔记基名>-<slug>.html   单文件自包含（内联 widgets.js / widgets.css + spec）
  清单    Maps/_widgets/Index.md                 脚本生成，勿手改
  注册表  Maps/_tools/widgets-index.json         笔记 ↔ 组件的唯一映射源

为什么 HTML 要完全内联：Obsidian 阅读视图会剥掉 <script>/<iframe>，笔记里放不了交互；
所以交互只能落在独立 HTML 文件里，由笔记正文的一个普通 Markdown 链接指向它。内联也保证
单文件可离线打开、可拷走、不依赖 CDN。

用法
    python3 Maps/_tools/make_widget.py --list                     # 渲染器目录 + 已有组件
    python3 Maps/_tools/make_widget.py --spec plot                # 某渲染器的 spec 骨架（kind|all）
    python3 Maps/_tools/make_widget.py new --note "Maps/Notes/有效久期.md" --spec /path/spec.json
    python3 Maps/_tools/make_widget.py new --note … --spec … --slug 久期凸性 --uid N1104.05 --force
    python3 Maps/_tools/make_widget.py --check                    # 注册表↔磁盘↔源笔记 一致性 + 过期检测
    python3 Maps/_tools/make_widget.py --index                    # 只重建 Maps/_widgets/Index.md
    python3 Maps/_tools/make_widget.py --layout                   # 已解析布局：一行 JSON（root / toolsDir / widgetsDir / registry / indexMd / notesIndex）
    python3 Maps/_tools/make_widget.py --root /path/to/project --list    # 指定项目根（默认按本脚本位置推导；--vault 是隐藏别名）

退出码：0 成功；1 用法或冲突错误；2 已登记但 Index.md 重建失败（不是登记失败，不要重跑登记）。
读法：批量问题清单写 stdout（带 [错误]/[提醒]），致命错误写 stderr 并带 [错误] 前缀。
布局：项目根下的默认布局是 Maps / Maps/_tools / Maps/_widgets；<root>/widgets.config.json 可选地改写
    mapsDir / toolsDir / widgetsDir / notesIndex（都是相对 root 的路径，禁止绝对路径与 ..）。
    运行时资产（widgets.js / widgets.css / vendor/）始终从本脚本目录读，与项目根无关。
边界：本工具只写 Maps/_widgets/ 与 Maps/_tools/widgets-index.json；绝不改 Books/ 原文、地图、
边界：本工具只写 Maps/_widgets/ 与 Maps/_tools/widgets-index.json；绝不改 Books/ 原文、地图、学习条目深度讲解或 notes-index.json，也不覆盖已存在的源 spec（要重建请显式 --force）。
"""
import argparse
import math
from urllib.parse import quote
import datetime
import hashlib
import json
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.dirname(HERE)
VAULT = os.path.dirname(MAPS)
WIDGETS_JS = os.path.join(HERE, 'widgets.js')
WIDGETS_CSS = os.path.join(HERE, 'widgets.css')
REGISTRY = os.path.join(HERE, 'widgets-index.json')
WIDGETS_DIR = os.path.join(MAPS, '_widgets')
INDEX_MD = os.path.join(WIDGETS_DIR, 'Index.md')
PLOTLY_BUNDLES = {
    # key → (仓库内相对路径, 说明)；spec.plotly.bundle 决定内联哪一份进页面
    'full': ('vendor/plotly.min.js', '全量包：2D + 3D + 地图 + parcoords'),
    'gl3d': ('vendor/plotly-gl3d.min.js', '裁剪包：只含 3D（surface / scatter3d / mesh3d / isosurface / volume…）'),
}
PLOTLY_BUNDLE_DEFAULT = 'full'
PLOTLY_JS = os.path.join(HERE, PLOTLY_BUNDLES['full'][0])
PLOTLY_ADAPTER = os.path.join(HERE, 'plotly-adapter.js')
PLOTLY_README = os.path.join(HERE, 'vendor', 'README.md')
PLOTLY_VENDOR_REL = 'Maps/_tools/' + PLOTLY_BUNDLES['full'][0]  # configure_layout() 会改成解析后的 toolsDir
PLOTLY_README_REL = 'Maps/_tools/vendor/README.md'              # configure_layout() 会改成解析后的 toolsDir


def plotly_bundle_key(spec):
    """spec.plotly.bundle 的取值（默认 full）。非法值由 validate_spec 拦，这里只兜底。"""
    p = spec.get('plotly') if isinstance(spec, dict) else None
    key = p.get('bundle') if isinstance(p, dict) else None
    return key if key in PLOTLY_BUNDLES else PLOTLY_BUNDLE_DEFAULT


def plotly_bundle_path(key):
    return os.path.join(HERE, PLOTLY_BUNDLES[key][0])


def plotly_bundle_rel(key):
    """返回页面元数据使用的 bundle 路径；路径跟随 widgets.config.json 的 toolsDir。"""
    return LAYOUT['toolsDir'] + '/' + PLOTLY_BUNDLES[key][0]



# 内联副本的等价转义：`="http` → `="\x68ttp`（JS 字符串里的转义，值完全不变）。
# 为什么要它：Plotly 全量包里带着地图瓦片、署名链接与图标的惰性 URL 字面量（页面 CSP 下永远不会发出请求），
# 而组件页有一条「页面里不出现 src="http / href="http」的自包含断言（test_widgets.py:82-83）。
# 字符串、模板、正则与注释里 \x68 都解析回 h，语义等价，只多 4 字节 × 31 处；上游文件本身不改（见 vendor/README.md）。
PLOTLY_ESCAPE = re.compile(r'="http')
# 内联副本把 `="http` 写成 `="\x68ttp`（一行 JS 字符串转义，值完全不变）。这里存的是**字面结果**，
# 替换时用函数形式——直接把含反斜杠的字符串当 re.subn 的替换模板会让它把 `\x` 当转义而抛
# `re.error: bad escape \x`（真踩过）。
PLOTLY_ESCAPE_TO = '="\\x68ttp'
NOTES_INDEX = os.path.join(HERE, 'notes-index.json')
# 布局可配置：项目根下的 widgets.config.json 可选地改写下面四个位置（都相对 root，见 configure_layout）。
CONFIG_NAME = 'widgets.config.json'
CONFIG_KEYS = ('mapsDir', 'toolsDir', 'widgetsDir', 'notesIndex')


def layout_state(root, config_rel, maps_dir, tools_dir, widgets_dir, notes_index):
    """已解析布局（--layout 与 --check 结尾那行布局信息的唯一来源）。

    root 是绝对路径；其余都是相对 root 的 vault 相对路径（posix 分隔符）。
    """
    return {
        'root': root,
        'configFile': config_rel,
        'mapsDir': maps_dir,
        'toolsDir': tools_dir,
        'widgetsDir': widgets_dir,
        'registry': tools_dir + '/widgets-index.json',
        'indexMd': widgets_dir + '/Index.md',
        'notesIndex': notes_index,
    }


LAYOUT = layout_state(VAULT, None, 'Maps', 'Maps/_tools', 'Maps/_widgets', 'Maps/_tools/notes-index.json')
def _refresh_layout_relatives():
    """把给页面/错误消息看的仓库相对路径同步到已解析布局。"""
    global PLOTLY_VENDOR_REL, PLOTLY_README_REL
    PLOTLY_VENDOR_REL = LAYOUT['toolsDir'] + '/' + PLOTLY_BUNDLES['full'][0]
    PLOTLY_README_REL = LAYOUT['toolsDir'] + '/vendor/README.md'


_refresh_layout_relatives()


def configure_vault(path):
    """把状态文件指到某个项目根（默认布局；不读配置文件）。运行时 JS/CSS 仍从本脚本目录读取。"""
    global VAULT, MAPS, REGISTRY, WIDGETS_DIR, INDEX_MD, NOTES_INDEX, LAYOUT
    VAULT = os.path.abspath(path)
    MAPS = os.path.join(VAULT, 'Maps')
    REGISTRY = os.path.join(MAPS, '_tools', 'widgets-index.json')
    WIDGETS_DIR = os.path.join(MAPS, '_widgets')
    INDEX_MD = os.path.join(WIDGETS_DIR, 'Index.md')
    NOTES_INDEX = os.path.join(MAPS, '_tools', 'notes-index.json')
    LAYOUT = layout_state(VAULT, None, 'Maps', 'Maps/_tools', 'Maps/_widgets', 'Maps/_tools/notes-index.json')
    _refresh_layout_relatives()


def read_layout_config(root):
    """读 <root>/widgets.config.json，返回 (配置对象, 'widgets.config.json')；没有文件时是 ({}, None)。

    不是合法 JSON 或顶层不是对象 → [错误] 退出（非零）。
    """
    path = os.path.join(root, CONFIG_NAME)
    if not os.path.exists(path):
        return {}, None
    try:
        cfg = load_json(read_text(path))
    except (ValueError, OSError) as e:
        sys.exit('[错误] %s 不是合法 JSON：%s' % (CONFIG_NAME, e))
    if not isinstance(cfg, dict):
        sys.exit('[错误] %s 顶层必须是 JSON 对象（可用键：%s）；当前是 %s'
                 % (CONFIG_NAME, ' / '.join(CONFIG_KEYS), type(cfg).__name__))
    return cfg, CONFIG_NAME


def layout_rel_value(value, key):
    """配置值 → 规范化相对路径；非字符串、空、绝对路径、含 .. 一律 [错误] 退出。"""
    if not isinstance(value, str) or not value.strip():
        sys.exit('[错误] %s 的 %s 必须是非空字符串（相对项目根的路径；当前 %r）'
                 % (CONFIG_NAME, key, value))
    rel = value.strip().replace('\\', '/')
    if rel.startswith('/') or os.path.isabs(rel) or re.match(r'^[A-Za-z]:', rel):
        sys.exit('[错误] %s 的 %s 必须是相对路径，不能是绝对路径：%r' % (CONFIG_NAME, key, value))
    parts = [p for p in rel.split('/') if p not in ('', '.')]
    if '..' in parts:
        sys.exit('[错误] %s 的 %s 不能含 ..（不许逃出项目根）：%r' % (CONFIG_NAME, key, value))
    if not parts:
        sys.exit('[错误] %s 的 %s 不能是项目根本身，要给出它下面的目录或文件：%r'
                 % (CONFIG_NAME, key, value))
    return '/'.join(parts)


def layout_inside_root(root, rel, key):
    """解析结果必须落在 root 内（与 vault_path 同口径：realpath + commonpath，能拦住 symlink）。"""
    base = os.path.realpath(root)
    target = os.path.realpath(os.path.join(base, rel))
    try:
        inside = (target == base) or (os.path.commonpath([base, target]) == base)
    except ValueError:
        inside = False
    if not inside:
        sys.exit('[错误] %s 的 %s 解析后落在项目根之外（含 symlink）：%r' % (CONFIG_NAME, key, rel))


def layout_paths(cfg, root):
    """配置 → 四个相对路径（默认链：tools/widgets 落在 mapsDir 下，notesIndex 落在 toolsDir 下）。

    未知键只 [提醒]（写 stderr，保证 --layout 的 stdout 只有一行 JSON），不失败；取值非法一律 [错误] 退出。
    """
    if not isinstance(cfg, dict):
        sys.exit('[错误] %s 顶层必须是 JSON 对象（可用键：%s）；当前是 %s'
                 % (CONFIG_NAME, ' / '.join(CONFIG_KEYS), type(cfg).__name__))
    for key in cfg:
        if key not in CONFIG_KEYS:
            print('  [提醒] %s 里有未知键 %r，会被忽略（可用键：%s）'
                  % (CONFIG_NAME, key, ' / '.join(CONFIG_KEYS)), file=sys.stderr)

    def pick(key, default):
        return layout_rel_value(cfg[key], key) if key in cfg else default

    maps = pick('mapsDir', 'Maps')
    tools = pick('toolsDir', maps + '/_tools')
    widgets = pick('widgetsDir', maps + '/_widgets')
    notes = pick('notesIndex', tools + '/notes-index.json')
    rels = {'mapsDir': maps, 'toolsDir': tools, 'widgetsDir': widgets, 'notesIndex': notes}
    for key in CONFIG_KEYS:
        layout_inside_root(root, rels[key], key)
    return rels


def configure_layout(root, cfg=None):
    """项目根 + 可选 <root>/widgets.config.json → 全部路径状态。

    先 configure_vault（默认 Maps / Maps/_tools / Maps/_widgets）再套配置覆盖；
    cfg=None 时自动读 <root>/widgets.config.json，传对象时直接用（不再读盘）。
    没有配置文件时结果与 configure_vault 完全一致，所以旧行为逐字节不变。
    """
    global MAPS, REGISTRY, WIDGETS_DIR, INDEX_MD, NOTES_INDEX, LAYOUT
    root = os.path.abspath(root)
    configure_vault(root)
    config_rel = None
    if cfg is None:
        cfg, config_rel = read_layout_config(root)
    elif os.path.exists(os.path.join(root, CONFIG_NAME)):
        config_rel = CONFIG_NAME
    rels = layout_paths(cfg, root)
    MAPS = os.path.join(root, rels['mapsDir'])
    REGISTRY = os.path.join(root, rels['toolsDir'], 'widgets-index.json')
    WIDGETS_DIR = os.path.join(root, rels['widgetsDir'])
    INDEX_MD = os.path.join(WIDGETS_DIR, 'Index.md')
    NOTES_INDEX = os.path.join(root, rels['notesIndex'])
    LAYOUT = layout_state(root, config_rel, rels['mapsDir'], rels['toolsDir'],
                          rels['widgetsDir'], rels['notesIndex'])

SCHEMA = 'widget/v1'
_refresh_layout_relatives()
RULE = '# ' + '-' * 62 + ' '

# --------------------------------------------------------------------------- #
# 渲染器目录：加一种渲染器 = 在 widgets.js 里加实现 + 在这里加一条
# 这里的 min 是给 agent 复制的最小可用骨架，字段含义见 INTERACTIVE-AUTHORING.md
# --------------------------------------------------------------------------- #
KINDS = {
    'plot': {
        'purpose': '参数滑块驱动的曲线族：让读者拖动一个量，看两条以上的曲线怎么分开',
        'use': '价格-利率曲线比较久期线性近似与凸性二阶近似；BSM 期权价对标的价或 σ；CAPM/SML 对 β',
        'require': 'kind, title, x{min,max,points}, series[].label + (expr 或 points)',
        'min': {
            'schema': SCHEMA, 'kind': 'plot', 'uid': '',
            'title': '一阶近似与二阶近似差多少',
            'subtitle': '拖动 Δy，比较只到久期与加上凸性的结果',
            'vars': {'P0': 98550, 'D': 9.0563, 'C': 114.05, 'y0': 0.071608, 'cpn': 7000, 'F': 100000, 'n': 15},
            'x': {'min': -0.02, 'max': 0.02, 'points': 121, 'label': '利率变动 Δy（小数）', 'fmt': '0.00%'},
            'controls': [
                {'key': 'dy', 'type': 'slider', 'label': '利率变动 Δy', 'min': -0.02, 'max': 0.02,
                 'step': 0.0005, 'value': -0.0025, 'fmt': '0.00%'},
            ],
            'series': [
                {'label': '只到久期（一阶）', 'expr': 'P0*(1 - D*(x + dy))', 'color': '#e0a15a', 'dash': True},
                {'label': '加凸性（二阶）', 'expr': 'P0*(1 - D*(x + dy) + 0.5*C*(x + dy)*(x + dy))', 'color': '#7fb8e6'},
                {'label': '真实重估价格', 'expr': 'cpn*(1 - pow(1+y0+x+dy, -n))/(y0+x+dy) + F*pow(1+y0+x+dy, -n)',
                 'color': '#4ec27a'},
            ],
            'markers': [{'x': 'dy', 'label': '当前 Δy', 'color': '#e0a15a'}],
            'readouts': [
                {'label': '一阶近似变化', 'expr': 'P0*(-D*dy)', 'fmt': '0,0.00'},
                {'label': '二阶近似变化', 'expr': 'P0*(-D*dy + 0.5*C*dy*dy)', 'fmt': '0,0.00'},
                {'label': '真实重估变化',
                 'expr': 'cpn*(1 - pow(1+y0+dy, -n))/(y0+dy) + F*pow(1+y0+dy, -n) - P0', 'fmt': '0,0.00'},
            ],
            'notes': ['数值取自 Maps/Notes/有效久期.md 的复算（P0=98,550、D*=9.06）；真实价格曲线与凸性 C=114.05 为按同一条件（15 年、7% 票息、面值 10 万）的数值演算。'],
        },
    },
    'bars': {
        'purpose': '可拖动参数改变条形长度：把一个"分配"或"权重"变成看得见的东西',
        'use': 'EWMA / GARCH 权重随 λ、α 变化；现金流现值分解；组合中各头寸的风险贡献',
        'require': 'kind, title, bars[].label + bars[].value（表达式，作用域含 i 与 item）',
        'min': {
            'schema': SCHEMA, 'kind': 'bars', 'uid': '',
            'title': 'λ 改变时各期权重怎么变',
            'subtitle': 'EWMA 权重 w_t ∝ (1-λ)λ^k，越靠近现在权重越大',
            'vars': {'k': 20},
            'controls': [{'key': 'lam', 'type': 'slider', 'label': '衰减因子 λ', 'min': 0.5, 'max': 0.99,
                          'step': 0.01, 'value': 0.94, 'fmt': '0.00'}],
            'bars': [
                {'label': '滞后 0', 'value': '(1-lam)*pow(lam, 0)'},
                {'label': '滞后 1', 'value': '(1-lam)*pow(lam, 1)'},
                {'label': '滞后 2', 'value': '(1-lam)*pow(lam, 2)'},
                {'label': '滞后 5', 'value': '(1-lam)*pow(lam, 5)'},
                {'label': '滞后 10', 'value': '(1-lam)*pow(lam, 10)'},
            ],
            'readouts': [{'label': '半衰期（期）', 'expr': 'log(0.5)/log(lam)', 'fmt': '0.00'}],
            'notes': ['权重之和趋于 1；λ 越大，权重衰减越慢、等价样本越长。'],
        },
    },
    'scatter': {
        'purpose': '散点与参考线：看数据有没有关系、离群点在哪',
        'use': '两资产收益散点；回归残差图；预测值 vs 实际值',
        'require': 'kind, title, series[].label + (points 或 expr)',
        'min': {
            'schema': SCHEMA, 'kind': 'scatter', 'uid': '',
            'title': '两资产收益散点',
            'vars': {},
            'x': {'min': -0.05, 'max': 0.05, 'points': 2, 'label': '资产 A 收益', 'fmt': '0.0%'},
            'series': [{'label': '样本', 'points': [[-0.02, -0.012], [-0.01, -0.005], [0.0, 0.002],
                                                    [0.01, 0.008], [0.02, 0.019], [0.03, 0.011]],
                        'color': '#7fb8e6'}],
            'readouts': [],
            'notes': ['把观测点替换成真实样本；不要为了让图好看而删离群点。'],
        },
    },
    'histogram': {
        'purpose': '抽样与分位线：把"分布"从一个词变成能看见的形状',
        'use': '蒙特卡洛 VaR / ES；t 分布厚尾对分位的影响；抽样分布与置信区间',
        'require': 'kind, title, histogram.sample（表达式，可用 randn()）或 histogram.values',
        'min': {
            'schema': SCHEMA, 'kind': 'histogram', 'uid': '',
            'title': 'σ 与自由度如何改变左尾',
            'subtitle': '每次重绘重新抽样，看分位线怎么抖',
            'vars': {'mu': 0},
            'controls': [
                {'key': 'sigma', 'type': 'slider', 'label': '波动率 σ', 'min': 0.005, 'max': 0.05,
                 'step': 0.001, 'value': 0.02, 'fmt': '0.00%'},
                {'key': 'n', 'type': 'number', 'label': '抽样次数', 'min': 500, 'max': 20000,
                 'step': 500, 'value': 5000},
            ],
            'histogram': {'sample': 'mu + sigma*randn()', 'bins': 41},
            'markers': [{'x': 'mu - 1.645*sigma', 'label': '95% 分位（正态）', 'color': '#e0a15a'}],
            'readouts': [{'label': '95% VaR（本次抽样）', 'expr': '-quantile(__samples__, 0.05)', 'fmt': '0.00%'}],
            'notes': ['σ 用小数。分位线在每次重绘时会抖动，这正是"抽样误差"，不要把它当成模型改变。'],
        },
    },
    'heatmap': {
        'purpose': '可编辑矩阵：让读者自己改相关系数，立刻看到组合风险怎么变',
        'use': '相关系数矩阵 → 组合方差；VIF 与共线；希腊字母敏感度表',
        'require': 'kind, title, heat{rows, cols, values}',
        'min': {
            'schema': SCHEMA, 'kind': 'heatmap', 'uid': '',
            'title': '两个资产的相关系数如何决定组合方差',
            'subtitle': '点击格子改相关系数，看下方组合波动率',
            'vars': {'w1': 0.5, 'w2': 0.5, 's1': 0.2, 's2': 0.3},
            'controls': [{'key': 'editable', 'type': 'toggle', 'label': '允许编辑矩阵', 'value': True}],
            'heat': {'rows': ['资产 1', '资产 2'], 'cols': ['资产 1', '资产 2'],
                     'values': [[1.0, 0.3], [0.3, 1.0]], 'fmt': '0.00', 'editable': True,
                     'symmetric': True, 'bind': {'rho12': [0, 1]}},
            'readouts': [{'label': '组合波动率', 'expr': 'sqrt(w1*w1*s1*s1 + w2*w2*s2*s2 + 2*w1*w2*s1*s2*rho12)',
                          'fmt': '0.00%'}],
            'notes': ['bind 显式把 [0,1] 格子命名为 rho12；symmetric=true 时编辑一边会镜像另一边，对角线不自动改变。'],
        },
    },
    'timeline': {
        'purpose': '事件时间轴：把"什么时候发生什么"摆平了看',
        'use': '现金流与折现；MBS 提前偿还；利率期限结构的关键期限',
        'require': 'kind, title, timeline.items[].t + .amount',
        'min': {
            'schema': SCHEMA, 'kind': 'timeline', 'uid': '',
            'title': '现金流在时间轴上的位置',
            'vars': {'cpn': 7000, 'F': 100000},
            'controls': [{'key': 'y', 'type': 'slider', 'label': '折现率 y', 'min': 0.01, 'max': 0.15,
                          'step': 0.005, 'value': 0.071608, 'fmt': '0.00%'}],
            'timeline': {'axis': 't（年）', 'items': [
                {'t': 1, 'label': '第 1 年票息现值', 'amount': 'cpn/pow(1+y,1)'},
                {'t': 5, 'label': '第 5 年票息现值', 'amount': 'cpn/pow(1+y,5)'},
                {'t': 10, 'label': '第 10 年票息现值', 'amount': 'cpn/pow(1+y,10)'},
                {'t': 15, 'label': '第 15 年票息 + 本金现值', 'amount': '(cpn+F)/pow(1+y,15)'},
            ]},
            'notes': ['条形高度是各现金流的现值，不是名义金额；拖动 y 时远端现金流下降更快。'],
        },
    },
    'tree': {
        'purpose': '可展开的分层结构：点击看下一步分叉，适合"路径"类知识',
        'use': '二项树定价；决策树与情景分析；提前偿还路径',
        'require': 'kind, title, tree.root{label, value, children}',
        'min': {
            'schema': SCHEMA, 'kind': 'tree', 'uid': '',
            'title': '一期二项树',
            'subtitle': '点击节点折叠；拖动 S0 与 u/d 看叶子价格',
            'vars': {'u': 1.1, 'd': 0.9},
            'controls': [{'key': 'S0', 'type': 'slider', 'label': '当前股价 S0', 'min': 50, 'max': 150,
                          'step': 1, 'value': 100}],
            'tree': {'root': {'label': 'S0', 'value': 'S0', 'children': [
                {'label': '上', 'prob': 0.5, 'value': 'S0*u', 'children': []},
                {'label': '下', 'prob': 0.5, 'value': 'S0*d', 'children': []},
            ]}},
            'notes': ['概率与上下幅度要来自题目或模型，不要随手填。'],
        },
    },
    'box': {
        'purpose': '箱线图：把一组样本的中位数、四分位距与异常点一次看清（Tukey 口径）',
        'use': '实验/观测数据的分布概括与异常点筛查；收益率、残差、测量值的离群点检查',
        'require': 'kind, title, box.sample（表达式，抽样 n 次）或 box.values（非空数字数组）',
        'min': {
            'schema': SCHEMA, 'kind': 'box', 'uid': '',
            'title': '一组样本的中位数、四分位距与异常点',
            'subtitle': '骨架示例：样本由 randn() 现抽；正式使用请换成源笔记里的数据并注明出处',
            'vars': {'mu': 0},          # sigma 由下面的控件给（同名会覆盖常量，校验器会提醒）
            'controls': [
                {'key': 'sigma', 'type': 'slider', 'label': '波动率 σ', 'min': 0.2, 'max': 3,
                 'step': 0.1, 'value': 1, 'fmt': '0.00'},
                {'key': 'n', 'type': 'number', 'label': '抽样次数', 'min': 10, 'max': 20000,
                 'step': 10, 'value': 500},
            ],
            'box': {'sample': 'mu + sigma*randn()', 'fmt': '0.00'},
            'readouts': [
                {'label': '中位数（本次抽样）', 'expr': 'quantile(__samples__, 0.5)', 'fmt': '0.00'},
                {'label': 'IQR = Q3 − Q1（本次抽样）',
                 'expr': 'quantile(__samples__, 0.75) - quantile(__samples__, 0.25)', 'fmt': '0.00'},
            ],
            'notes': [
                '口径：Q1/Q2/Q3 = quantile(vals, .25/.50/.75)，与内置 quantile() 相同的线性插值；须 = 1.5×IQR 以内最远的数据点，须外的点单独画成异常点（Tukey 口径）。',
                'readouts 里的 __samples__ 是本次抽样的只读副本（每次重绘重新抽样）；n<2 时四分位数不可用，按缺失处理并报 [错误]，不会画成 0 或假箱体。',
                '骨架里的 randn() 只演示抽样口径：正式使用时把样本换成源笔记里的数据，并在 subtitle/notes 写明出处。',
            ],
        },
    },
    'ecdf': {
        'purpose': '经验分布函数：不假设分布形状，直接看"不超过 x 的样本占比"这条单调阶梯',
        'use': '样本累积概率与理论分位对照（配 markers）；中位数与尾部概率；样本量对经验分布的抖动',
        'require': 'kind, title, ecdf.sample（表达式）或 ecdf.values（非空数字数组）；可选 ecdf.maxPoints',
        'min': {
            'schema': SCHEMA, 'kind': 'ecdf', 'uid': '',
            'title': '经验分布：样本的累积概率阶梯',
            'subtitle': '骨架示例：values 是手写小样本；正式使用请换成源笔记里的数据并注明出处',
            'vars': {},
            'ecdf': {'values': [1.2, 1.6, 1.7, 2.1, 2.3, 2.6, 3.4, 5.1], 'fmt': '0.0', 'maxPoints': 400},
            'markers': [{'x': 'quantile(__samples__, 0.5)', 'label': '中位数'}],
            'readouts': [
                {'label': '中位数（本页样本）', 'expr': 'quantile(__samples__, 0.5)', 'fmt': '0.0'},
                {'label': '样本量 n', 'expr': '__samples__.length', 'fmt': '0,0'},
            ],
            'notes': [
                '口径：F(x) = #{样本 ≤ x}/n，单调阶梯，起点 (min, 0)、终点 (max, 1)；位置与概率都直接来自样本，不假设任何分布。',
                'n 超过 maxPoints（默认 1200）时按等间隔抽稀：被跳过的台阶合并到下一个保留点，阶梯位置为近似，累积概率与端点精确。',
                '骨架里的样本只是占位示例；正式使用必须换成源笔记里的数据并写明出处。',
            ],
        },
    },
    'qq': {
        'purpose': 'Q-Q 图：把样本分位数对着理论分位数画，点大致落在一条直线上说明形状对得上',
        'use': '残差/收益率是否近似正态；尾部偏离（厚尾、偏斜）在两端最明显；离群点会单独甩出去',
        'require': 'kind, title, qq.sample（表达式）或 qq.values（非空数字数组）；qq.dist 目前只支持 "normal"；可选 qq.maxPoints',
        'min': {
            'schema': SCHEMA, 'kind': 'qq', 'uid': '',
            'title': '样本分位数 vs 正态分位数：直就说明形状接近正态',
            'subtitle': '骨架示例：样本由 randn() 现抽（本来就正态，所以应当接近直线）；正式使用请换成源笔记里的数据',
            'vars': {'mu': 0},          # sigma 由下面的控件给（同名会覆盖常量，校验器会提醒）
            'controls': [
                {'key': 'sigma', 'type': 'slider', 'label': '波动率 σ', 'min': 0.2, 'max': 3,
                 'step': 0.1, 'value': 1, 'fmt': '0.00'},
                {'key': 'n', 'type': 'number', 'label': '抽样次数', 'min': 10, 'max': 20000,
                 'step': 10, 'value': 800},
            ],
            'qq': {'sample': 'mu + sigma*randn()', 'dist': 'normal', 'fmt': '0.00'},
            'readouts': [
                {'label': 'Q1（本次抽样）', 'expr': 'quantile(__samples__, 0.25)', 'fmt': '0.00'},
                {'label': 'Q3（本次抽样）', 'expr': 'quantile(__samples__, 0.75)', 'fmt': '0.00'},
            ],
            'notes': [
                '口径：横轴 = 标准正态分位数 z((i−0.5)/n)（Hazen 位置，避免 p=0/1 处的 ±∞），纵轴 = 排序后的样本；参考线只过 (z(0.25), Q1) 与 (z(0.75), Q3) 两点，斜率 = IQR / (z(0.75)−z(0.25)) ≈ IQR/1.34898。',
                'z 用 Acklam 反向正态近似（全区间相对误差 < 1.15e-9；p=0.5 处精确为 0）。σ 只改变参考线斜率（尺度），不改变"直不直"。',
                'n 超过 maxPoints（默认 2000）时图上按等间隔抽稀：点仍按完整 n 计算 Hazen 位置，因此仍落在真 Q-Q 线上；参考线与四分位不受影响。',
                '骨架里的 randn() 只演示抽样口径；正式使用必须换成源笔记里的数据并写明出处。dist 目前只支持 "normal"，未知值报 [错误]。',
            ],
        },
    },
    'contour': {
        'purpose': '标量场等高线：拖动参数看 f(x,y) 的等值线怎么移动、变形',
        'use': '两个自变量的函数（组合方差对权重与相关系数、BSM 价格对 S 与 σ）；效用无差异曲线；概率密度的水平集',
        'require': 'kind, title, x{min,max[,points]} 与 y{min,max[,points]}（网格定义域），contour.expr（f(x,y) 表达式，变量是 x 与 y）',
        'min': {
            'schema': SCHEMA, 'kind': 'contour', 'uid': '',
            'title': 'f(x,y) 的等值线：同一条线上函数值相同',
            'subtitle': '骨架示例：f = x² + y²，等值线是圆；正式使用请换成源笔记里的函数、范围与出处',
            'vars': {'x0': 0.5, 'y0': 0.5},
            'x': {'min': -2, 'max': 2, 'points': 41, 'label': 'x', 'fmt': '0.0'},
            'y': {'min': -2, 'max': 2, 'points': 41, 'label': 'y', 'fmt': '0.0'},
            'aspect': 'auto',
            'contour': {'expr': 'x*x + y*y', 'levels': 4},
            'readouts': [{'label': 'f(x0, y0)（示例点 x0=0.5、y0=0.5）',
                          'expr': 'x0*x0 + y0*y0', 'fmt': '0.00'}],
            'notes': [
                '口径：等值线用 marching squares 在 x×y 网格上取；含缺失值（除零、log 负值等非有限结果）的格子整体跳过，不会画出假线，缺失点数在图注与 [错误] 里给出。',
                '层数：contour.levels=N 时取 N 条等值线，取值 z_k = zmin + k·(zmax−zmin)/(N+1)（k=1…N），全部落在 (zmin, zmax) 内部，避免在极值点退化成单点。',
                'readouts 的作用域里没有网格自变量：x 是读数下标。要在读数里算某点的 f，请把点写进 vars（如上例 x0/y0）并保持与 f 相同的表达式口径。',
                '坐标轴：默认 aspect="auto"（两轴各自独立拉伸填满图框），几何形状会被压扁——f = x² + y² 的等值线画出来是椭圆而不是圆，图注会写明；要形状保真就设 aspect="equal"。',
            ],
        },
    },
    'vector': {
        'purpose': '向量场箭头：在一个网格上画出 (u,v) 的方向与相对大小（相图、梯度场）',
        'use': '动力系统相图；梯度/最速上升方向；风险因子的联合漂移；经济学的方向场',
        'require': 'kind, title, x{min,max[,points]} 与 y{min,max[,points]}，vector.u + vector.v（两个分量表达式）；可选 vector.scale=auto|unit',
        'min': {
            'schema': SCHEMA, 'kind': 'vector', 'uid': '',
            'title': '相图：每一点的箭头是 (u, v) 的方向与相对大小',
            'subtitle': '骨架示例：u = y、v = −x（绕原点旋转的线性系统）；正式使用请换成源笔记里的模型与出处',
            'vars': {'x0': 1, 'y0': 0},
            'x': {'min': -2, 'max': 2, 'points': 21, 'label': 'x', 'fmt': '0.0'},
            'y': {'min': -2, 'max': 2, 'points': 21, 'label': 'y', 'fmt': '0.0'},
            'aspect': 'auto',
            'vector': {'u': 'y', 'v': '-x', 'scale': 'auto'},
            'readouts': [{'label': '在 (x0, y0) 处 |(u,v)|（示例）',
                          'expr': 'sqrt(y0*y0 + x0*x0)', 'fmt': '0.00'}],
            'notes': [
                '箭头锚在单元格中心，网格 (nx−1)×(ny−1) 个箭头；终点 = 起点 + k·(u,v)，即箭头是向量在数据坐标里的像（两轴单位不同时，屏幕角度会随坐标轴刻度缩放）。',
                '长度口径：auto（默认）＝数据空间长度 ∝ |(u,v)|，按全场最大模长归一化，最长箭头 = 0.9×min(单元格宽, 单元格高)；unit ＝ 所有箭头等长（数据长度 = 0.9×min(Δx, Δy)），只表示方向、不表示大小。口径写进图注。',
                '缺失与零向量：u 或 v 在某个位置不是有限数（如除零、log 负值）时该位置按缺失处理、不画箭头并报 [错误]；|(u,v)|=0 的位置不画箭头。readouts 的 x 是下标不是横轴。',
                '坐标轴：默认 aspect="auto"（两轴各自拉伸填满图框），屏幕上的箭头方向会随两轴刻度被压扁——要方向保真就设 aspect="equal"（两轴同一 px/单位），图注会写明当前用的哪一种。',
            ],
        },
    },
    'matrix': {
        'purpose': '2×2 矩阵的向量几何：单位正方形与基向量怎么被拉、压、转，det 就是面积缩放倍数',
        'use': '线性变换与行列式；特征方向与特征值（含复数）；协方差矩阵、相关矩阵的几何直观',
        'require': 'kind, title, matrix.values（2×2 数字矩阵 [[a,b],[c,d]]）；可选 matrix.editable / matrix.samples / matrix.bind（默认把格子绑成表达式变量 a/b/c/d）',
        'min': {
            'schema': SCHEMA, 'kind': 'matrix', 'uid': '',
            'title': '2×2 矩阵：行列式是面积缩放倍数，特征方向不变',
            'subtitle': '骨架示例 A = [[2,1],[1,2]]（det=3、特征值 3 与 1）；点击格子可改，正式使用请注明出处',
            'vars': {},
            'aspect': 'equal',
            'matrix': {'values': [[2, 1], [1, 2]], 'editable': True, 'samples': [[1, 0.5]]},
            'readouts': [
                {'label': 'det = 面积缩放倍数', 'expr': 'a*d - b*c', 'fmt': '0.00'},
                {'label': '迹 tr = a + d', 'expr': 'a + d', 'fmt': '0.00'},
                {'label': '判别式 Δ = tr² − 4det', 'expr': '(a+d)*(a+d) - 4*(a*d - b*c)', 'fmt': '0.00'},
                {'label': 'λ₁（Δ ≥ 0 时才有实数解）',
                 'expr': 'ifelse((a+d)*(a+d) - 4*(a*d-b*c) >= 0, ((a+d) + sqrt((a+d)*(a+d) - 4*(a*d-b*c)))/2, NaN)',
                 'fmt': '0.00'},
            ],
            'notes': [
                '口径：A = [a b; c d]（第 1 行 a、b，第 2 行 c、d），格子自动绑定为表达式变量 a/b/c/d（可用 matrix.bind 改名）；第 j 列 = A·e_j。',
                'det = ad − bc 是单位正方形变换后的有向面积：|det| = 面积缩放倍数；det < 0 表示方向翻转；det = 0 时矩阵奇异（面积缩到 0，不可逆）。',
                '特征值 λ = (tr ± √(tr²−4det))/2（tr = a+d）：Δ ≥ 0 给两个实特征值并画出特征方向（A·v = λv：方向不变、长度乘 λ）；Δ < 0 时如实标为复数 p ± qi，虚部不省略，且实平面上没有不变方向、不画特征向量。',
                'samples 是可选的样本点：空心点 = 原来的点，实心点 = 变换后的像；数字位数默认按数量级取，也可用 matrix.fmt 固定（如 "0.00"）。',
                '坐标轴：默认 aspect="equal"（两轴同一 px/单位）——单位正方形在屏幕上仍是正方形，det = 面积缩放倍数与"特征方向不变"才看得出来；显式写 aspect="auto" 会退回两轴独立拉伸，形状被压扁（图注会提醒这一点）。',
            ],
        },
    },
    'regression': {
        'purpose': '散点 + 拟合直线：改斜率/截距或换一批样本，看残差竖线与 R² 怎么变',
        'use': '最小二乘回归（CAPM 的市场模型 β、久期回归、因子暴露）；手动线对照 OLS；残差结构与 R²/RMSE 的口径',
        'require': 'kind, title, points（[[x, y], …]）或 data{n, x, y[, seed]}；可选 fit{mode: "ols"|"manual"|"compare"|"none"（默认 ols）、slopeKey、interceptKey、slope、intercept、residuals}；x/y 只用来定范围与刻度格式',
        'min': {
            'schema': SCHEMA, 'kind': 'regression', 'uid': '',
            'title': 'OLS 直线：竖线是残差，R² 是残差平方和占变动量的比例',
            'subtitle': '骨架示例：5 个手写点；正式使用必须换成源笔记里的数据并写明出处',
            'vars': {},
            'x': {'min': 0, 'max': 4, 'fmt': '0.0'},
            'y': {'fmt': '0.0'},
            'controls': [
                {'key': 'slope', 'type': 'slider', 'label': '手动线斜率', 'min': -1, 'max': 4,
                 'step': 0.05, 'value': 2, 'animate': {'from': 0, 'to': 3, 'seconds': 8}},
                {'key': 'intercept', 'type': 'slider', 'label': '手动线截距', 'min': -2, 'max': 4,
                 'step': 0.05, 'value': 0.5},
            ],
            'points': [[0, 1], [1, 3], [2, 4], [3, 7], [3.6, 8.2]],
            'fit': {'mode': 'compare'},
            'readouts': [
                {'label': 'OLS 斜率', 'expr': '__reg__.slope', 'fmt': '0.0000'},
                {'label': 'OLS 截距', 'expr': '__reg__.intercept', 'fmt': '0.0000'},
                {'label': 'R²', 'expr': '__reg__.r2', 'fmt': '0.0000'},
                {'label': '手动线 R²', 'expr': '__reg__.mR2', 'fmt': '0.0000'},
            ],
            'notes': [
                '口径：普通最小二乘（让 Σ(y − ŷ)² 最小）；斜率 = sxy/sxx、截距 = ȳ − 斜率·x̄；R² = 1 − SS_res/SS_tot（SS_tot 以 ȳ 为基准）；RMSE = √(SS_res/n)（除以 n）。',
                '所有 x 相同（sxx = 0）时斜率没有定义，如实给 NaN、不硬凑 0；有效点少于 2 个时只画散点并报 [错误]。',
                'fit.mode：ols（默认，只画最小二乘线）、manual（只画手动线）、compare（两条都画，读出比较）、none（只画散点）。手动线读控件（默认 key 是 slope / intercept，可用 fit.slopeKey / interceptKey 改名），也可用 fit.slope / fit.intercept 写常量。',
                'data 逐点求值的作用域有 i（行号，0 起）、n、rand()、randn()；rand/randn 按 data.seed（默认 12345）播种，所以拖别的控件不会重抽样本，同 seed 可复现。x 算完会写回作用域，y 里可以直接引用同一个 x（如 y: "0.8*x + 2*randn()"）。',
                '骨架里的 5 个点只是占位示例；正式使用必须换成源笔记里的数据并在 notes 里写明出处。',
            ],
        },
    },
    'pca': {
        'purpose': '主成分方向：转一条候选轴，看它上面的方差什么时候最大（正好等于 PC1）',
        'use': 'PCA / 特征分解的几何含义；正交化与降维保留多少方差；相关矩阵的主轴方向',
        'require': 'kind, title, points（[[x, y], …]）或 data{n, x, y[, seed]}；可选 meanCenter（默认 true）、candidate{angle、angleControl、ellipse、showProjection}；坐标轴默认 aspect="equal"',
        'min': {
            'schema': SCHEMA, 'kind': 'pca', 'uid': '',
            'title': 'PC1 就是方差最大的方向：转候选轴找最大值',
            'subtitle': '骨架示例：8 个沿对角线散开的点；正式使用必须换成源笔记里的数据并写明出处',
            'vars': {},
            'x': {'fmt': '0.0'},
            'y': {'fmt': '0.0'},
            'controls': [
                {'key': 'theta', 'type': 'slider', 'label': '候选轴方向 θ（度）', 'min': 0, 'max': 180,
                 'step': 1, 'value': 0, 'animate': {'from': 0, 'to': 180, 'seconds': 10}},
            ],
            'points': [[-2, -1.8], [-1.4, -1.5], [-1, -0.7], [-0.4, -0.5],
                       [0.3, 0.4], [0.9, 0.6], [1.5, 1.7], [2, 1.9]],
            'candidate': {'ellipse': True, 'showProjection': True},
            'readouts': [
                {'label': 'λ₁（PC1 上的方差）', 'expr': '__pca__.l1', 'fmt': '0.000'},
                {'label': 'λ₂', 'expr': '__pca__.l2', 'fmt': '0.000'},
                {'label': 'PC1 方向角（度）', 'expr': '__pca__.angle1', 'fmt': '0.0'},
                {'label': '候选轴方差 / 总方差', 'expr': '__pca__.candRatio', 'fmt': '0.000'},
            ],
            'notes': [
                '口径：协方差用无偏样本口径（除以 n − 1）；λ₁ ≥ λ₂ 是 2×2 协方差矩阵的两个特征值，PC1 方向角 θ₁ = ½·atan2(2·sxy, sxx − syy)，单位是度、落在 [0, 180)；解释方差比 = λ₁/(λ₁+λ₂)。',
                '候选轴由 θ（度）给出，θ 与 θ + 180 是同一条轴；「候选轴方差 / 总方差」在 θ = θ₁ 时取到最大值，正好等于 λ₁/(λ₁+λ₂)——这就是"主轴 = 方差最大方向"。',
                '画的是 1σ 椭圆（半轴 = √λ，即各主方向上的标准差），不是置信域；投影线 = 样本点到候选轴的垂足。candidate.ellipse / candidate.showProjection 写 false 可以关掉。',
                'meanCenter 默认 true（先平移到均值再算协方差）；写 false 时算的是绕原点的二阶矩，PC1 会被均值方向拉偏——图注会写明当前用的是哪一种。',
                '坐标轴默认 aspect="equal"（两轴同一 px/单位）：主轴方向、正交关系与点到轴的距离都要几何保真才读得对。',
            ],
        },
    },
    'descent': {
        'purpose': '梯度下降的轨迹：在损失曲面的等高线上跑迭代，改学习率 α 看它是走稳、来回震荡还是发散',
        'use': '梯度下降的步长选择；凸与非凸函数的下降路径；学习率过大的发散；点图框换起点',
        'require': 'kind, title, x{min,max[,points]} 与 y{min,max[,points]}（定义域），contour.expr（f(x, y) = 损失函数），控件 lr 与 steps；可选 grad{dfdx, dfdy}、descent{lrKey, stepsKey, clickToSetStart}、start[x, y]；坐标轴默认 aspect="equal"',
        'min': {
            'schema': SCHEMA, 'kind': 'descent', 'uid': '',
            'title': '梯度下降：α 太大就会越走越远',
            'subtitle': '骨架示例：f = x² + 4y²（正定二次型）；正式使用必须换成源笔记里的目标函数并写明出处',
            'vars': {},
            'x': {'min': -3, 'max': 3, 'points': 61, 'label': 'x', 'fmt': '0.0'},
            'y': {'min': -3, 'max': 3, 'points': 61, 'label': 'y', 'fmt': '0.0'},
            'aspect': 'equal',
            'controls': [
                {'key': 'lr', 'type': 'slider', 'label': '学习率 α', 'min': 0.01, 'max': 0.6, 'step': 0.01,
                 'value': 0.1, 'fmt': '0.00', 'animate': {'from': 0.02, 'to': 0.55, 'seconds': 10}},
                {'key': 'steps', 'type': 'number', 'label': '迭代步数', 'min': 1, 'max': 60, 'step': 1, 'value': 24},
            ],
            'contour': {'expr': 'x*x + 4*y*y', 'levels': 8},
            'grad': {'dfdx': '2*x', 'dfdy': '8*y'},
            'start': [-3, 2],
            'descent': {'lrKey': 'lr', 'stepsKey': 'steps', 'clickToSetStart': True},
            'readouts': [
                {'label': '起点 f(p₀)', 'expr': '__gd__.f0', 'fmt': '0.000'},
                {'label': '末点 f(p_k)', 'expr': '__gd__.f', 'fmt': '0.000'},
                {'label': '一共降了多少', 'expr': '__gd__.lost', 'fmt': '0.000'},
                {'label': '实际走的步数', 'expr': '__gd__.done', 'fmt': '0'},
            ],
            'notes': [
                '迭代口径：p_{k+1} = p_k − α·∇f(p_k)，从起点的定义域内格点出发（spec.start 可指定，也可以在图上点一下换起点），最多走 steps 步。',
                '梯度：spec.grad.dfdx / dfdy 给了就用解析式，否则用中心差分（h = 定义域宽/1000）——图注会写明用的哪一种，两种的轨迹在数值上会有微小差别。',
                '越界与发散：某一步走出定义域就停在那一步并记 escaped；模长超过定义域对角线 3 倍记 diverged（只画到最后一个有限点）。两种情况都在 readouts 与图注里如实标出，不悄悄把点画到框外。',
                'α 为正但过小时收敛很慢（轨迹贴着等高线爬），α 超过 2/λmax 时开始来回振荡、更大就发散——这正是"学习率上限由曲率决定"的直观来源。',
                '骨架的 f = x² + 4y² 是口径演示；正式使用必须换成源笔记里的损失函数，并在 notes 里写明条件与出处。',
            ],
        },
    },
    'surface3d': {
        'purpose': '三维曲面：拖着看 z = f(x, y) 的形状，方位角与仰角都是控件',
        'use': '两个自变量的函数形状（损失曲面、效用面、联合分布密度）；三维散点云；带一条三维轨迹的寻优路径',
        'require': 'kind, title, x/y 定义域，surface{expr 或 points, mode: "surface"|"wireframe"|"points"}；也可用 adaboost 或 boosting 动态块（二选一数据源）；控件 azimuth 与 elevation（可选 zoom）；可选 spec.path（三维轨迹 [[x, y, z], …]）',
        'min': {
            'schema': SCHEMA, 'kind': 'surface3d', 'uid': '',
            'title': 'z = sin(x)·cos(y)：转个角度看懂曲面的起伏',
            'subtitle': '骨架示例：解析曲面（正式使用请换成源笔记里的函数或数据并写明出处）',
            'vars': {},
            'x': {'min': -3, 'max': 3, 'points': 25, 'label': 'x', 'fmt': '0.0'},
            'y': {'min': -3, 'max': 3, 'points': 25, 'label': 'y', 'fmt': '0.0'},
            'controls': [
                {'key': 'azimuth', 'type': 'slider', 'label': '方位角', 'min': 0, 'max': 355, 'step': 5,
                 'value': 35, 'fmt': '0', 'animate': {'from': 0, 'to': 355, 'seconds': 12}},
                {'key': 'elevation', 'type': 'slider', 'label': '仰角（0° 平视，90° 俯视）', 'min': 0, 'max': 89,
                 'step': 1, 'value': 25, 'fmt': '0'},
                {'key': 'zoom', 'type': 'slider', 'label': '缩放', 'min': 0.5, 'max': 1.8, 'step': 0.05,
                 'value': 1, 'fmt': '0.00'},
            ],
            'surface': {'expr': 'sin(x)*cos(y)', 'mode': 'surface'},
            'view': {'azimuthKey': 'azimuth', 'elevationKey': 'elevation', 'zoomKey': 'zoom'},
            'readouts': [
                {'label': 'z 最小值', 'expr': '__s3__.zmin', 'fmt': '0.000'},
                {'label': 'z 最大值', 'expr': '__s3__.zmax', 'fmt': '0.000'},
                {'label': '画出的面片数', 'expr': '__s3__.cells', 'fmt': '0'},
            ],
            'notes': [
                '投影口径：正交投影（不是透视）。u = x·cos az − y·sin az 是屏幕横向，w = x·sin az + y·cos az；v = cos φ·w + sin φ·z 是屏幕纵向（φ = 90° − 仰角），深度 d = cos φ·z − sin φ·w 用来排序与上色浓淡。u 与 v 用同一个像素尺度，所以曲面不会被拉扁。',
                '仰角 0° = 平视（z 竖直向上），90° = 正上方俯视；方位角 0°–355° 绕 z 轴转。也可以在图上按住拖动旋转（拖过之后以拖动的视角为准），控件缺失时才用 spec.view.azimuth / elevation 的默认值。',
                '面片按平均深度从远到近画（画家算法），不做遮挡剔除——极端视角下远处的面可能压住近处的面，这是已知的近似，不影响"形状对不对"的判断。轨迹（spec.path）画在最上层，方便看清。',
                'surface.points 给 [[x, y, z], …] 点云时忽略 expr（图注会写明）；surface.mode 可选 surface（默认，实心面）、wireframe（只画网格线，看清拓扑）、points（只画点云）。',
                'markers 在这里必须带 z（{x, y, z, label}）：二维的竖线标记在三维里没有意义。',
                '动态算法曲面：adaboost 或 boosting 作为顶层二选一数据块时，由 verifier 按块内递推重算主图；不能用旧的 surface.expr 冒充。动态块仍要提供合法 x/y、notes 与 teaching，块内引用的 semantic 控件 key 必须来自 controls。',
            ],
        },
    },
    'treefit': {
        'purpose': '决策树的轴对齐切分：把 depth 拖深，看叶子越切越细、训练误差一路降',
        'use': '回归树/分类树的划分与叶值；过拟合的来源；特征切分点怎么选；bagging/boosting 的基学习器',
        'require': 'kind, title, mode "1d"|"2d"（默认 1d），points 或 data{n, x, y[, cls][, seed]}，treefit.splits（1d: [{at}]；2d: [{axis, at}]），控件 depth',
        'min': {
            'schema': SCHEMA, 'kind': 'treefit', 'uid': '',
            'title': '回归树切深一点：训练 MSE 一路降，这就是过拟合的来源',
            'subtitle': '骨架示例：9 个手写点与 2 个切分点；正式使用必须换成源笔记里的数据并写明出处',
            'vars': {},
            'x': {'min': 0, 'max': 4, 'fmt': '0.0'},
            'y': {'fmt': '0.0'},
            'controls': [
                {'key': 'depth', 'type': 'slider', 'label': '树深（用几个切分点）', 'min': 0, 'max': 2,
                 'step': 1, 'value': 1, 'fmt': '0', 'animate': {'from': 0, 'to': 2, 'seconds': 8}},
            ],
            'mode': '1d',
            'treefit': {'mode': '1d', 'splits': [{'at': 2}, {'at': 3.5}]},
            'points': [[0, 1], [0.5, 1.1], [1, 1], [1.5, 1.2], [2, 3],
                       [2.5, 3.1], [3, 3], [3.5, 4.9], [4, 5]],
            'readouts': [
                {'label': '训练 MSE', 'expr': '__tree__.mse', 'fmt': '0.000'},
                {'label': 'y 的方差（作对照）', 'expr': '__tree__.varY', 'fmt': '0.000'},
                {'label': '叶子数', 'expr': '__tree__.leaves', 'fmt': '0'},
            ],
            'notes': [
                '口径：树不在这里学习——切分点由 spec 给出（splits 按"加深一层"的顺序，depth 决定用前几个）；叶子的取值/类别由当前划分从数据里现算。这样拖 depth 看到的是"同一棵树越深越细"，可复算、可写进笔记。',
                'mode "1d"：切分点是 x 上的阈值；叶子值 = 区间内 y 的均值，预测是阶梯函数；训练 MSE = mean((y − 预测)²)。区间边界约定左闭右开、最后一段闭右端，归类与评估共用同一套边界规则。',
                'mode "2d"：splits 写 [{axis: "x"|"y", at: 阈值}]，按顺序作用在"包含该点"的区域上；叶子类别 = 区域内出现最多的类别，训练错误率 = 被分错的点占比；数据要带 cls（类别）。',
                '读法：depth 从 0 加到满，训练误差单调降（最极端时每个叶子只剩一个点、训练误差 0），而泛化误差不跟着降——这就是过拟合的形状。空叶子（没有点落进去的区间）会如实标出。',
                '骨架的 9 个点与 2 个切分点只是占位示例；正式使用必须换成源笔记里的数据并在 notes 里写明出处。',
            ],
        },
    },
    'custom': {
        'purpose': '逃生口：既有渲染器都不合适时，自带一段 HTML/JS',
        'use': '仅在确实无法用 plot / bars / scatter / histogram / heatmap / timeline / tree / box / ecdf / qq / contour / vector / matrix / regression / pca / descent / surface3d / treefit 表达时使用；必须在 notes 里说明为什么必须自定义',
        'require': 'kind, title, custom.html 和/或 custom.js',
        'min': {
            'schema': SCHEMA, 'kind': 'custom', 'uid': '',
            'title': '自定义组件',
            'vars': {},
            'custom': {
                'html': '<div class="wg-custom-note">把要拖的东西放在这里</div>',
                'js': "var box = root.querySelector('.wg-custom-note');\n"
                      "box.textContent = 'fmt 示例：' + api.fmt(1.2345, '0.00');",
            },
            'notes': ['自定义组件不享受内置渲染器的保障：请自行确保数值有出处、窄面板不破版、无外部依赖。'],
        },
    },
    'plotly': {
        'purpose': '手写 SVG 渲染器做不到的图：真 WebGL 三维（可平滑旋转/缩放）、蜡烛图/K 线、地图、10⁵ 以上的点',
        'use': '三维曲面或点云要真实光照与流畅旋转；K 线 + 成交量；地图上的点线；十万级散点（内置 SVG 渲染器一个点一个 DOM 节点，画不动）。能用 surface3d / contour / vector / scatter / heatmap 说清的一律别用它',
        'require': 'kind, title, plotly.data（非空数组）；其余全部写在 plotly.* 里',
        'min': {
            'schema': SCHEMA, 'kind': 'plotly', 'uid': '',
            'title': 'Plotly 三维曲面：z = a·x² + y²',
            'subtitle': '拖动曲率 a，看碗口怎么收紧（网格 25×25）',
            'vars': {'n': 25},
            'controls': [
                {'key': 'a', 'type': 'slider', 'label': '曲率 a', 'min': 0.2, 'max': 3,
                 'step': 0.1, 'value': 1},
            ],
            'plotly': {
                'data': [{
                    'type': 'surface',
                    # 生成器：本文件（plotly-adapter.js）在 JS 侧逐元素求值，作用域里 i/j/n/m 与 x=i、y=j 可用。
                    # 为什么不用表达式里的循环：引擎没有对象字面量/for/Array.from（见 plotly-adapter.js 的说明）。
                    'x': {'by': '-2 + 4 * i / (n - 1)', 'n': '=n'},
                    'y': {'by': '-2 + 4 * i / (n - 1)', 'n': '=n'},
                    'z': {'by': 'a * xx * xx + yy * yy', 'rows': '=n', 'cols': '=n',
                          'vars': {'xx': '-2 + 4 * j / (n - 1)', 'yy': '-2 + 4 * i / (n - 1)'}},
                    'colorscale': 'Blues', 'showscale': False,
                }],
                'layout': {'scene': {'aspectmode': 'cube'}},
                'config': {'displayModeBar': False},
            },
            'readouts': [
                {'label': '网格点数', 'expr': '__plotly__.points', 'fmt': '0'},
                {'label': '画布高度', 'expr': '__plotly__.height', 'fmt': '0 px'},
            ],
            'notes': [
                '骨架/演示：z = a·x² + y² 是本节现算的示例曲面（非任何教材算例），拖 a 看曲率对形状的影响。',
                '数据两种写法：① 直接写数组（Plotly 要什么写什么）；② 生成器 '
                '{"by": "<表达式>", "n": 25}（一维）或 {"by": "…", "rows": 25, "cols": 25}（二维网格）——'
                '适配器逐元素求值，作用域里有一维的 i/n（x = i）、二维的 i/j/n/m（x = i、y = j），'
                'n/rows/cols 也可以写 "=表达式"，还能带 "vars": {"名字": "表达式"} 在同一作用域里先定义名字'
                '（把"范围常量只写一遍"放进 spec.vars，多个生成器共用）。上限：一维 ≤ 300 个元素、'
                '二维 ≤ 40000 格。表达式引擎没有对象字面量 / for / Array.from，所以"按控件生成一列数"'
                '只能用生成器。',
                '正式使用请把 plotly.data 换成源笔记里的数据或原文公式，并在 notes 里写明出处；数据量大时'
                '（>10⁵ 点）只有 plotly 这条路径画得动。',
            ],
        },
        'notes': [
            '代价（按 plotly.bundle 选）：`"full"`（默认）内联 Maps/_tools/vendor/plotly.min.js，4.85 MB，'
            'gzip 后约 1.4 MB，2D + 3D + 地图 + parcoords 什么都能画；`"gl3d"` 内联 '
            'vendor/plotly-gl3d.min.js，1.69 MB，**只含 3D**（surface / scatter3d / mesh3d / isosurface / '
            'volume / cone / streamtube），写 2D trace 全校验阶段就提醒你。只有 kind="plotly" 的页面带库，'
            '其他页面零负担；--check 会打印每页「页面 X MB，其中内联 Plotly Y MB（bundle=…）」。',
            '选型：只在手写 SVG 渲染器真的做不到时才用——真 WebGL 三维（surface3d 是正交投影手绘，转不动、没有光照）、'
            '蜡烛图/K 线、地图、10⁵ 以上的点。能用 surface3d / contour / vector / scatter / heatmap 表达的，一律用内置渲染器'
            '（选型见 INTERACTIVE-AUTHORING.md；内核规范仍以它为准）。',
            '库是本地 vendored + 内联（没有 CDN、没有 fetch、没有 import）：来源 / 版本 / sha256 / 许可（MIT）见 '
            'Maps/_tools/vendor/README.md；--check 会校验该 sha256，并为 plotly 页打印「页面 X MB，其中 Plotly Y MB」。',
            '`=` 约定：plotly.data / plotly.layout 里**以 = 开头**的字符串在渲染时按表达式求值（去掉前导 = 后交给运行时引擎，'
            '作用域 = vars 的键 + controls 的 key + 助手函数；整条求值一次，不是逐点作用域），求值结果写回同一位置，'
            '所以控件能驱动数据、颜色、坐标轴范围；不以 = 开头的字符串原样传给 Plotly。字符串里是同一套表达式引擎'
            '（`^` 与 `**` 都是乘方、右结合，报错精确到列号），乘方写 x^2、pow(x, 2) 或 x*x 都行。',
            '渲染口径：固定白底（只有 spec.theme="system" 且系统是深色时才读组件主题取深色）；高度必须显式给，'
            '默认 clamp(图区宽 × 0.72, 260, 520) px，可用 plotly.height 覆盖；宽度交给 Plotly（不写 layout.width），'
            '窗口/容器变宽由 widgets.js 的 ResizeObserver/resize 重绘接管；layout.title 被忽略（标题由页面 DOM 承担）；'
            '重绘走 Plotly.react 复用同一个图，容器被换掉时先 purge。'
            '读数：it.vars.__plotly__ = {traces, points, height, width, mode}（冻结）。traces/points/height/width 是数字，'
            '可以直接进 readouts；mode 是 "2d" / "3d" / "mixed" 字符串，而读数框只格式化数字（字符串会显示成 —），'
            '要看 mode 请写进图注或说明文字。points 的口径：每个 trace 取 x/y/z 等数组里最大的长度（二维网格按 rows×cols 计）再求和。',
            '限制（不放宽 CSP，失败就如实报 [错误]）：parcoords 一类需要 blob: worker 的特性不可用；'
            '地图类 trace 要联网取瓦片、不可用；宿主没有 WebGL 时 3D trace 画不出来（会提示）。',
            '演示页面：--spec plotly 打印的最小骨架本身就是能跑的演示（25×25 曲面 + 一个曲率滑杆）；'
            '仓库里已登记的实例见 [[交互组件示例-Plotly三维.md]] 与 Maps/_widgets/ 下的 plotly 页。',
        ],
    },
}
def _teaching_skeleton(kind, title):
    """给 --spec 的最小样例补齐质量门禁元数据（正式 spec 必须改成真实内容）。"""
    teaching = {
        'question': title,
        'sourceSection': '待填写：源笔记中的章节或段落定位',
        'controlEffect': '待填写：每个 semantic 控件如何改变主绘制数据',
        'visualEvidence': '待填写：主图中随控件变化、可直接观察的证据',
    }
    # 这些控件改变的是运行时生成的数据量/切分过程，静态字段没有可写入的 expr；
    # 动态门禁会逐值比较主图，不能只靠这项声明通过。
    if kind in ('histogram', 'box', 'qq', 'treefit'):
        teaching['dynamicRenderer'] = True
    # editable 只切换编辑界面，不是教学图形参数；显式空数组避免把它误当 semantic。
    if kind == 'heatmap':
        teaching['controls'] = []
    return teaching


for _kind, _info in KINDS.items():
    _info['min'].setdefault('teaching', _teaching_skeleton(_kind, _info['min'].get('title', _kind)))


ASPECT_KINDS = ('contour', 'vector', 'matrix', 'pca', 'descent')
# aspect（等比例坐标轴）只对上面几个渲染器有意义；默认值按渲染器区分：matrix / pca / descent 用 "equal"
# （det = 面积缩放倍数、特征方向不变、点到主轴的距离、最陡方向这些结论依赖几何形状保真），
# contour / vector 保持 "auto"（两轴各自拉伸填满图框，向后兼容）。spec.aspect 可显式覆盖，非法值在校验阶段报 [错误]。
ASPECT_DEFAULT = {'contour': 'auto', 'vector': 'auto', 'matrix': 'equal', 'pca': 'equal', 'descent': 'equal'}
KIND_ORDER = ['plot', 'bars', 'scatter', 'histogram', 'heatmap', 'timeline', 'tree',
               'box', 'ecdf', 'qq', 'contour', 'vector', 'matrix',
               'regression', 'pca', 'descent', 'surface3d', 'treefit', 'custom',
               # plotly 放最后：它是唯一「独立文件 + 内联 4.85 MB 库」的外部渲染器
               # （widgets.js 里 WG.registerKind('plotly', …)，不是内置分派），列在末尾表示 opt-in
               'plotly']

# plotly 页只读 plotly.*：这些是别的渲染器的块或速记，写了也不会生效（渲染器不读它们），
# 一律在校验阶段拦下，避免"写了却没生效"的静默困惑。aspect / renderer 有自己的提醒，不重复列。
PLOTLY_FOREIGN_BLOCKS = ('series', 'bars', 'histogram', 'box', 'ecdf', 'qq', 'contour', 'vector',
                         'matrix', 'heat', 'timeline', 'tree', 'surface', 'treefit', 'custom',
                         'fit', 'candidate', 'grad', 'descent', 'start', 'data', 'points', 'path',
                         'view', 'markers', 'x', 'y', 'mode')

POINTS_WARN_CANVAS = 100000  # 写了 renderer="canvas" 时的提醒线（canvas 上 10 万点 ≈ 24ms/帧）
POINTS_MAX_CANVAS = 200000   # canvas 的硬上限（再大就是 spec JSON 本身大到不合理了）
# 数据规模预算：SVG 是「一个 mark 一个 DOM 节点」，实测（Chromium，见 Maps/_tools/INTERACTIVE-AUTHORING.md §11）
# scatter 1 点 ≈ 2 节点、heatmap 1 格 ≈ 1 节点、surface3d 1 面片 ≈ 1 节点；
# 5k 点 ≈ 46ms、10k 点 ≈ 93ms、50k 点 ≈ 455ms（10 万节点）——拖滑杆要每帧重绘，
# 所以「能画」和「能交互」之间有硬门槛。这里给数据量设上限，别让 spec 悄悄生成 10 万节点：
#   超过 POINTS_MAX + 抽稀建议 = 硬错误（校验阶段就拦住）；超过 POINTS_WARN 但能画 = 提醒。
# 抽稀口径可复用 ecdf / qq 的 maxPoints（等间隔保留、位置近似、端点精确）。
POINTS_MAX = 20000        # series[].points / 顶层 points 的元素个数上限
POINTS_WARN = 5000        # 同样口径的提醒线（≈46ms/帧，再大就影响滑杆手感）
HEAT_CELLS_MAX = 40000    # heat.rows × heat.cols（200×200）上限
HEAT_CELLS_WARN = 10000   # 100×100 以上提醒
DATA_N_MAX = 2000         # data{n} 逐点表达式的样本量上限（与运行时的 clamp 一致）
# spec.renderer 只对这三个「mark 密集」的渲染器有意义（其余一律 SVG，写了会被忽略）
RENDERER_KINDS = ('scatter', 'heatmap', 'surface3d')

# 渲染器写进 it.vars 的只读状态（供 readouts / markers 引用），--spec 里列给作者看
INTERNAL_VARS = {
    'box': '__samples__（本次样本的冻结副本：n / 四分位 / 须 / 异常点都从这里算）',
    'ecdf': '__samples__（本次样本的冻结副本）',
    'qq': '__samples__（本次样本的冻结副本）',
    'regression': '__reg__：n, mx, my, slope, intercept, r2, rmse, ssRes, varY；compare/manual 时另加 mSlope, mIntercept, mR2, mRmse',
    'pca': '__pca__：n, mx, my, l1, l2, ratio, angle1, totalVar, candAngle, candVar, candRatio, candShare, meanCentered',
    'descent': '__gd__：steps, lr, done, escaped, diverged, x, y, f, f0, gnorm, lost, grad（analytic|numeric）',
    'surface3d': '__s3__：az, el, zoom, zmin, zmax, zRange, cells, pathPoints, bad',
    'histogram': '__samples__（本次抽样的冻结副本；柱高是频数，bins 决定分组）',
    'treefit': '__tree__：mode, depth, maxDepth, leaves, emptyLeaves, mse, rmse, varY, n（2d 另有 regions, err, wrong, classes）',
    'plotly': '__plotly__：traces, points, height, width, mode（前四个是数字，可直接进 readouts；mode 是 "2d"/"3d"/"mixed" 字符串）',
}

# control 的 key 与这些保留名重名会悄悄遮住辅助函数，所以校验时只提醒、不拦截
RESERVED = ('x', 'i', 'item', '__samples__', '__reg__', '__pca__', '__gd__', '__s3__', '__tree__',
            '__plotly__',
            'rand', 'randn', 'phi', 'quantile', 'sum', 'mean', 'sd', 'clamp',
            'min', 'max', 'abs', 'exp', 'log', 'sqrt', 'pow', 'sin', 'cos', 'tan', 'floor',
            'round', 'PI', 'E', 'ifelse', 'fmt', 'true', 'false', 'null', 'undefined')

REGISTRY_NOTE = ('交互组件注册表：笔记 ↔ 交互组件的唯一映射。由 make_widget.py 维护，不要手改。'
                 'json 是源 spec（可编辑），html 是派生产物（重建即覆盖），两者是否同步由 --check 校验。')


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def today():
    return datetime.date.today().isoformat()


def file_sha(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write_bytes_atomic(path, data):
    """Single-file atomic replacement; stage beside destination for same-filesystem rename."""
    path = vault_path(rel_vault(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.widget-', dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_text(path, text):
    write_bytes_atomic(path, text.encode('utf-8'))


def write_transaction(files):
    """Rollback caught write failures. Not crash-atomic across files; no concurrent writers.

    Index is deliberately outside this transaction (exit 2 after registration).
    Rollback failure is reported explicitly; power loss requires --check/rebuild.
    """
    backups = {}
    for path, _ in files:
        vault_path(rel_vault(path))
        backups[path] = None
        if os.path.exists(path):
            with open(path, 'rb') as f:
                backups[path] = f.read()
    done = []
    try:
        for path, text in files:
            write_text(path, text)
            done.append(path)
    except (OSError, ValueError) as error:
        failures = []
        for path in reversed(done):
            try:
                if backups[path] is None:
                    os.unlink(path)
                else:
                    write_bytes_atomic(path, backups[path])
            except OSError as rollback_error:
                failures.append('%s: %s' % (path, rollback_error))
        if failures:
            raise OSError('%s；回滚失败：%s' % (error, '; '.join(failures))) from error
        raise


def finite_json(value):
    if isinstance(value, (int, float)):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ValueError('JSON 不允许 NaN/Infinity 或溢出数值')
    if isinstance(value, dict):
        for item in value.values():
            finite_json(item)
    elif isinstance(value, list):
        for item in value:
            finite_json(item)
    return value


def load_json(text):
    return finite_json(json.loads(text))


def json_for_script(obj):
    """把 spec 塞进 <script> 里：转义 < > & 与 U+2028/2029，防 </script> 截断（同 build_map.py 的做法）。"""
    s = json.dumps(finite_json(obj), ensure_ascii=False, allow_nan=False)
    return (s.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
             .replace('\u2028', '\\u2028').replace('\u2029', '\\u2029'))


def attr(s):
    return (str(s).replace('&', '&amp;').replace('"', '&quot;')
            .replace('<', '&lt;').replace('>', '&gt;'))
# --------------------------------------------------------------------------- #
# Plotly（opt-in 后端）的构建侧辅助
#   只有 kind="plotly" 的页面才内联 vendor/plotly.min.js + plotly-adapter.js；
def plotly_vendor(bundle_key=None):
    """读内联用的 Plotly 发行包，返回 (内联文本, 上游 sha256, 等价转义次数, 版本串, 仓库内相对路径)。

    bundle_key 取 PLOTLY_BUNDLES 的键（默认 full）。文件缺失时返回 (None, None, 0, None, rel)：
    构建照常进行（页面里没有库，打开会显示 [错误]），validate_spec 已经就缺少库给过 [提醒]。
    """
    key = bundle_key if bundle_key in PLOTLY_BUNDLES else PLOTLY_BUNDLE_DEFAULT
    path = plotly_bundle_path(key)
    rel = plotly_bundle_rel(key)
    if not os.path.exists(path):
        return None, None, 0, None, rel
    text = read_text(path)
    text, escapes = PLOTLY_ESCAPE.subn(lambda m: PLOTLY_ESCAPE_TO, text)   # 见 PLOTLY_ESCAPE 的说明
    if re.search(r'</script', text, re.I):
        text = re.sub(r'</script', '<\\/script', text, flags=re.I)
    # 版本串两种写法：full 是 `plotly.js v3.7.0`，gl3d 裁剪包是 `plotly.js (gl3d - minified) v3.7.0`
    # （都在文件头注释里），所以中间允许任意非换行字符
    m = re.search(r'plotly\.js[^\n]{0,40}? v(\d+\.\d+\.\d+)', text)
    return text, file_sha(path), escapes, (m.group(1) if m else None), rel


def vendor_sha_from_readme(bundle_key=None):
    """从 vendor/README.md 的机器可读块里解析出该 bundle 登记的 sha256；解析不到返回 None。

    块的样子（每个 bundle 一块）：
        file: vendor/plotly.min.js
        version: 3.7.0
        bytes: 4851164
        sha256: <64 位十六进制>
    按 `file:` 匹配；只有一块且没写 file 时退回第一个 sha256（老格式）。
    """
    if not os.path.exists(PLOTLY_README):
        return None
    text = read_text(PLOTLY_README)
    key = bundle_key if bundle_key in PLOTLY_BUNDLES else PLOTLY_BUNDLE_DEFAULT
    rel = PLOTLY_BUNDLES[key][0]
    blocks = re.findall(r'file:\s*(\S+)\s+version:\s*(\S+)\s+bytes:\s*(\d+)\s+sha256:\s*([0-9a-fA-F]{64})', text)
    for f, _ver, _bytes, sha in blocks:
        if f.strip().lstrip('./') == rel:
            return sha.lower()
    if len(blocks) == 1:
        return blocks[0][3].lower()
    m = re.search(r'sha256[^0-9a-fA-F]{0,24}([0-9a-fA-F]{64})', text)
    return m.group(1).lower() if m else None


def plotly_meta_tags(vendor):
    """页面上的 vendor 标记：证明这一页内联了哪一份库（上游 sha256 + 内联副本 sha256 + 转义处数）。"""
    text, sha, escapes, version, rel = vendor
    tags = ['<meta name="wg-vendor" content="plotly.js">',
            '<meta name="wg-vendor-path" content="%s">' % attr(rel)]
    if text is None:
        tags.append('<meta name="wg-vendor-missing" content="%s：文件不存在，本页没有内联的 Plotly">'
                    % attr(rel))
    else:
        tags.append('<meta name="wg-vendor-version" content="%s">' % attr(version or '未定位'))
        tags.append('<meta name="wg-vendor-sha256" content="%s">' % sha)
        tags.append('<meta name="wg-vendor-inline-sha256" content="%s">' % sha_text(text))
        tags.append('<meta name="wg-vendor-inline-escapes" content="%d">' % escapes)
    return '\n'.join(tags) + '\n'


def size_text(n):
    """体积报告用的粗刻度（够看清"页面里库占了多少"就行）。"""
    mb = n / 1048576.0
    return ('%.2f MB' % mb) if mb >= 1 else ('%.0f KB' % (n / 1024.0))


def slugify(s):
    """文件名片段：保留中文，只把路径分隔符、空白与 #[] 压成连字符。"""
    s = re.sub(r'[\s/\\:*?"<>|#\[\]]+', '-', str(s).strip())
    s = re.sub(r'-{2,}', '-', s).strip('-')
    return (s or 'widget')[:40]


def note_base(note):
    b = os.path.basename(note)
    return b[:-3] if b.endswith('.md') else b
def rel_vault(path):
    return os.path.relpath(path, VAULT).replace(os.sep, '/')




def is_vault_rel(rel):
    try:
        vault_path(rel)
        return True
    except (ValueError, OSError):
        return False


def vault_path(rel):
    if not isinstance(rel, str) or not rel or '\\' in rel or '\x00' in rel:
        raise ValueError('非法 vault 相对路径：%r' % rel)
    if os.path.isabs(rel) or '..' in rel.split('/'):
        raise ValueError('路径逃出 vault：%r' % rel)
    root = os.path.realpath(VAULT)
    target = os.path.realpath(os.path.join(root, rel))
    if os.path.commonpath([root, target]) != root:
        raise ValueError('路径逃出 vault（含 symlink）：%r' % rel)
    books = os.path.realpath(os.path.join(root, 'Books'))
    if rel.split('/')[0].lower() == 'books' or os.path.commonpath([books, target]) == books:
        raise ValueError('不能挂到或写入 Books/ 原文：%r' % rel)
    return target


def load_registry():
    if not os.path.exists(REGISTRY):
        return {'vault': os.path.basename(VAULT), '_说明': REGISTRY_NOTE, 'widgets': []}
    try:
        vault_path(rel_vault(REGISTRY))
        reg = load_json(read_text(REGISTRY))
    except (ValueError, OSError) as e:
        sys.exit('[错误] %s 不是合法 JSON：%s' % (rel_vault(REGISTRY), e))
    if not isinstance(reg, dict) or not isinstance(reg.get('widgets'), list):
        sys.exit('[错误] %s 结构不对：应为 {"widgets": [...]}' % rel_vault(REGISTRY))
    seen = {'json': set(), 'html': set()}
    for w in reg['widgets']:
        if not isinstance(w, dict):
            sys.exit('[错误] 注册项必须是对象')
        for key in ('note', 'json', 'html'):
            try:
                resolved = vault_path(w.get(key))
                if key != 'note':
                    base = os.path.realpath(WIDGETS_DIR)
                    if os.path.commonpath([base, resolved]) != base or not resolved.endswith('.' + key):
                        raise ValueError('组件路径必须在 Maps/_widgets/ 且扩展名正确')
                    if resolved in seen[key]:
                        raise ValueError('registry 重复 %s：%s' % (key, w[key]))
                    seen[key].add(resolved)
            except (ValueError, OSError) as e:
                sys.exit('[错误] registry %s：%s' % (key, e))
        for key in ('uid', 'kind', 'title'):
            if not isinstance(w.get(key), str):
                sys.exit('[错误] registry %s 必须是字符串' % key)
    reg.setdefault('vault', os.path.basename(VAULT))
    reg.setdefault('_说明', REGISTRY_NOTE)
    return reg


def save_registry(reg):
    """按 (笔记, 源文件) 排序后落盘：让 diff 只反映真实变化，而不是字典顺序。"""
    reg['widgets'] = sorted(reg['widgets'], key=lambda w: (w.get('note', ''), w.get('json', '')))
    write_text(REGISTRY, json.dumps(finite_json(reg), ensure_ascii=False, indent=1, allow_nan=False) + '\n')


def note_uid(note):
    """从 notes-index.json 反查笔记的 uid；查不到返回 ''（不算错误：组件允许挂在未登记笔记上）。"""
    if not os.path.exists(NOTES_INDEX):
        return ''
    try:
        data = load_json(read_text(vault_path(rel_vault(NOTES_INDEX))))
    except (ValueError, OSError) as e:
        sys.exit('[错误] notes-index.json：%s' % e)
    for n in data.get('notes', []):
        if (n.get('file') or '').replace('\\', '/') == note:
            return n.get('uid') or ''
    return ''


def known_uids():
    if not os.path.exists(NOTES_INDEX):
        return set()
    try:
        data = load_json(read_text(vault_path(rel_vault(NOTES_INDEX))))
    except (ValueError, OSError) as e:
        sys.exit('[错误] notes-index.json：%s' % e)
    return set(n.get('uid') for n in data.get('notes', []) if n.get('uid'))


# --------------------------------------------------------------------------- #
# spec 校验：返回 (errors, warns)，由调用方决定打印方式与退出码
# --------------------------------------------------------------------------- #
def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _valid_name(k):
    return isinstance(k, str) and bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k))


def _sample_source(spec, kind, name, err):
    """box / ecdf / qq 共用：sample（表达式）或 values（非空数字数组）二选一；返回该子对象。"""
    node = spec.get(name)
    if not isinstance(node, dict):
        err.append('%s 必须给 %s{sample 或 values}' % (kind, name))
        return None
    ok_sample = isinstance(node.get('sample'), str) and node['sample'].strip()
    ok_values = isinstance(node.get('values'), list) and bool(node['values'])
    if not (ok_sample or ok_values):
        err.append('%s 必须给 %s.sample（表达式，可用 randn()）或 %s.values（非空数字数组）'
                   % (kind, name, name))
    return node


def _max_points(node, where, err):
    """ecdf / qq 共用：maxPoints 是抽稀上限（绘制点数上界），不是样本量。"""
    if isinstance(node, dict) and node.get('maxPoints') is not None and not (
            isinstance(node['maxPoints'], int) and not isinstance(node['maxPoints'], bool)
            and 2 <= node['maxPoints'] <= 5000):
        err.append('%s.maxPoints 若给出，须是 2–5000 的整数（绘制点数上限，超出则抽稀）' % where)


def _field_axis(node, kind, name, err):
    """contour / vector 共用的网格轴校验：min/max 必填；points 可选（3–121，上限防拖动时重算过大网格）。"""
    if not isinstance(node, dict):
        err.append('%s 必须给 spec.%s{min, max[, points]}（两轴定义域都要有）' % (kind, name))
        return
    for f in ('min', 'max'):
        if not _num(node.get(f)):
            err.append('%s.%s 必须是数字' % (name, f))
    if _num(node.get('min')) and _num(node.get('max')) and node['min'] >= node['max']:
        err.append('%s：min 必须小于 max' % name)
    pts = node.get('points')
    if pts is not None and not (isinstance(pts, int) and not isinstance(pts, bool) and 3 <= pts <= 121):
        err.append('%s.points 若给出，须是 3–121 的整数（网格点数上限 121；省略时 contour 默认 41、vector 默认 21）' % name)


SEMANTIC_CONTROL_TYPES = ('slider', 'number', 'toggle', 'select')
CAMERA_CONTROL_KEYS = frozenset(('azimuth', 'elevation', 'zoom'))
TEACHING_REQUIRED = ('question', 'sourceSection', 'controlEffect', 'visualEvidence')


def _collect_strings(value, out):
    """收集字段值里的字符串，不收集对象键；用于主绘制字段的静态依赖扫描。"""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, out)
    elif isinstance(value, list):
        for item in value:
            _collect_strings(item, out)


def _main_draw_strings(spec, kind):
    """返回该 renderer 真正用来画主图的数据字段文本。

    readouts / markers / title / subtitle / notes 有意不在这里：它们即使引用了
    控件，也不能证明控件改变了教学图形。
    """
    out = []

    def add(value):
        _collect_strings(value, out)

    def add_fields(holder, fields):
        if isinstance(holder, dict):
            for field in fields:
                if field in holder:
                    add(holder[field])

    def add_series():
        if isinstance(spec.get('series'), list):
            for item in spec['series']:
                add_fields(item, ('expr', 'points'))

    if kind in ('plot', 'scatter'):
        add_series()
    elif kind == 'bars':
        for item in spec.get('bars') or []:
            add_fields(item, ('value',))
    elif kind == 'histogram':
        add_fields(spec.get('histogram'), ('sample', 'values'))
    elif kind in ('box', 'ecdf', 'qq'):
        add_fields(spec.get(kind), ('sample', 'values'))
    elif kind == 'heatmap':
        add_fields(spec.get('heat'), ('values',))
    elif kind == 'timeline':
        timeline = spec.get('timeline')
        if isinstance(timeline, dict):
            for item in timeline.get('items') or []:
                add_fields(item, ('t', 'amount'))
    elif kind == 'tree':
        def add_tree(node):
            if not isinstance(node, dict):
                return
            add_fields(node, ('value', 't', 'amount'))
            for child in node.get('children') or []:
                add_tree(child)
        tree = spec.get('tree')
        add_tree(tree.get('root') if isinstance(tree, dict) else None)
    elif kind == 'vector':
        add_fields(spec.get('vector'), ('u', 'v'))
    elif kind == 'matrix':
        add_fields(spec.get('matrix'), ('values',))
    elif kind == 'regression':
        add(spec.get('points'))
        add_fields(spec.get('data'), ('x', 'y', 'cls'))
        fit = spec.get('fit')
        if isinstance(fit, dict) and fit.get('mode') in ('manual', 'compare'):
            add_fields(fit, ('slope', 'intercept', 'slopeKey', 'interceptKey'))
            if 'slopeKey' not in fit:
                out.append('slope')
            if 'interceptKey' not in fit:
                out.append('intercept')
    elif kind == 'pca':
        add(spec.get('points'))
        add_fields(spec.get('data'), ('x', 'y', 'cls'))
        candidate = spec.get('candidate')
        if isinstance(candidate, dict):
            add_fields(candidate, ('angle', 'angleControl'))
            if 'angleControl' not in candidate:
                out.append('theta')
    elif kind == 'descent':
        add_fields(spec.get('contour'), ('expr',))
        add_fields(spec.get('grad'), ('dfdx', 'dfdy'))
        descent = spec.get('descent')
        if isinstance(descent, dict):
            add_fields(descent, ('lr', 'steps', 'lrKey', 'stepsKey'))
            if 'lrKey' not in descent:
                out.append('lr')
            if 'stepsKey' not in descent:
                out.append('steps')
    elif kind == 'contour':
        add_fields(spec.get('contour'), ('expr',))
    elif kind == 'surface3d':
        add_fields(spec.get('surface'), ('expr', 'points'))
        add(spec.get('path'))
        # 动态 surface3d 由 renderer 读取这些 key 并重算整张算法面；把它们纳入静态依赖索引，
        # 最终校验不再把“已有动态实现”的 controls 误报成提醒。
        for dyn_name in ('adaboost', 'boosting'):
            dyn = spec.get(dyn_name)
            if isinstance(dyn, dict):
                add_fields(dyn, ('errorModeKey', 'shrinkageKey', 'nuKey', 'roundsKey', 'targetKey'))
    elif kind == 'treefit':
        add(spec.get('points'))
        add_fields(spec.get('data'), ('x', 'y', 'cls'))
        treefit = spec.get('treefit')
        add_fields(treefit, ('splits',))
    elif kind == 'custom':
        custom = spec.get('custom')
        add_fields(custom, ('js',))
    elif kind == 'plotly':
        plotly = spec.get('plotly')
        if isinstance(plotly, dict):
            add(plotly.get('data'))
            layout = plotly.get('layout')
            if isinstance(layout, dict):
                for key, value in layout.items():
                    if key != 'title':
                        add(value)
    return out


def _contains_identifier(text, key):
    if not isinstance(text, str) or not isinstance(key, str):
        return False
    return re.search(r'(?<![A-Za-z0-9_$])%s(?![A-Za-z0-9_$])' % re.escape(key), text) is not None


def _validate_teaching(spec, kind, controls, valid_control_keys, err, warn):
    """校验 teaching 元数据，并返回真正需要进入主绘制的控件 key 集合。"""
    teaching = spec.get('teaching')
    if not isinstance(teaching, dict):
        for field in TEACHING_REQUIRED:
            err.append('teaching.%s 必填（teaching 必须说明问题、来源、控件作用与图形证据）' % field)
        return set()

    for field in TEACHING_REQUIRED:
        value = teaching.get(field)
        if field == 'controlEffect':
            ok = ((isinstance(value, str) and bool(value.strip())) or
                  (isinstance(value, list) and bool(value) and all(isinstance(x, str) and x.strip() for x in value)) or
                  (isinstance(value, dict) and bool(value) and all(isinstance(k, str) and k.strip() and
                                                                   isinstance(v, str) and v.strip()
                                                                   for k, v in value.items())))
            if not ok:
                err.append('teaching.controlEffect 必须是非空字符串、字符串数组或“控件 key → 作用”对象')
        elif not isinstance(value, str) or not value.strip():
            err.append('teaching.%s 必须是非空字符串' % field)

    def names_field(field):
        value = teaching.get(field)
        if value is None:
            return None
        if not isinstance(value, list) or any(not isinstance(x, str) or not x.strip() for x in value):
            err.append('teaching.%s 若给出，必须是字符串数组' % field)
            return []
        names = []
        for name in value:
            if name in names:
                err.append('teaching.%s 不能重复列出控件 key=%r' % (field, name))
            else:
                names.append(name)
        return names

    listed = names_field('controls')
    camera = set(CAMERA_CONTROL_KEYS)
    extra_camera = names_field('cameraControls')
    if extra_camera is not None:
        camera.update(extra_camera)
    for name in camera:
        if name not in valid_control_keys and extra_camera and name in extra_camera:
            err.append('teaching.cameraControls 列出了不存在的控件 key=%r' % name)
    if listed is not None:
        for name in listed:
            if name not in valid_control_keys:
                err.append('teaching.controls 列出了不存在的控件 key=%r' % name)
        semantic = set(name for name in listed if name in valid_control_keys)
    else:
        semantic = set(valid_control_keys)
    semantic.difference_update(camera)

    dynamic = teaching.get('dynamicRenderer', False)
    if not isinstance(dynamic, bool):
        err.append('teaching.dynamicRenderer 若给出，必须是 true/false')
        dynamic = False
    source = '\n'.join(_main_draw_strings(spec, kind))
    for name in sorted(semantic):
        if _contains_identifier(source, name):
            continue
        message = ('teaching 控件 %r 没有进入主绘制数据（只在 readouts/markers/title 等非主绘制字段中出现）；'
                   '请把它写入主绘制表达式/数据，或声明 teaching.dynamicRenderer=true 并由动态门禁验证' % name)
        if dynamic:
            warn.append(message + '（等待 verify_widget_pages.js 比较主图指纹）')
        else:
            err.append(message)
    return semantic


def validate_spec(spec):
    err, warn = [], []
    try:
        finite_json(spec)
    except ValueError as e:
        return [str(e)], warn
    if not isinstance(spec, dict):
        return ['spec 必须是 JSON 对象'], warn
    if spec.get('schema') != SCHEMA:
        err.append('schema 必须是 "%s"（当前 %r）' % (SCHEMA, spec.get('schema')))
    kind = spec.get('kind')
    if kind not in KINDS:
        err.append('kind 必须是 %s 之一（当前 %r）' % ('/'.join(KIND_ORDER), kind))
    if not isinstance(spec.get('title'), str) or not spec['title'].strip():
        err.append('title 必填且非空：写"能看出什么"，不要写"某图"')

    rnd = spec.get('renderer') if isinstance(spec.get('renderer'), str) else 'auto'

    def _points_budget(n, where):
        """给「一个点一个 DOM 节点」的数组设预算：超上限报错，超提醒线只提醒。
           写 renderer="canvas" 时点画在 canvas 上、不产生 DOM 节点，所以门槛抬高、措辞也跟着换。"""
        canvas = (rnd == 'canvas')
        warn_at = POINTS_WARN_CANVAS if canvas else POINTS_WARN
        max_at = POINTS_MAX_CANVAS if canvas else POINTS_MAX
        per_ms = (n / 50000.0) * (12.0 if canvas else 455.0)
        if n > max_at:
            err.append('%s 有 %d 个点，超过上限 %d（%s）：实测 %d 点约 %.0f ms/帧，拖滑杆会卡。'
                       '请抽稀，或改用 data{n, x, y} 逐点表达式让运行时只生成需要的点'
                       % (where, n, max_at, 'canvas 口径' if canvas else 'SVG 口径', n, per_ms))
        elif n > warn_at:
            warn.append('%s 有 %d 个点（> %d，%s）：实测约 %.0f ms/帧，拖滑杆会变钝；'
                        '建议抽稀到 %d 以内'
                        % (where, n, warn_at, '已写 renderer="canvas"' if canvas else '每点约 2 个 DOM 节点',
                           per_ms, warn_at))

    if spec.get('renderer') is not None:
        rnd = spec['renderer']
        if rnd not in ('auto', 'svg', 'canvas'):
            err.append('renderer 只能是 "auto"（默认：按数据量自选）、"svg"（每点一个 DOM 节点，'
                       '小数据便于悬停读数）或 "canvas"（密集 mark 画在 canvas 上，省节点）；当前 %r' % rnd)
        elif kind not in RENDERER_KINDS:
            warn.append('renderer 只对 %s 有意义（当前 kind=%r），其他渲染器一律用 SVG，会忽略它'
                        % ('/'.join(RENDERER_KINDS), kind))

    vars_ = spec.get('vars', {})
    if not isinstance(vars_, dict):
        err.append('vars 必须是对象')
        vars_ = {}
    for k, v in vars_.items():
        if not _valid_name(k):
            err.append('vars 的键 %r 不是合法变量名（字母或下划线开头）' % k)
        if not (_num(v) or (isinstance(v, list) and v and all(_num(x) for x in v))):
            err.append('vars[%s] 只能是数字或非空数字数组' % k)

    ctrls = spec.get('controls', [])
    if not isinstance(ctrls, list):
        err.append('controls 必须是数组')
        ctrls = []
    if len(ctrls) > 4:
        err.append('controls 最多 4 个（当前 %d）：一个组件只回答一个问题' % len(ctrls))
    seen = set()
    for i, c in enumerate(ctrls):
        where = 'controls[%d]' % i
        if not isinstance(c, dict):
            err.append('%s 必须是对象' % where)
            continue
        key, typ = c.get('key'), c.get('type')
        if not _valid_name(key):
            err.append('%s.key 缺失或不是合法变量名' % where)
        elif key in seen:
            err.append('%s.key 重复：%s' % (where, key))
        else:
            seen.add(key)
            if key in RESERVED:
                warn.append('control key %r 与内置名同名，会遮住辅助函数或变量（建议改名）' % key)
            if key in vars_:
                warn.append('control key %r 与 vars 同名，控件值会覆盖常量' % key)
        if typ not in ('slider', 'select', 'toggle', 'number'):
            err.append('%s.type 必须是 slider/select/toggle/number（当前 %r）' % (where, typ))
        if not isinstance(c.get('label'), str) or not c['label'].strip():
            err.append('%s.label 必填（控件上显示的中文标签）' % where)
        if typ in ('slider', 'number'):
            for f in ('min', 'max', 'step', 'value'):
                if not _num(c.get(f)):
                    err.append('%s.%s 必须是数字' % (where, f))
            if _num(c.get('min')) and _num(c.get('max')):
                if c['min'] >= c['max']:
                    err.append('%s：min 必须小于 max' % where)
                elif _num(c.get('value')) and not (c['min'] <= c['value'] <= c['max']):
                    err.append('%s.value 不在 [min, max] 区间内' % where)
        if typ == 'select':
            opts = c.get('options')
            if not isinstance(opts, list) or len(opts) < 2 or not all(
                    isinstance(o, list) and len(o) == 2 for o in opts):
                err.append('%s.options 必须是 [[值, 显示文本], …]，至少两项' % where)
            elif c.get('value') is not None and str(c['value']) not in [str(o[0]) for o in opts]:
                err.append('%s.value 不在 options 的值里' % where)
        if typ == 'toggle' and 'value' in c and not isinstance(c['value'], bool):
            err.append('%s.value 必须是 true/false' % where)

        # animate：让数值在读者不动手时自己走（渐进增强，读者一操作就停）
        if c.get('animate') is not None:
            if typ not in ('slider', 'number'):
                warn.append('%s.animate 只对 slider/number 生效（当前 type=%s），会被忽略' % (where, typ))
            else:
                an = c['animate']
                if an is True:
                    pass
                elif not isinstance(an, dict):
                    err.append('%s.animate 必须是 true 或对象 {from, to, seconds, pingpong, autostart}' % where)
                else:
                    for k in ('from', 'to', 'seconds'):
                        if k in an and not _num(an[k]):
                            err.append('%s.animate.%s 必须是数字' % (where, k))
                    if _num(an.get('from')) and _num(an.get('to')) and an['from'] == an['to']:
                        err.append('%s.animate：from 与 to 相同，自动演示不会动' % where)
                    if _num(an.get('seconds')) and not (0.5 <= an['seconds'] <= 120):
                        err.append('%s.animate.seconds 建议在 0.5–120 秒之间（当前 %s）' % (where, an['seconds']))
                    for k in ('autostart',):
                        if k in an and not isinstance(an[k], bool):
                            err.append('%s.animate.%s 必须是 true/false' % (where, k))
                    if 'pingpong' in an and not isinstance(an['pingpong'], bool):
                        err.append('%s.animate.pingpong 必须是 true/false' % where)
                    for k in ('from', 'to'):
                        if _num(an.get(k)) and not (min(c.get('min', an[k]), c.get('max', an[k]))
                                                    <= an[k] <= max(c.get('min', an[k]), c.get('max', an[k]))):
                            warn.append('%s.animate.%s 落在控件 [min, max] 之外，会被夹到端点' % (where, k))

    _validate_teaching(spec, kind, ctrls, set(seen), err, warn)

    series = spec.get('series', [])
    if series and not isinstance(series, list):
        err.append('series 必须是数组')
        series = []
    for i, s in enumerate(series or []):
        where = 'series[%d]' % i
        if not isinstance(s, dict):
            err.append('%s 必须是对象' % where)
            continue
        if not isinstance(s.get('label'), str) or not s['label'].strip():
            err.append('%s.label 必填（图例上的中文名）' % where)
        has_expr = isinstance(s.get('expr'), str) and s['expr'].strip()
        pts = s.get('points')
        has_pts = isinstance(pts, list) and len(pts) >= 1
        if not (has_expr or has_pts):
            err.append('%s 必须给 expr（表达式）或 points（[[x, y], …]）' % where)
        elif has_pts and not all(isinstance(p, list) and len(p) == 2 and _num(p[0]) and _num(p[1])
                                 for p in pts):
            err.append('%s.points 必须是 [[x, y], …] 且两项都是数字' % where)
        elif has_pts:
            _points_budget(len(pts), where)

    x = spec.get('x')
    if kind in ('plot', 'scatter'):
        if not isinstance(x, dict):
            err.append('%s 必须给 x{min, max, points}（横轴范围）' % kind)
        else:
            for f in ('min', 'max', 'points'):
                if not _num(x.get(f)):
                    err.append('x.%s 必须是数字' % f)
            if _num(x.get('min')) and _num(x.get('max')):
                if x['min'] >= x['max']:
                    err.append('x：min 必须小于 max')
            if kind == 'plot' and _num(x.get('points')) and x['points'] < 2:
                err.append('x.points 至少 2（曲线采样点数）')
    if kind in ('contour', 'vector', 'descent', 'surface3d'):
        _field_axis(spec.get('x'), kind, 'x', err)
        _field_axis(spec.get('y'), kind, 'y', err)
    if kind == 'plot' and not series:
        err.append('plot 至少需要 1 条 series')

    if kind == 'bars':
        bars = spec.get('bars')
        if not isinstance(bars, list) or not bars:
            err.append('bars 必须是非空数组')
        else:
            for i, b in enumerate(bars):
                if not isinstance(b, dict):
                    err.append('bars[%d] 必须是对象' % i)
                    continue
                if not isinstance(b.get('label'), str) or not b['label'].strip():
                    err.append('bars[%d].label 必填' % i)
                if not isinstance(b.get('value'), str) or not b['value'].strip():
                    err.append('bars[%d].value 必填（数字或表达式）' % i)

    if kind == 'histogram':
        h = spec.get('histogram')
        if not isinstance(h, dict):
            err.append('histogram 必须给 histogram{sample|values}')
        else:
            ok_sample = isinstance(h.get('sample'), str) and h['sample'].strip()
            ok_values = isinstance(h.get('values'), list) and bool(h['values'])
            if not (ok_sample or ok_values):
                err.append('histogram 必须给 sample（表达式，可用 randn()）或 values（[[值, 频数], …]）')
            if h.get('bins') is not None and not (isinstance(h['bins'], int) and 2 <= h['bins'] <= 500):
                err.append('histogram.bins 若给出，须是 2–500 的整数')

    if kind == 'box':
        b = _sample_source(spec, kind, 'box', err)
        if isinstance(b, dict) and b.get('fmt') is not None and not isinstance(b['fmt'], str):
            err.append('box.fmt 若给出，必须是格式字符串（如 "0.00"）')

    if kind == 'ecdf':
        e = _sample_source(spec, kind, 'ecdf', err)
        _max_points(e, 'ecdf', err)

    if kind == 'qq':
        q = _sample_source(spec, kind, 'qq', err)
        _max_points(q, 'qq', err)
        if isinstance(q, dict):
            dist = q.get('dist')
            if dist is not None and (not isinstance(dist, str) or not dist.strip()):
                err.append('qq.dist 若是给出，必须是非空字符串（目前只支持 "normal"）')
            elif isinstance(dist, str) and dist.strip() and dist != 'normal':
                err.append('qq.dist 目前只支持 "normal"（当前 %r）：未知分布不会静默按正态画' % dist)

    if kind == 'contour':
        c = spec.get('contour')
        if not isinstance(c, dict):
            err.append('contour 必须给 contour{expr}（f(x, y) 表达式）')
        else:
            if not isinstance(c.get('expr'), str) or not c['expr'].strip():
                err.append('contour.expr 必填（f(x, y) 表达式，变量是 x 与 y）')
            lv = c.get('levels')
            if lv is not None and not (isinstance(lv, int) and not isinstance(lv, bool) and 1 <= lv <= 20):
                err.append('contour.levels 若给出，须是 1–20 的整数（等值线条数；省略时默认 6）')
            if c.get('fmt') is not None and not isinstance(c['fmt'], str):
                err.append('contour.fmt 若给出，必须是格式字符串（如 "0.00"）')

    if kind == 'vector':
        v = spec.get('vector')
        if not isinstance(v, dict):
            err.append('vector 必须给 vector{u, v}（两个分量表达式）')
        else:
            for f in ('u', 'v'):
                if not isinstance(v.get(f), str) or not v[f].strip():
                    err.append('vector.%s 必填（分量表达式，变量是 x 与 y）' % f)
            sc = v.get('scale')
            if sc is not None and sc not in ('auto', 'unit'):
                err.append('vector.scale 只能是 "auto"（长度 ∝ |(u,v)|，默认）或 "unit"（等长，只表示方向）')
            if v.get('fmt') is not None and not isinstance(v['fmt'], str):
                err.append('vector.fmt 若给出，必须是格式字符串（如 "0.00"）')

    if kind == 'matrix':
        m = spec.get('matrix')
        if not isinstance(m, dict):
            err.append('matrix 必须给 matrix{values}（2×2 矩阵 [[a, b], [c, d]]）')
        else:
            vals = m.get('values')
            ok_vals = (isinstance(vals, list) and len(vals) == 2 and all(
                isinstance(r, list) and len(r) == 2 and all(_num(x) for x in r) for r in vals))
            if not ok_vals:
                err.append('matrix.values 必须是 2×2 数字矩阵 [[a, b], [c, d]]')
            if m.get('editable') is not None and not isinstance(m['editable'], bool):
                err.append('matrix.editable 若给出，必须是 true/false')
            if m.get('fmt') is not None and not isinstance(m['fmt'], str):
                err.append('matrix.fmt 若给出，必须是格式字符串（如 "0.00"）')
            smp = m.get('samples')
            if smp is not None and not (isinstance(smp, list) and all(
                    isinstance(p, list) and len(p) == 2 and _num(p[0]) and _num(p[1]) for p in smp)):
                err.append('matrix.samples 若给出，必须是 [[x, y], …] 数字点数组')
            bind = m.get('bind')
            if bind is not None:
                if not isinstance(bind, dict):
                    err.append('matrix.bind 必须是 {变量名: [行, 列]} 对象（行/列取 0 或 1）')
                else:
                    for name, at in bind.items():
                        if not _valid_name(name):
                            err.append('matrix.bind 的键 %r 不是合法变量名（字母或下划线开头）' % name)
                        if not (isinstance(at, list) and len(at) == 2 and all(
                                isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 1 for x in at)):
                            err.append('matrix.bind.%s 必须是 [行, 列]，行/列取 0 或 1' % name)

    if kind == 'heatmap':
        h = spec.get('heat')
        if not isinstance(h, dict):
            err.append('heatmap 必须给 heat{rows, cols, values}')
        else:
            rows, cols, vals = h.get('rows'), h.get('cols'), h.get('values')
            if not (isinstance(rows, list) and rows and isinstance(cols, list) and cols):
                err.append('heat.rows / heat.cols 必须是非空数组')
            elif not (isinstance(vals, list) and len(vals) == len(rows)
                      and all(isinstance(r, list) and len(r) == len(cols) for r in vals)):
                err.append('heat.values 必须是 %d×%d 的矩阵' % (len(rows), len(cols)))
            else:
                cells = len(rows) * len(cols)
                if cells > HEAT_CELLS_MAX:
                    err.append('heat 有 %d×%d = %d 格，超过上限 %d 格：非可编辑热力图每格一个 DOM 节点'
                               '（120×120 实测约 92 ms/帧）。请降低维度，或写 renderer="canvas" 让'
                               '密集格子画在 canvas 上' % (len(rows), len(cols), cells, HEAT_CELLS_MAX))
                elif cells > HEAT_CELLS_WARN and rnd != 'canvas':
                    warn.append('heat 有 %d×%d = %d 格（> %d）：每格一个 DOM 节点，120×120 实测约 92 ms/帧；'
                                '拖滑杆会变钝——数据量大时写 renderer="canvas"'
                                % (len(rows), len(cols), cells, HEAT_CELLS_WARN))

    if kind == 'timeline':
        t = spec.get('timeline')
        if not isinstance(t, dict) or not isinstance(t.get('items'), list) or not t['items']:
            err.append('timeline 必须给 timeline.items（非空数组）')
        else:
            for i, it in enumerate(t['items']):
                ok_expr = (isinstance(it, dict) and
                           (_num(it.get('t')) or isinstance(it.get('t'), str) and it['t'].strip()) and
                           (_num(it.get('amount')) or isinstance(it.get('amount'), str) and it['amount'].strip()))
                if not ok_expr:
                    err.append('timeline.items[%d] 必须有数字或表达式字符串 t 与 amount' % i)
                elif not isinstance(it.get('label'), str) or not it['label'].strip():
                    err.append('timeline.items[%d].label 必填' % i)

    if kind == 'tree':
        t = spec.get('tree')
        if not isinstance(t, dict) or not isinstance(t.get('root'), dict):
            err.append('tree 必须给 tree.root（对象，含 label / value / children）')

    def _pairs_source(where):
        """regression / pca / treefit 共用：points 或 data{n, x, y} 二选一。"""
        ok_pts = isinstance(spec.get('points'), list) and bool(spec['points'])
        d = spec.get('data')
        ok_data = isinstance(d, dict) and isinstance(d.get('x'), str) and bool(d['x'].strip()) \
            and isinstance(d.get('y'), str) and bool(d['y'].strip())
        if not (ok_pts or ok_data):
            err.append('%s 必须给 points（[[x, y], …]）或 data{n, x, y}（逐点表达式）' % where)
        if ok_pts:
            bad = [p for p in spec['points'] if not (isinstance(p, list) and len(p) == 2 and all(_num(v) for v in p))]
            if bad:
                err.append('%s.points 必须是 [[x, y], …] 且两项都是数字' % where)
            else:
                _points_budget(len(spec['points']), where)
        if ok_data:
            n = d.get('n')
            if n is not None and not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= DATA_N_MAX):
                err.append('%s.data.n 必须是 1–%d 的整数（逐点表达式生成的样本量；当前 %r）：'
                           '要更多点请改用 points 并抽稀' % (where, DATA_N_MAX, n))
        return ok_pts, ok_data

    if kind == 'regression':
        _pairs_source('regression')
        f = spec.get('fit')
        if f is not None and not isinstance(f, dict):
            err.append('fit 若给出，必须是对象 {mode, slopeKey, interceptKey, slope, intercept, residuals}')
        elif isinstance(f, dict):
            md = f.get('mode')
            if md is not None and md not in ('ols', 'manual', 'compare', 'none'):
                err.append('fit.mode 只能是 "ols"/"manual"/"compare"/"none"（当前 %r）' % md)
            if md in ('manual', 'compare'):
                sk = f.get('slopeKey', 'slope')
                if not _valid_name(sk):
                    err.append('fit.slopeKey 必须是控件 key（字母或下划线开头）')
                elif not (sk in seen or _num(f.get('slope')) or _num(f.get('intercept'))):
                    err.append('fit.mode=%s 需要控件 key="%s"（或 fit.slope / fit.intercept 常量）' % (md, sk))
            for k in ('slope', 'intercept'):
                if k in f and not _num(f[k]):
                    err.append('fit.%s 必须是数字' % k)
            if 'residuals' in f and not isinstance(f['residuals'], bool):
                err.append('fit.residuals 必须是 true/false（false = 不画残差竖线）')

    if kind == 'pca':
        _pairs_source('pca')
        if spec.get('meanCenter') is not None and not isinstance(spec['meanCenter'], bool):
            err.append('pca.meanCenter 若给出，必须是 true/false')
        ca = spec.get('candidate')
        if ca is not None and not isinstance(ca, dict):
            err.append('candidate 若给出，必须是对象 {angle, angleControl, ellipse, showProjection}')
        elif isinstance(ca, dict):
            if ca.get('angle') is not None and not _num(ca['angle']):
                err.append('candidate.angle 必须是数字（候选轴方向，单位是度）')
            ac = ca.get('angleControl')
            if ac is not None:
                if not _valid_name(ac):
                    err.append('candidate.angleControl 必须是控件 key（字母或下划线开头）')
                elif ac not in seen:
                    warn.append('candidate.angleControl="%s" 不是任何控件的 key：运行时改用 candidate.angle' % ac)
            for k in ('ellipse', 'showProjection'):
                if k in ca and not isinstance(ca[k], bool):
                    err.append('candidate.%s 必须是 true/false' % k)

    if kind == 'descent':
        c = spec.get('contour')
        if not isinstance(c, dict) or not (isinstance(c.get('expr'), str) and c['expr'].strip()):
            err.append('descent 必须给 contour.expr（f(x, y) = 损失函数；步长与步数由控件控制）')
        gr = spec.get('grad')
        if gr is not None and not isinstance(gr, dict):
            err.append('grad 若给出，必须是对象 {dfdx, dfdy}（两个偏导表达式）')
        elif isinstance(gr, dict):
            okx = bool(isinstance(gr.get('dfdx'), str) and gr['dfdx'].strip())
            oky = bool(isinstance(gr.get('dfdy'), str) and gr['dfdy'].strip())
            if okx != oky:
                err.append('grad.dfdx 与 grad.dfdy 必须同时给：只给一个会静默退回中心差分')
        st = spec.get('start')
        if st is not None and not (isinstance(st, list) and len(st) == 2 and all(_num(v) for v in st)):
            err.append('start 若给出，必须是 [x, y] 两个数字（迭代起点）')
        dd = spec.get('descent')
        if dd is not None and not isinstance(dd, dict):
            err.append('descent 若给出，必须是对象 {lrKey, stepsKey, clickToSetStart}')
        elif isinstance(dd, dict):
            for k, dk, what in (('lrKey', 'lr', '学习率 α'), ('stepsKey', 'steps', '迭代步数')):
                kk = dd.get(k, dk)
                if not _valid_name(kk):
                    err.append('descent.%s 必须是控件 key（字母或下划线开头）' % k)
                elif kk not in seen and not _num(dd.get(dk)):
                    warn.append('descent 找不到%s：控件 key="%s" 不存在、也没有 descent.%s 常量 → 用默认值'
                                % (what, kk, dk))
            if 'clickToSetStart' in dd and not isinstance(dd['clickToSetStart'], bool):
                err.append('descent.clickToSetStart 必须是 true/false')

    if kind == 'surface3d':
        # 动态算法曲面仍是 surface3d，但数据由 renderer 按算法递推生成；不能让一个空 surface
        # 触发“缺 expr/points”，也不能同时给静态 surface 造成两个事实来源。
        dyn_names = [k for k in ('adaboost', 'boosting')
                     if isinstance(spec.get(k), dict) and not isinstance(spec.get(k), list)]
        if len(dyn_names) > 1:
            err.append('surface3d 的 adaboost 与 boosting 只能二选一')
        dynamic_name = dyn_names[0] if dyn_names else None
        s3 = spec.get('surface')
        if dynamic_name:
            if s3 is not None and not isinstance(s3, dict):
                err.append('动态 surface3d 的 surface 若给出必须是对象（通常只写 {"mode":"surface"}）')
            elif isinstance(s3, dict) and (s3.get('expr') is not None or s3.get('points') is not None):
                err.append('surface3d.%s 不能与 surface.expr/points 同时给：动态算法块必须是唯一主数据源'
                           % dynamic_name)
            dyn = spec[dynamic_name]
            if dyn.get('version') is not None and not (isinstance(dyn['version'], int) and not isinstance(dyn['version'], bool)):
                err.append('%s.version 若给出必须是整数' % dynamic_name)
            if not isinstance(dyn.get('data'), dict):
                err.append('%s.data 必须是对象（确定性算法算例数据）' % dynamic_name)
            elif dynamic_name == 'adaboost':
                data = dyn['data']
                if not isinstance(data.get('labels'), list) or len(data['labels']) < 2:
                    err.append('adaboost.data.labels 至少需要 2 个标签')
                if not isinstance(data.get('learners'), list) or not data['learners']:
                    err.append('adaboost.data.learners 至少需要一条弱学习器')
                for key in ('errorModeKey', 'shrinkageKey', 'roundsKey'):
                    if dyn.get(key) is not None and (not _valid_name(dyn[key]) or dyn[key] not in seen):
                        err.append('%s.%s 必须指向 controls 里的 key（当前 %r）' % (dynamic_name, key, dyn.get(key)))
                if not isinstance(spec.get('teaching'), dict) or spec['teaching'].get('dynamicRenderer') is not True:
                    err.append('surface3d.%s 必须声明 teaching.dynamicRenderer=true（主图由 renderer 递推生成）' % dynamic_name)
            else:
                data = dyn['data']
                response = data.get('response', data.get('y', data.get('targets')))
                if not isinstance(response, list) or len(response) < 2:
                    err.append('boosting.data.response（或 y/targets）至少需要 2 个数字')
                if dyn.get('rounds') is not None and not (_num(dyn['rounds']) and 1 <= dyn['rounds'] <= 300):
                    err.append('boosting.rounds 若给出须是 1–300 的数字')
                key = dyn.get('nuKey', 'nu')
                if not _valid_name(key) or key not in seen:
                    err.append('boosting.nuKey 必须指向 controls 里的 key（当前 %r）' % key)
                if not isinstance(spec.get('teaching'), dict) or spec['teaching'].get('dynamicRenderer') is not True:
                    err.append('surface3d.boosting 必须声明 teaching.dynamicRenderer=true（主图由 renderer 递推生成）')
        elif not isinstance(s3, dict):
            err.append('surface3d 必须给 surface{expr 或 points}（z = f(x, y) 或三维点云）')
        else:
            ok_expr = isinstance(s3.get('expr'), str) and s3['expr'].strip()
            pts = s3.get('points')
            ok_pts = isinstance(pts, list) and bool(pts)
            if not (ok_expr or ok_pts):
                err.append('surface3d 必须给 surface.expr（z = f(x, y)）或 surface.points（[[x, y, z], …]）')
            if ok_pts and not all(isinstance(p, list) and len(p) == 3 and all(_num(v) for v in p) for p in pts):
                err.append('surface.points 必须是 [[x, y, z], …] 且三项都是数字')
            if s3.get('mode') is not None and s3['mode'] not in ('surface', 'wireframe', 'points'):
                err.append('surface.mode 只能是 "surface"/"wireframe"/"points"（当前 %r）' % s3['mode'])
            if s3.get('fmt') is not None and not isinstance(s3['fmt'], str):
                err.append('surface.fmt 若给出，必须是格式字符串（如 "0.00"）')
        pth = spec.get('path')
        if pth is not None and not (isinstance(pth, list) and pth and all(
                isinstance(p, list) and len(p) == 3 and all(_num(v) for v in p) for p in pth)):
            err.append('path 若给出，必须是三维轨迹 [[x, y, z], …]')
        vw = spec.get('view')
        if vw is not None and not isinstance(vw, dict):
            err.append('view 若给出，必须是对象 {azimuthKey, elevationKey, zoomKey, azimuth, elevation, zoom}')
        elif isinstance(vw, dict):
            for k in ('azimuthKey', 'elevationKey', 'zoomKey'):
                if vw.get(k) is None:
                    continue
                if not _valid_name(vw[k]):
                    err.append('view.%s 必须是控件 key（字母或下划线开头）' % k)
                elif vw[k] not in seen:
                    warn.append('view.%s="%s" 不是任何控件的 key：运行时用 view 里的常量或默认视角' % (k, vw[k]))
            for k in ('azimuth', 'elevation', 'zoom'):
                if vw.get(k) is not None and not _num(vw[k]):
                    err.append('view.%s 必须是数字' % k)

    if kind == 'treefit':
        t = spec.get('treefit')
        if t is not None and not isinstance(t, dict):
            err.append('treefit 若给出，必须是对象 {mode, splits, depth, depthKey}')
        t = t if isinstance(t, dict) else {}
        md = spec.get('mode') if spec.get('mode') is not None else t.get('mode')
        if md is not None and md not in ('1d', '2d'):
            err.append('treefit 的 mode 只能是 "1d"（阈值切 x）或 "2d"（轴对齐区域）（当前 %r）' % md)
        ok_pts, _ = _pairs_source('treefit')
        splits = t.get('splits')
        if not (isinstance(splits, list) and splits):
            err.append('treefit 必须给 treefit.splits（1d: [{"at": 阈值}, …]；'
                       '2d: [{"axis": "x", "at": 阈值}, …]），按"加深一层"的顺序')
        elif md == '2d':
            for i, sp in enumerate(splits):
                if not (isinstance(sp, dict) and sp.get('axis') in ('x', 'y')
                        and (_num(sp.get('at')) or isinstance(sp.get('at'), str))):
                    err.append('treefit.splits[%d] 在 mode="2d" 下必须是 {axis: "x"|"y", at: 数字或表达式}' % i)
        else:
            for i, sp in enumerate(splits):
                if not (_num(sp) or (isinstance(sp, dict)
                                     and (_num(sp.get('at')) or isinstance(sp.get('at'), str)))):
                    err.append('treefit.splits[%d] 在 mode="1d" 下必须是数字阈值或 {"at": 数字或表达式}' % i)
        if md == '2d' and ok_pts and any(len(p) < 3 for p in spec['points']):
            err.append('treefit mode="2d" 的 points 每点要写 [x, y, cls]（cls 是类别标签）')
        dk = t.get('depthKey') if t.get('depthKey') is not None else 'depth'
        if not _valid_name(dk):
            err.append('treefit.depthKey 必须是控件 key（字母或下划线开头）')
        elif dk not in seen and t.get('depth') is None:
            warn.append('treefit 找不到深度控件 key="%s"：运行时用 treefit.depth 或取最大深度' % dk)

    for i, m in enumerate(spec.get('markers') or []):
        if kind == 'surface3d' and isinstance(m, dict) and not _num(m.get('z')):
            err.append('markers[%d] 在 surface3d 里必须带 z（{x, y, z, label}）：二维的竖线标记不适用' % i)

    if kind == 'custom':
        c = spec.get('custom')
        if not isinstance(c, dict) or not (c.get('html') or c.get('js')):
            err.append('custom 必须给 custom.html 和/或 custom.js')
        else:
            # custom 允许逻辑，但组件页必须能离线打开；拒绝最常见的外部依赖与网络入口。
            blob = str(c.get('html') or '') + '\n' + str(c.get('js') or '')
            if re.search(r'<\s*(?:script|link)\b[^>]*(?:src|href)\s*=', blob, re.I):
                err.append('custom 不允许 <script src> 或 <link href>：组件必须单文件、零外部依赖')
            if re.search(r'\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(', blob):
                err.append('custom 不允许网络请求：组件必须离线可用')
            if re.search(r'https?://|//cdn\.', blob, re.I):
                err.append('custom 不允许 URL/CDN：把必要资源内联，或改用内置渲染器')

    if kind == 'plotly':
        p = spec.get('plotly')
        if not isinstance(p, dict):
            err.append('plotly 必须给 plotly{data: […]}（骨架见 make_widget.py --spec plotly）')
        else:
            data = p.get('data')
            # 生成器 {"by": …, "n"|"rows"/"cols": …, "vars": {…}}：形状写错是最常见的手滑，静态能查的先查掉
            GEN_FIELDS = ('by', 'n', 'rows', 'cols', 'vars')

            def _gen_check(node, where):
                if not isinstance(node, dict):
                    return
                if 'by' in node:
                    if not isinstance(node['by'], str) or not node['by'].strip():
                        err.append('%s.by 必须是表达式字符串（生成器逐元素求值它）' % where)
                    two = ('rows' in node) or ('cols' in node)
                    if two and ('rows' not in node or 'cols' not in node):
                        err.append('%s 是二维生成器，rows 与 cols 必须同时给（当前只有 %s）'
                                   % (where, 'rows' if 'rows' in node else 'cols'))
                    for f in (('rows', 'cols') if two else ('n',)):
                        v = node.get(f)
                        if v is None:
                            continue
                        if not (_num(v) or (isinstance(v, str) and v.startswith('='))):
                            err.append('%s.%s 必须是数字或以 = 开头的表达式（当前 %r）' % (where, f, v))
                    if node.get('vars') is not None:
                        gv = node['vars']
                        if not isinstance(gv, dict) or not gv:
                            err.append('%s.vars 若是生成器的逐元素名字表，必须是非空对象（当前 %r）'
                                       % (where, gv))
                        else:
                            for gk, gval in gv.items():
                                if not _valid_name(gk):
                                    err.append('%s.vars 的名字 %r 不合法（字母/下划线开头，后接字母数字下划线）'
                                               % (where, gk))
                                if not isinstance(gval, str) or not gval.strip():
                                    err.append('%s.vars.%s 必须是表达式字符串（当前 %r）' % (where, gk, gval))
                    for k in node:
                        if k not in GEN_FIELDS:
                            warn.append('%s 里的字段 %r 会被生成器忽略（生成器只认 %s）'
                                        % (where, k, ' / '.join(GEN_FIELDS)))
                    return
                for k, v in node.items():
                    if isinstance(v, dict):
                        _gen_check(v, '%s.%s' % (where, k))

            if isinstance(data, list):
                for i, t in enumerate(data):
                    if isinstance(t, dict):
                        for k, v in t.items():
                            if isinstance(v, dict):
                                _gen_check(v, 'plotly.data[%d].%s' % (i, k))
            if isinstance(p.get('layout'), dict):
                for k, v in p['layout'].items():
                    if isinstance(v, dict):
                        _gen_check(v, 'plotly.layout.%s' % k)
            if not isinstance(data, list) or not data:
                err.append('plotly.data 必须是非空数组（每个元素是一个 Plotly trace 对象）')
            else:
                bad_traces = [i for i, t in enumerate(data) if not isinstance(t, dict)]
                if bad_traces:
                    err.append('plotly.data 的元素必须是对象（Plotly trace）：第 %s 个不是'
                               % '、'.join(str(i) for i in bad_traces[:4]))
            for f in ('layout', 'config'):
                if p.get(f) is not None and not isinstance(p[f], dict):
                    err.append('plotly.%s 若给出，必须是对象（当前 %r）' % (f, p[f]))
            if p.get('height') is not None and not (_num(p['height']) and 120 <= p['height'] <= 2000):
                err.append('plotly.height 若给出，须是 120–2000 的数字（px；省略时按图区宽度取 '
                           'clamp(宽×0.72, 260, 520)）')
            bundle = p.get('bundle')
            if bundle is not None:
                if not isinstance(bundle, str) or bundle not in PLOTLY_BUNDLES:
                    err.append('plotly.bundle 只能是 %s 之一（当前 %r）；省略时用 %r——'
                               '%s'
                               % ('/'.join('"%s"' % k for k in sorted(PLOTLY_BUNDLES)),
                                  bundle, PLOTLY_BUNDLE_DEFAULT,
                                  '；'.join('"%s" = %s' % (k, PLOTLY_BUNDLES[k][1])
                                            for k in sorted(PLOTLY_BUNDLES))))
                elif bundle == 'gl3d':
                    # gl3d 裁剪包里没有 2D（scatter/bar/pie…）与地图、parcoords；写了 2D trace 只会白屏
                    traces = [t.get('type') for t in (data if isinstance(data, list) else [])
                              if isinstance(t, dict)]
                    three_d = ('surface', 'scatter3d', 'mesh3d', 'isosurface', 'volume', 'cone',
                               'streamtube')
                    plain = [t for t in traces if (t or '').lower() not in three_d]
                    if plain:
                        warn.append('plotly.bundle="gl3d" 是只含 3D 的裁剪包，但 data 里有 %s —— '
                                    '这些 trace 在这个包下画不出来（页面里会 [错误]/空白）；'
                                    '要 2D 就改用 bundle="full"（页面会大到约 5 MB）'
                                    % '、'.join('type=%r' % t for t in sorted(set(plain))[:4]))
        foreign = [k for k in PLOTLY_FOREIGN_BLOCKS if spec.get(k) is not None]
        if foreign:
            err.append('plotly 只读 plotly.*（data / layout / config / height / bundle）：请把 %s 的内容写进 '
                       'plotly.data 或 plotly.layout——其他渲染器的块在这里不生效，所以直接拦下'
                       % '、'.join(foreign))
        key = plotly_bundle_key(spec if isinstance(spec, dict) else {})
        lib_rel = plotly_bundle_rel(key)
        if not os.path.exists(plotly_bundle_path(key)):
            warn.append('kind=plotly（bundle=%r）但缺少 %s：构建出的页面里没有内联的 Plotly，打开会显示 '
                        '[错误]；下载与校验方式见 %s' % (key, lib_rel, PLOTLY_README_REL))
        elif vendor_sha_from_readme(key) is None:
            warn.append('%s 里没有登记 %s 的 sha256 机器可读块：--check 没法校验这份库没被换过；'
                        '补法见 %s' % (PLOTLY_README_REL, PLOTLY_BUNDLES[key][0], PLOTLY_README_REL))
        if not os.path.exists(PLOTLY_ADAPTER):
            warn.append('缺少 %s：页面会内联库但没有注册 plotly 渲染器，打开会显示「未知 kind」；'
                        '从仓库里恢复该文件即可' % rel_vault(PLOTLY_ADAPTER))
    for i, m in enumerate(spec.get('markers') or []):
        if not isinstance(m, dict) or 'x' not in m:
            err.append('markers[%d] 必须有 x（数字或表达式）' % i)
        elif not (_num(m['x']) or isinstance(m['x'], str)):
            err.append('markers[%d].x 必须是数字或表达式' % i)

    for i, r in enumerate(spec.get('readouts') or []):
        if not isinstance(r, dict) or not isinstance(r.get('label'), str) or not r['label'].strip():
            err.append('readouts[%d].label 必填' % i)
        elif not isinstance(r.get('expr'), str) or not r['expr'].strip():
            err.append('readouts[%d].expr 必填（表达式）' % i)

    if spec.get('notes') is not None:
        if not isinstance(spec['notes'], list) or not all(isinstance(n, str) for n in spec['notes']):
            err.append('notes 必须是字符串数组')
    if isinstance(spec.get('uid'), str) and spec['uid'] and not re.match(r'^N\d{4}\.\d{2}$', spec['uid']):
        warn.append('uid %r 不像二级条目 uid（形如 N1104.05），确认没写错' % spec['uid'])
    aspect = spec.get('aspect')
    if aspect is not None:
        if not isinstance(aspect, str) or aspect not in ('auto', 'equal'):
            err.append('aspect 只能是 \"auto\"（两轴各自独立拉伸填满图框，几何形状会被压扁）或 '
                       '\"equal\"（两轴同一 px/单位，几何保真）；默认 contour/vector 是 \"auto\"、'
                       'matrix 是 \"equal\"，当前 %r' % (aspect,))
        elif kind not in ASPECT_KINDS:
            warn.append('aspect 只对 %s 有意义（当前 kind=%r），其他渲染器会忽略它'
                        % ('/'.join(ASPECT_KINDS), kind))
    theme = spec.get('theme', 'light')
    if theme not in ('light', 'system'):
        err.append('theme 只能是 "light"（默认白底）或 "system"（跟随系统黑白）；当前 %r' % theme)
    return err, warn


# --------------------------------------------------------------------------- #
# HTML 组装：单文件、全内联、零外部请求
# --------------------------------------------------------------------------- #
SHELL = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; media-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="color-scheme" content="light dark">
<meta name="wg-note" content="__NOTE__">
<meta name="wg-uid" content="__UID__">
<meta name="wg-vault" content="__VAULT_NAME__">
<meta name="wg-spec-src" content="__SPEC_SRC__">
<meta name="wg-built-by" content="__TOOLS_DIR__/make_widget.py">
<meta name="wg-generated" content="派生文件：改 __SPEC_SRC__ 后重跑 make_widget.py，不要手改本文件">
__VENDOR_META__<title>__TITLE__</title>
<style>
/*__WIDGETSCSS__*/
</style>
</head>
<body>
<main id="wg-root"></main>
<script type="application/json" id="wg-spec">__SPEC__</script>
__VENDOR_LIB__<script>
/*__WIDGETSJS__*/
</script>
__VENDOR_ADAPTER__<script>
/* 只启动一次：运行时若自己也监听 DOMContentLoaded，这里用 dataset 标记避免重复渲染 */
(function () {
  var d = document.documentElement;
  if (window.WG && typeof WG.boot === 'function' && !d.dataset.wgBooted) {
    d.dataset.wgBooted = '1';
    WG.boot();
  }
})();
</script>
</body>
</html>
'''


def build_html(spec, note, spec_rel):
    """spec + 运行时 → 单文件 HTML。输出必须确定性（不含时间戳），否则 --check 没法靠 sha 判过期。"""
    missing = [rel_vault(p) for p in (WIDGETS_JS, WIDGETS_CSS) if not os.path.exists(p)]
    if missing:
        sys.exit('[错误] 缺少运行时文件：%s' % '、'.join(missing))
    js, css = read_text(WIDGETS_JS), read_text(WIDGETS_CSS)
    # 内联进 <script>/<style> 的代码里若出现 </script>、</style> 字面量会提前闭合，做最小安全替换
    if re.search(r'</script', js, re.I):
        js = re.sub(r'</script', '<\\/script', js, flags=re.I)
    if re.search(r'</style', css, re.I):
        css = re.sub(r'</style', '<\\/style', css, flags=re.I)
    out = SHELL
    vendor_meta, vendor_lib, vendor_adapter = '', '', ''
    if isinstance(spec, dict) and spec.get('kind') == 'plotly':
        # 只有 plotly 页带库：其他页面这三处替换成空串，页面里连一行 vendor 痕迹都没有
        vendor = plotly_vendor(plotly_bundle_key(spec))
        text, _sha, _escapes, _version, _rel = vendor
        vendor_meta = plotly_meta_tags(vendor)
        if text is not None:
            vendor_lib = '<script id="wg-vendor-lib">\n%s\n</script>\n' % text
            # adapter 必须排在 widgets.js **之后**：它靠 WG.registerKind 注册渲染器
            if os.path.exists(PLOTLY_ADAPTER):
                adapter = read_text(PLOTLY_ADAPTER)
                if re.search(r'</script', adapter, re.I):
                    adapter = re.sub(r'</script', '<\\/script', adapter, flags=re.I)
                vendor_adapter = '<script id="wg-vendor-adapter">\n%s\n</script>\n' % adapter
    for key, val in (
            ('__TOOLS_DIR__', attr(LAYOUT['toolsDir'])),
            ('__NOTE__', attr(note)),
            ('__UID__', attr(spec.get('uid') or '')),
            ('__VAULT_NAME__', attr(os.path.basename(VAULT))),
            ('__SPEC_SRC__', attr(spec_rel)),
            ('__TITLE__', attr(spec.get('title') or '交互组件')),
            ('__SPEC__', json_for_script(spec)),
            ('__VENDOR_META__', vendor_meta),
            ('__VENDOR_LIB__', vendor_lib),
            ('__VENDOR_ADAPTER__', vendor_adapter),
    ):
        out = out.replace(key, val)
    out = out.replace('/*__WIDGETSJS__*/', js).replace('/*__WIDGETSCSS__*/', css)
    left = [k for k in ('__TOOLS_DIR__', '__VENDOR_META__', '__VENDOR_LIB__', '__VENDOR_ADAPTER__') if k in out]
    if left:                                        # 占位符拼错/被改名时别静默漏掉
        sys.exit('[错误] SHELL 里的占位符没被替换：%s' % '、'.join(left))
    return out


def md_target(rel):
    """普通 Markdown 链接：完整 percent-encoding，避免空格、#、% 或 ) 破坏路径。"""
    return quote(rel, safe='/._~-')


def index_md_text(reg):
    """生成 <widgetsDir>/Index.md 内容。排序固定，只有真实变化才产生 diff。"""
    rows = sorted(reg['widgets'], key=lambda w: (w.get('note', ''), w.get('json', '')))
    notes = set(w.get('note') for w in rows)
    tool_dir = LAYOUT['toolsDir']
    out = [
        # frontmatter + 无 wikilink：这份索引会被放在普通 vault 里，得先能过通用的 vault 体检
        # （不少 vault 的体检脚本会检查"缺 frontmatter"和"悬空 wikilink"，而组件目录未必是链接目标）
        '---',
        'type: index',
        'title: "交互组件清单"',
        '---',
        '',
        '# 交互组件清单',
        '',
        '> 本文件由 `python3 %s/make_widget.py --index` 生成，**不要手改**。' % tool_dir,
        '> 写作规范见 `%s/INTERACTIVE-AUTHORING.md`；源 spec 与派生产物都在本目录。' % tool_dir,
        '> Obsidian 阅读视图会剥掉 `<script>`/`<iframe>`，所以笔记正文只能放一个指向 `.html` 的普通链接。',
        '> 想在浏览器里一眼看全部组件（卡片 + 就地预览）：打开同目录的 `%s`（`--board` 生成）。' % BOARD_NAME,
        '',
        '共 %d 个组件，来自 %d 篇笔记。' % (len(rows), len(notes)),
        '',
    ]
    if not rows:
        out += ['还没有组件。先看 `python3 %s/make_widget.py --list` 有哪些渲染器，' % tool_dir,
                '再用 `--spec <kind>` 取骨架。', '']
        return '\n'.join(out)
    out += ['| 组件 | 渲染器 | 源笔记 | 条目 | 打开 | 源 spec |', '|---|---|---|---|---|---|']
    for w in rows:
        out.append('| %s | `%s` | [%s](%s) | %s | [打开](%s) | [json](%s) |' % (
            (w.get('title') or '（无标题）').replace('|', '\\|'),
            w.get('kind', ''),
            note_base(w.get('note', '')),
            md_target(rel_to_widgets(w.get('note', ''))),
            w.get('uid') or '—',
            md_target(rel_to_widgets(w.get('html', ''))),
            md_target(rel_to_widgets(w.get('json', ''))),
        ))
    out.append('')
    return '\n'.join(out)

BOARD_NAME = '看板.html'


def board_path():
    """看板文件的位置（跟着 widgetsDir 走，所以自定义布局也能用）。"""
    return os.path.join(WIDGETS_DIR, BOARD_NAME)


BOARD_SHELL = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>交互组件看板</title>
<style>
  :root{
    --bg:#f6f7f9; --card:#fff; --text:#1c2330; --dim:#5b6577; --faint:#8a93a3;
    --border:#dfe3ea; --accent:#1769aa; --badge:#eef2f7;
    color-scheme:light;
  }
  @media (prefers-color-scheme: dark){
    :root{--bg:#0f1116; --card:#161a21; --text:#e6e9ef; --dim:#a6afc0; --faint:#7b8494;
          --border:#262b36; --accent:#7fb8e6; --badge:#1e242e; color-scheme:dark;}
  }
  *{box-sizing:border-box}
  body{margin:0; background:var(--bg); color:var(--text);
       font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;}
  header.top{padding:22px 22px 14px; border-bottom:1px solid var(--border); background:var(--card);}
  h1{margin:0; font-size:19px; letter-spacing:-.01em;}
  .sub{margin:4px 0 0; color:var(--dim); font-size:12.5px;}
  .note-gen{margin:2px 0 0; color:var(--faint); font-size:11.5px;}
  .toolbar{display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:12px 22px;
           border-bottom:1px solid var(--border); background:var(--card); position:sticky; top:0; z-index:5;}
  .toolbar input[type=search]{flex:1 1 220px; min-width:180px; padding:7px 10px; font:inherit;
      color:var(--text); background:var(--bg); border:1px solid var(--border); border-radius:8px;}
  .toolbar label{color:var(--dim); font-size:12.5px; display:inline-flex; gap:6px; align-items:center;}
  .toolbar select{padding:6px 8px; font:inherit; color:var(--text); background:var(--bg);
      border:1px solid var(--border); border-radius:8px;}
  .count{color:var(--faint); font-size:12.5px; margin-left:auto;}
  main{display:grid; gap:16px; padding:16px 22px 40px;
       grid-template-columns:repeat(auto-fill,minmax(340px,1fr));}
  .card{background:var(--card); border:1px solid var(--border); border-radius:12px; overflow:hidden;
        display:flex; flex-direction:column;}
  .card-head{padding:12px 14px 6px;}
  .card-title{margin:0; font-size:14.5px; font-weight:600;}
  .card-meta{display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:5px;
             color:var(--dim); font-size:12px;}
  .k{background:var(--badge); border-radius:999px; padding:1px 8px;
     font-family:ui-monospace,Menlo,monospace; font-size:11.5px;}
  .uid{font-family:ui-monospace,Menlo,monospace; font-size:11.5px; color:var(--faint);}
  .when{margin-left:auto; color:var(--faint); font-size:11.5px;}
  .card-actions{display:flex; gap:8px; align-items:center; padding:8px 14px 10px; flex-wrap:wrap;}
  .btn{display:inline-flex; align-items:center; padding:5px 11px; border-radius:8px; font-size:12.5px;
       text-decoration:none; border:1px solid var(--border); color:var(--text);}
  .btn.primary{background:var(--accent); border-color:var(--accent); color:#fff;}
  .btn.disabled{color:var(--faint); border-style:dashed;}
  .mini{font-size:12px; color:var(--accent); text-decoration:none; border-bottom:1px dotted var(--accent);}
  .card-body{border-top:1px solid var(--border); background:var(--bg);}
  .frame{display:block; width:100%; height:460px; border:0; background:#fff;}
  .frame.empty{padding:14px; color:var(--faint); font-size:12.5px; height:auto;}
  body.no-preview .frame{display:none;}
  body.no-preview .card-body{display:none;}
  .empty-state{color:var(--dim); padding:20px; grid-column:1/-1;}
  code{font-family:ui-monospace,Menlo,monospace; font-size:12px; background:var(--badge);
       padding:1px 5px; border-radius:5px;}
</style>
</head>
<body>
<header class="top">
  <h1>交互组件看板</h1>
  <p class="sub">__BOARD_SUB__</p>
  <p class="note-gen">本文件由 <code>make_widget.py --board</code> 生成，不要手改；改完 spec 重跑一次即可。</p>
</header>
<div class="toolbar">
  <input id="q" type="search" placeholder="按标题 / 渲染器 / 条目 / 源笔记筛选…" autocomplete="off">
  <label>排序
    <select id="sort">
      <option value="title">标题</option>
      <option value="when">更新时间 · 新 → 旧</option>
      <option value="kind">渲染器</option>
    </select>
  </label>
  <label><input id="prev" type="checkbox" checked> 显示预览</label>
  <span class="count" id="count">__BOARD_COUNT__</span>
</div>
<main id="grid">
__BOARD_CARDS__
</main>
<script>
/* 只做筛选 / 排序 / 预览开关。卡片是生成时渲染好的，所以禁掉 JS 也照样能看、能打开。 */
(function () {
  var grid = document.getElementById('grid');
  var cards = Array.prototype.slice.call(grid.querySelectorAll('.card'));
  var q = document.getElementById('q');
  var sort = document.getElementById('sort');
  var prev = document.getElementById('prev');
  var count = document.getElementById('count');
  function apply() {
    var needle = (q.value || '').trim().toLowerCase();
    var shown = 0;
    cards.forEach(function (c) {
      var hit = !needle || (c.getAttribute('data-hay') || '').indexOf(needle) >= 0;
      c.hidden = !hit;
      if (hit) shown++;
    });
    count.textContent = needle ? ('匹配 ' + shown + ' / ' + cards.length + ' 个组件')
                               : ('共 ' + cards.length + ' 个组件');
  }
  function resort() {
    var mode = sort.value;
    var sorted = cards.slice().sort(function (a, b) {
      var x = String(a.getAttribute('data-' + mode)), y = String(b.getAttribute('data-' + mode));
      if (mode === 'when') return y.localeCompare(x);                 /* 新 → 旧 */
      return x.localeCompare(y, 'zh') || String(a.getAttribute('data-title')).localeCompare(String(b.getAttribute('data-title')), 'zh');
    });
    sorted.forEach(function (c) { grid.appendChild(c); });
  }
  q.addEventListener('input', apply);
  sort.addEventListener('change', resort);
  prev.addEventListener('change', function () {
    document.body.classList.toggle('no-preview', !prev.checked);
  });
  apply();
})();
</script>
</body>
</html>
'''


def board_html_text(reg):
    """组件看板：一个**独立**的单文件 HTML 索引，双击就能在浏览器里看全部组件。

    与 Index.md 的分工：Index.md 给 Obsidian 与纯文本阅读器（表格 + 链接）；
    看板给"想一眼看到所有图"的场合（卡片 + 就地预览），不依赖 Obsidian、也不依赖 PI-Desktop 插件。
    预览用同目录兄弟文件做 iframe（相对路径 + loading="lazy"：滚到哪才加载哪），所以看板本身只有几十 KB。
    输出**不含时间戳**，与 Index.md 同样可以被 --check 判过期。
    """
    rows = sorted(reg['widgets'], key=lambda w: (w.get('note', ''), w.get('json', '')))
    notes = set(w.get('note') for w in rows)

    def safe_rel(p):
        try:
            return rel_to_widgets(p)
        except (ValueError, OSError):
            return ''

    def link(rel):
        return attr(quote(rel, safe='/._~-'))

    cards = []
    for w in rows:
        title = w.get('title') or '（无标题）'
        kind = w.get('kind') or '?'
        uid = w.get('uid') or '—'
        note = note_base(w.get('note', '')) or '—'
        note_rel = safe_rel(w.get('note', '')) if w.get('note') else ''
        html_rel = safe_rel(w.get('html', '')) if w.get('html') else ''
        json_rel = safe_rel(w.get('json', '')) if w.get('json') else ''
        when = w.get('updated') or w.get('created') or ''
        hay = ' '.join([title, kind, uid, note]).lower()
        open_btn = ('<a class="btn primary" href="%s" target="_blank" rel="noopener">打开</a>' % link(html_rel)
                    if html_rel else '<span class="btn disabled">文件缺失</span>')
        srcs = []
        if note_rel:
            srcs.append('<a class="mini" href="%s">源笔记</a>' % link(note_rel))
        if json_rel:
            srcs.append('<a class="mini" href="%s">源 spec</a>' % link(json_rel))
        if html_rel:
            preview = ('<iframe class="frame" loading="lazy" title="预览：%s" src="%s"></iframe>'
                       % (attr(title), link(html_rel)))
        else:
            preview = ('<div class="frame empty">派生 HTML 不在：跑一次 <code>--check</code> '
                       '看是哪一个组件过期了</div>')
        cards.append(
            '    <article class="card" data-hay="%s" data-kind="%s" data-title="%s" data-when="%s">\n'
            '      <div class="card-head">\n'
            '        <h3 class="card-title">%s</h3>\n'
            '        <div class="card-meta"><span class="k">%s</span><span class="uid">%s</span>'
            '<span class="note">%s</span><span class="when">%s</span></div>\n'
            '      </div>\n'
            '      <div class="card-actions">%s%s</div>\n'
            '      <div class="card-body">%s</div>\n'
            '    </article>'
            % (attr(hay), attr(kind), attr(title), attr(when),
               attr(title), attr(kind), attr(uid), attr(note), attr(when),
               open_btn, ''.join(srcs), preview))
    if cards:
        body = '\n'.join(cards)
        count_text = '共 %d 个组件' % len(rows)
    else:
        body = ('    <p class="empty-state">还没有组件。先跑 <code>make_widget.py --list</code> 看有哪些渲染器，'
                '再用 <code>--spec &lt;kind&gt;</code> 取骨架建一个。</p>')
        count_text = '共 0 个组件'
    return (BOARD_SHELL
            .replace('__BOARD_SUB__', attr('%s · %d 个组件 · %d 篇源笔记'
                                           % (os.path.basename(VAULT), len(rows), len(notes))))
            .replace('__BOARD_COUNT__', count_text)
            .replace('__BOARD_CARDS__', body))

def rel_to_widgets(p):
    """把 vault 相对路径转成"相对组件目录"的路径（看板 iframe 与 Index.md 链接都用它）。

    两边都要 realpath：vault_path() 返回的是**解析过符号链接**的真路径，而 WIDGETS_DIR 可能没解析
    （macOS 的 /var → /private/var、/tmp 之类）。只解析一边时前缀不同，os.path.relpath 会一路往上绕成
    `../../../../private/var/...` —— 看板预览就是这么坏掉的（真踩过）。顺带拒绝越出组件目录的结果。
    """
    base = os.path.realpath(WIDGETS_DIR)
    target = vault_path(p)                      # vault_path 内部已经 realpath
    rel = os.path.relpath(target, base).replace(os.sep, '/')
    return rel


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# --------------------------------------------------------------------------- #
# 命令
# --------------------------------------------------------------------------- #
def cmd_list():
    reg = load_registry()
    print('交互式知识组件 · 渲染器目录（写作规范：Maps/_tools/INTERACTIVE-AUTHORING.md）')
    print('=' * 68)
    for k in KIND_ORDER:
        info = KINDS[k]
        print('  %-10s %s' % (k, info['purpose']))
        print('             %s' % info['use'])
    print('')
    rows = sorted(reg['widgets'], key=lambda w: (w.get('note', ''), w.get('json', '')))
    print('已有组件：%d 个' % len(rows))
    for w in rows:
        print('  [%s] %s（%s）' % (w.get('kind', ''), w.get('title', ''), w.get('uid') or '未关联条目'))
        print('      %s → %s' % (w.get('note', ''), w.get('html', '')))
    if not rows:
        print('  （还没有组件）')
    print('')
    print('下一步：')
    print('  python3 Maps/_tools/make_widget.py --spec plot      # 取某渲染器的 spec 骨架')
    print('  python3 Maps/_tools/make_widget.py new --note "Maps/Notes/xxx.md" --spec /path/spec.json')
    print('  python3 Maps/_tools/make_widget.py --check           # 一致性 + 过期检测')
    return 0


def cmd_spec(arg):
    kinds = KIND_ORDER if arg in (None, '', 'all') else [arg]
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        sys.exit('[错误] 未知渲染器 %s；可用：%s' % ('、'.join(bad), '/'.join(KIND_ORDER)))
    for i, k in enumerate(kinds):
        info = KINDS[k]
        if i:
            print('')
        print('%s —— %s' % (k, info['purpose']))
        print('  适用：%s' % info['use'])
        print('  必填：%s' % info['require'])
        print('  教学元数据：teaching.question / sourceSection / controlEffect / visualEvidence 必填；')
        print('             teaching.controls 可列 semantic 控件，teaching.cameraControls 可声明相机控件；')
        print('             控件必须进入主绘制数据（dynamicRenderer=true 时由动态门禁核验）')
        if k == 'plotly':
            # plotly 的表达式是"整体求值一次"，没有逐点 x/i；逐点靠生成器 {"by": …}
            print('  作用域：vars 的键 / controls 的 key，以及 rand, randn, phi, quantile, sum, mean, sd, clamp,')
            print('          min, max, abs, exp, log, sqrt, pow, sin, cos, floor, round, ifelse, fmt')
            print('          （"=表达式" 整体求值一次，没有逐点 x / i；要按控件生成数组用生成器 '
                  '{"by": …, "n": …}：')
            print('           一维作用域 i/n（x = i）、二维 i/j/n/m（x = i、y = j），可带 "vars" 先定义名字）')
        else:
            grid_scope = ('x、y（网格的两个自变量）' if k in ('contour', 'vector', 'descent', 'surface3d')
                          else 'x（下标，不是横轴；图上的 x 是逐点变量）')
            print('  作用域：vars 的键 / controls 的 key / %s / i / item，以及 rand, randn, phi, quantile,' % grid_scope)
            print('          sum, mean, sd, clamp, min, max, abs, exp, log, sqrt, pow, sin, cos, floor, round, ifelse, fmt')
        state_vars = INTERNAL_VARS.get(k)
        if state_vars:
            print('  读数变量：%s' % state_vars)
        print('')
        print(json.dumps(info['min'], ensure_ascii=False, indent=1))
    return 0


def cmd_new(a):
    note = a.note.replace('\\', '/').strip()
    while note.startswith('./'):
        note = note[2:]
    if not is_vault_rel(note):
        sys.exit('[错误] --note 必须位于当前 vault 内，不能使用绝对路径或 ../：%s' % a.note)
    if not note.endswith('.md'):
        sys.exit('[错误] --note 必须是 vault 内的 .md 笔记路径：%r' % a.note)
    if not os.path.exists(vault_path(note)):
        sys.exit('[错误] 源笔记不存在：%s' % note)
    if note.startswith('Books/'):
        sys.exit('[错误] 不能挂到 Books/ 原文上：原文只读，组件属于派生层')
    if not os.path.exists(a.spec):
        sys.exit('[错误] 找不到 spec 文件：%s' % a.spec)
    try:
        spec = load_json(read_text(a.spec))
    except (ValueError, OSError) as e:
        sys.exit('[错误] spec 不是合法 JSON：%s' % e)
    if not isinstance(spec, dict):
        sys.exit('[错误] spec 顶层必须是对象')
    if spec.get('note') and spec['note'].replace('\\', '/') != note:
        sys.exit('[错误] spec.note（%s）与 --note（%s）不一致' % (spec['note'], note))
    spec['note'] = note
    uid = a.uid or spec.get('uid') or note_uid(note)
    spec['uid'] = uid or ''
    if spec['uid']:
        uids = known_uids()
        if uids and spec['uid'] not in uids:
            sys.exit('[错误] uid 未在 notes-index.json 登记：%s' % spec['uid'])
    err, warn = validate_spec(spec)
    if err:
        for e in err:
            print('  [错误] ' + e, file=sys.stderr)
        print('[错误] spec 校验未通过：%d 个错误，未生成任何文件' % len(err), file=sys.stderr)
        return 1
    for w in warn:
        print('  [提醒] ' + w)
    slug = slugify(a.slug or spec.get('slug') or spec['title'])
    spec['slug'] = slug
    json_path = os.path.join(WIDGETS_DIR, '%s-%s.json' % (note_base(note), slug))
    html_path = os.path.join(WIDGETS_DIR, '%s-%s.html' % (note_base(note), slug))
    reg = load_registry()
    old = [w for w in reg['widgets'] if w.get('html') == rel_vault(html_path)]
    exists = [p for p in (json_path, html_path) if os.path.exists(p)]
    if exists and not a.force:
        sys.exit('[错误] 目标已存在：%s —— 不覆盖（确需重建请显式加 --force）' % rel_vault(exists[0]))
    json_text = json.dumps(spec, ensure_ascii=False, indent=1, allow_nan=False) + '\n'
    html_text = build_html(spec, note, rel_vault(json_path))
    entry = {
        'json': rel_vault(json_path),
        'html': rel_vault(html_path),
        'note': note,
        'uid': spec['uid'],
        'kind': spec['kind'],
        'title': spec['title'],
        'spec_sha256': sha_text(json_text),
        'html_sha256': sha_text(html_text),
        'created': (old[0].get('created') if old else today()),
        'updated': today(),
    }
    reg['widgets'] = [w for w in reg['widgets'] if w.get('html') != entry['html']] + [entry]
    reg['widgets'].sort(key=lambda w: (w['note'], w['json']))
    try:
        write_transaction([(json_path, json_text), (html_path, html_text),
                           (REGISTRY, json.dumps(reg, ensure_ascii=False, indent=1, allow_nan=False) + '\n')])
    except (OSError, ValueError) as e:
        print('[错误] 写入失败（已尝试回滚）：%s' % e, file=sys.stderr)
        return 1
    print('已生成组件：%s（%s%s）' % (spec['title'], spec['kind'], '，%s' % uid if uid else ''))
    print('  源 spec   %s' % entry['json'])
    print('  派生 HTML %s' % entry['html'])
    if not a.no_index and cmd_index(quiet=True) != 0:
        print(('[错误] 组件已登记成功，但 %s 重建失败'
               '（不是登记失败，不要重跑登记）') % rel_vault(INDEX_MD), file=sys.stderr)
        return 2
    link = os.path.relpath(vault_path(entry['html']), os.path.dirname(vault_path(note))).replace(os.sep, '/')
    print('笔记正文里加这两行（普通 Markdown 相对链接，基准是笔记自己所在的目录）：')
    print('  > [!interactive] %s' % spec['title'])
    print('  > [打开交互组件](%s)' % md_target(link))
    return 0


def cmd_check():
    reg = load_registry()
    errs, warns = [], []
    uids = known_uids()
    seen = set()
    sizes = []          # (页面相对路径, 页面字节, bundle 键或 None, 内联库字节)
    vendored = {}       # bundle 键 → (相对路径, 实测 sha, 登记 sha, 字节)
    for w in reg['widgets']:
        html_rel, json_rel, note_rel = w.get('html', ''), w.get('json', ''), w.get('note', '')
        tag = html_rel or json_rel or '（注册项缺路径）'
        if not html_rel or not json_rel:
            errs.append('%s：注册项缺少 json/html 路径' % tag)
            continue
        seen.add(html_rel)
        seen.add(json_rel)
        for rel in (json_rel, html_rel):
            if not os.path.exists(vault_path(rel)):
                errs.append('%s：文件不存在' % rel)
        if not note_rel or not os.path.exists(vault_path(note_rel)):
            errs.append('%s：源笔记不存在（%s）' % (tag, note_rel or '未填'))
        if not (os.path.exists(vault_path(json_rel)) and os.path.exists(vault_path(html_rel))):
            continue
        try:
            spec = load_json(read_text(vault_path(json_rel)))
        except (ValueError, OSError) as e:
            errs.append('%s：源 spec 不是合法 JSON：%s' % (json_rel, e))
            continue
        e2, w2 = validate_spec(spec)
        errs += ['%s：%s' % (json_rel, x) for x in e2]
        warns += ['%s：%s' % (json_rel, x) for x in w2]
        if not isinstance(spec, dict):
            continue
        for key in ('uid', 'kind', 'title'):
            if spec.get(key) != w.get(key):
                errs.append('%s：spec.%s 与注册表镜像不一致' % (json_rel, key))
        spec_sha = file_sha(vault_path(json_rel))
        html_sha = file_sha(vault_path(html_rel))
        if w.get('spec_sha256') != spec_sha:
            errs.append('%s：注册表 spec_sha256 与源 spec 不一致（源文件被手工改过）' % json_rel)
        if w.get('html_sha256') != html_sha:
            errs.append('%s：注册表 html_sha256 与派生产物不一致（HTML 被手工改过或待重建）' % html_rel)
        if (spec.get('note') or '').replace('\\', '/') != note_rel:
            errs.append('%s：spec.note（%s）与注册表 note（%s）不一致' % (json_rel, spec.get('note'), note_rel))
        if os.path.exists(vault_path(note_rel)):
            rebuilt = build_html(spec, note_rel, json_rel)
            if sha_text(rebuilt) != file_sha(vault_path(html_rel)):
                errs.append('%s：派生 HTML 与源 spec 不同步（已过期或被手工编辑）'
                            '——重建：make_widget.py new --note "%s" --spec "%s" --force'
                            % (html_rel, note_rel, json_rel))
        if spec.get('uid') and uids and spec['uid'] not in uids:
            errs.append('%s：uid %s 不在 notes-index.json 里（笔记未登记，或 uid 写错）'
                        % (json_rel, spec['uid']))
        page_bytes = os.path.getsize(vault_path(html_rel))
        if spec.get('kind') == 'plotly':
            key = plotly_bundle_key(spec)
            lib = plotly_bundle_path(key)
            lib_bytes = os.path.getsize(lib) if os.path.exists(lib) else 0
            if os.path.exists(lib):
                actual, booked = file_sha(lib), vendor_sha_from_readme(key)
                vendored[key] = (plotly_bundle_rel(key), actual, booked, lib_bytes)
                if booked is None:
                    warns.append('%s：%s 里没有登记 %s 的 sha256，没法证明内联的库没被换过'
                                 % (html_rel, PLOTLY_README_REL, PLOTLY_BUNDLES[key][0]))
                elif booked != actual:
                    errs.append('%s：%s 的 sha256 实测 %s，README 登记 %s——不一致，先核对是不是'
                                '故意升级（升级要同时改 vendor/README.md 与这份实测值）'
                                % (html_rel, plotly_bundle_rel(key), actual[:16], booked[:16]))
            sizes.append((html_rel, page_bytes, key, lib_bytes))
        else:
            sizes.append((html_rel, page_bytes, None, 0))
            if page_bytes > 1_500_000:
                warns.append('%s：页面 %.2f MB（非 plotly 页不该这么大）——确认没误内联大文件'
                             % (html_rel, page_bytes / 1048576.0))
    if os.path.isdir(WIDGETS_DIR):
        for fn in sorted(os.listdir(WIDGETS_DIR)):
            if not fn.endswith(('.json', '.html')):
                continue
            if fn == BOARD_NAME:
                continue          # 看板是本工具生成的派生索引（--board），不是组件
            rel = rel_vault(os.path.join(WIDGETS_DIR, fn))
            if rel not in seen:
                warns.append('%s：存在于磁盘但未登记（不是本工具生成的？）' % rel)
    if reg['widgets']:
        want = index_md_text(reg)
        if not os.path.exists(INDEX_MD):
            errs.append('缺少 %s —— 跑一下 --index' % rel_vault(INDEX_MD))
        elif read_text(INDEX_MD) != want:
            errs.append('%s 不是最新 —— 跑一下 --index' % rel_vault(INDEX_MD))
    # 看板是**可选**的派生索引：不存在不报错（老项目没跑过 --board），存在但不新鲜才提醒
    if os.path.exists(board_path()):
        try:
            if read_text(board_path()) != board_html_text(reg):
                warns.append('%s 不是最新 —— 跑一下 --board 重建（刚改过组件/注册表时正常）'
                             % rel_vault(board_path()))
        except (OSError, ValueError) as e:
            warns.append('%s 读不了：%s' % (rel_vault(board_path()), e))
    # 体积报告：plotly 页要一眼看出「页面里库占了多少」，sha 校验要能证明内联的就是登记的那份
    for key in sorted(vendored):
        rel, actual, booked, lib_bytes = vendored[key]
        state = ('与 %s 登记一致' % PLOTLY_README_REL) if booked == actual else (
            '未登记' if booked is None else '与登记不一致')
        print('  vendor：%s（%s，%s）sha256 %s… %s'
              % (rel, PLOTLY_BUNDLES[key][1], size_text(lib_bytes), actual[:16], state))
    plotly_pages = [s for s in sizes if s[2]]
    if plotly_pages:
        for rel, page_bytes, key, lib_bytes in sorted(plotly_pages):
            share = (100.0 * lib_bytes / page_bytes) if page_bytes else 0.0
            print('  体积：%s 页面 %s，其中内联 Plotly（bundle=%s）%s（%.0f%%）'
                  % (rel, size_text(page_bytes), key, size_text(lib_bytes), share))
    if sizes:
        total = sum(s[1] for s in sizes)
        biggest = max(sizes, key=lambda s: s[1])
        print('  体积：%d 个页面共 %s；最大的 %s（%s）'
              % (len(sizes), size_text(total), biggest[0], size_text(biggest[1])))
    for e in errs:
        print('  [错误] ' + e, file=sys.stderr)
    for w in warns:
        print('  [提醒] ' + w)
    print('校验：%d 个组件 / %d 篇笔记…错误 %d、提醒 %d'
          % (len(reg['widgets']), len(set(x.get('note') for x in reg['widgets'])), len(errs), len(warns)))
    print('布局：root=%s · tools=%s · widgets=%s · config=%s'
          % (LAYOUT['root'], LAYOUT['toolsDir'], LAYOUT['widgetsDir'], LAYOUT['configFile'] or '无'))
    return 0 if not errs else 1


def cmd_layout():
    """只打印一行 JSON（机器可读）：已解析布局。root 是绝对路径，其余是相对 root 的 vault 相对路径。"""
    print(json.dumps(LAYOUT, ensure_ascii=False, allow_nan=False))
    return 0


def cmd_board(quiet=False):
    """写出组件看板（独立 HTML 索引）。幂等：内容没变就不落盘。"""
    try:
        reg = load_registry()
        vault_path(rel_vault(board_path()))
        text = board_html_text(reg)
        if os.path.exists(board_path()) and read_text(board_path()) == text:
            if not quiet:
                print('%s 已是最新（%d 个组件）' % (rel_vault(board_path()), len(reg['widgets'])))
            return 0
        write_text(board_path(), text)
    except (OSError, ValueError) as e:
        print('[错误] 写 %s 失败：%s' % (rel_vault(board_path()), e), file=sys.stderr)
        return 1
    if not quiet:
        print('已重建 %s（%d 个组件）' % (rel_vault(board_path()), len(reg['widgets'])))
    return 0


def cmd_index(quiet=False):
    """重建两份派生索引：Index.md（给 Obsidian / 纯文本）与看板（给浏览器）。"""
    try:
        reg = load_registry()
        vault_path(rel_vault(INDEX_MD))
        text = index_md_text(reg)
        if os.path.exists(INDEX_MD) and read_text(INDEX_MD) == text:
            if not quiet:
                print('%s 已是最新（%d 个组件）' % (rel_vault(INDEX_MD), len(reg['widgets'])))
        else:
            write_text(INDEX_MD, text)
            if not quiet:
                print('已重建 %s（%d 个组件）' % (rel_vault(INDEX_MD), len(reg['widgets'])))
    except (OSError, ValueError) as e:
        print('[错误] 写 %s 失败：%s' % (rel_vault(INDEX_MD), e), file=sys.stderr)
        return 1
    return cmd_board(quiet=quiet)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog='make_widget.py',
        description='交互式知识组件生成器：spec.json → 单文件 HTML（笔记里只放链接，不放代码）')
    ap.add_argument('--list', action='store_true', help='渲染器目录与已有组件')
    ap.add_argument('--spec', nargs='?', const='all', metavar='KIND',
                    help='打印某渲染器的 spec 骨架（含必填 teaching.question/sourceSection/controlEffect/visualEvidence；plot|bars|scatter|histogram|heatmap|timeline|tree|box|ecdf|qq|contour|vector|matrix|regression|pca|descent|surface3d|treefit|custom|plotly|all）')
    ap.add_argument('--check', action='store_true', help='校验注册表↔磁盘↔源笔记，并检测 HTML 是否过期')
    ap.add_argument('--index', action='store_true', help='重建两份派生索引：Index.md + 看板.html')
    ap.add_argument('--board', action='store_true',
                    help='只重建组件看板（<widgetsDir>/看板.html：独立单文件索引，卡片 + 就地预览）')
    ap.add_argument('--layout', action='store_true',
                    help='打印已解析布局（一行 JSON：root / toolsDir / widgetsDir / registry / indexMd / notesIndex）')
    ap.add_argument('--root', metavar='DIR',
                    help='项目根（默认按本脚本所在位置推导）；<root>/widgets.config.json 可改写目录布局')
    ap.add_argument('--vault', help=argparse.SUPPRESS)  # --root 的隐藏别名（测试夹具也在用）
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('new', help='从 spec 生成组件并登记（不覆盖已存在文件）')
    p.add_argument('--note', required=True, help='源笔记，相对 vault 根，如 Maps/Notes/有效久期.md')
    p.add_argument('--spec', required=True, metavar='FILE', help='spec JSON 文件路径')
    p.add_argument('--slug', help='文件名片段（默认取 title）')
    p.add_argument('--uid', help='关联的二级条目 uid（默认从 notes-index.json 反查）')
    p.add_argument('--force', action='store_true', help='重建已存在的组件（覆盖派生产物）')
    p.add_argument('--no-index', action='store_true', help='跳过 Index.md 重建')
    a = ap.parse_args(argv)
    configure_layout(a.root or a.vault or VAULT)   # 先按默认布局落一遍，再套 <root>/widgets.config.json
    if a.cmd == 'new':
        return cmd_new(a)
    if a.layout:
        return cmd_layout()
    if a.check:
        return cmd_check()
    if a.index:
        return cmd_index()
    if a.board:
        return cmd_board()
    if a.list:
        return cmd_list()
    if a.spec is not None:
        return cmd_spec(a.spec)
    ap.print_help()
    return 0
    ap.print_help()
    return 0


if __name__ == '__main__':
    sys.exit(main())