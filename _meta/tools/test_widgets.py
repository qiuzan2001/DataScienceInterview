#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_widget.py 的临时 vault 回归测试（标准库，不碰真实 vault）。

用法：python3 Maps/_tools/test_widgets.py
覆盖：14 个内置渲染器的最小 spec、登记/Index、HTML 全内联、过期检测、不可覆盖、
显式 --force、uid 反查、custom 的零外部依赖约束，以及组件看板（--board / --index）。
"""
import copy
import importlib.util
import json
import os
import re
import tempfile
import unittest
import urllib.parse
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import shutil
import contextlib
import io

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location('make_widget_under_test', os.path.join(HERE, 'make_widget.py'))
mw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mw)


class WidgetToolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('PI_SCRATCH_DIR'))
        self.root = self.tmp.name
        os.makedirs(os.path.join(self.root, 'Maps', '_tools'))
        os.makedirs(os.path.join(self.root, 'Maps', 'Notes'))
        self.note = 'Maps/Notes/样本.md'
        self.write(self.note, '# 样本\n\n这篇笔记已经写清了交互组件要验证的结论。\n')
        self.write('Maps/_tools/notes-index.json', json.dumps({
            'vault': 'test-vault', '_说明': 'test fixture',
            'notes': [{'file': self.note, 'uid': 'N0001.01', 'node': 'N0001', 'sub': 0,
                       'sub_title': '样本', 'title': '样本'}],
        }, ensure_ascii=False, indent=1) + '\n')
        mw.configure_vault(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, rel):
        return os.path.join(self.root, rel.replace('/', os.sep))

    def write(self, rel, text):
        path = self.path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def input_spec(self, kind, suffix=''):
        data = copy.deepcopy(mw.KINDS[kind]['min'])
        data['title'] = '测试：%s%s' % (kind, suffix)
        data['uid'] = 'N0001.01'
        # 这里测试的是每个 kind 能否生成自包含 HTML，不测试教学语义；
        # 明确声明没有 semantic 控件，避免把骨架里的占位控件作用误当成合格 spec。
        data['teaching'] = {
            'question': '测试：组件能否生成单文件页面',
            'sourceSection': 'test fixture',
            'controlEffect': '本测试不声明 semantic 控件',
            'visualEvidence': '页面存在并内联运行时',
            'controls': []
        }
        path = self.path('input-%s%s.json' % (kind, suffix))
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write('\n')
        return path

    def new(self, spec, slug, force=False, uid='N0001.01'):
        return mw.cmd_new(SimpleNamespace(note=self.note, spec=spec, slug=slug, uid=uid,
                                          force=force, no_index=False))

    def test_all_builtin_kinds_generate_self_contained_html(self):
        for kind in mw.KIND_ORDER:
            with self.subTest(kind=kind):
                spec = self.input_spec(kind)
                self.assertEqual(self.new(spec, kind), 0)
                html = self.path('Maps/_widgets/样本-%s.html' % kind)
                source = self.path('Maps/_widgets/样本-%s.json' % kind)
                self.assertTrue(os.path.exists(html))
                self.assertTrue(os.path.exists(source))
                with open(html, encoding='utf-8') as f:
                    text = f.read()
                self.assertIn('window.WG', text)
                self.assertIn('id="wg-spec"', text)
                self.assertNotIn('src="http', text)
                self.assertNotIn('href="http', text)
                self.assertNotIn('/*__WIDGETSJS__*/', text)
                self.assertNotIn('/*__WIDGETSCSS__*/', text)
        self.assertEqual(mw.cmd_check(), 0)
        self.assertTrue(os.path.exists(self.path('Maps/_widgets/Index.md')))
        with open(self.path('Maps/_widgets/Index.md'), encoding='utf-8') as f:
            index = f.read()
        self.assertIn('共 %d 个组件' % len(mw.KIND_ORDER), index)
        for kind in mw.KIND_ORDER:
            self.assertIn('`%s`' % kind, index)

    def test_conflict_force_and_outdated_detection(self):
        spec = self.input_spec('plot')
        self.assertEqual(self.new(spec, '曲线'), 0)
        with self.assertRaises(SystemExit) as caught:
            self.new(spec, '曲线')
        self.assertIn('目标已存在', str(caught.exception))
        source = self.path('Maps/_widgets/样本-曲线.json')
        with open(source, encoding='utf-8') as f:
            changed = json.load(f)
        changed['title'] = '手工改过的 source spec'
        with open(source, 'w', encoding='utf-8') as f:
            json.dump(changed, f, ensure_ascii=False, indent=1)
            f.write('\n')
        self.assertEqual(mw.cmd_check(), 1)       # hash 与 HTML 都应检出过期
        self.assertEqual(self.new(source, '曲线', force=True), 0)
        self.assertEqual(mw.cmd_check(), 0)

    def test_explicit_unknown_uid_is_rejected(self):
        spec = self.input_spec('bars')
        with self.assertRaises(SystemExit) as caught:
            self.new(spec, '未知-uid', uid='N9999.99')
        self.assertIn('uid 未在 notes-index.json 登记', str(caught.exception))
        self.assertFalse(os.path.exists(self.path('Maps/_widgets/样本-未知-uid.html')))

    def test_custom_rejects_external_dependency(self):
        spec = self.input_spec('custom')
        with open(spec, encoding='utf-8') as f:
            data = json.load(f)
        data['custom']['js'] = "fetch('https://example.test/data')"
        with open(spec, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
            f.write('\n')
        self.assertEqual(self.new(spec, '外链'), 1)
        self.assertFalse(os.path.exists(self.path('Maps/_widgets/样本-外链.html')))

    def test_distribution_kinds_validation_and_generation(self):
        """box / ecdf / qq：骨架自身必须合法、能生成；缺 sample/values、非法 dist、
        越界 maxPoints 必须在生成阶段就报 [错误]，而不是画错。"""
        for kind in ('box', 'ecdf', 'qq'):
            with self.subTest(kind=kind):
                self.assertEqual(mw.validate_spec(copy.deepcopy(mw.KINDS[kind]['min']))[0], [])
                self.assertEqual(self.new(self.input_spec(kind), kind), 0)
        for kind in ('box', 'ecdf', 'qq'):
            with self.subTest(kind=kind, missing='sample/values'):
                bad = copy.deepcopy(mw.KINDS[kind]['min'])
                bad[kind].pop('sample', None)
                bad[kind].pop('values', None)
                self.assertTrue(any('sample' in e or 'values' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['qq']['min'])
        bad['qq']['dist'] = 'cauchy'
        self.assertTrue(any('dist' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['ecdf']['min'])
        bad['ecdf']['maxPoints'] = 99999
        self.assertTrue(any('maxPoints' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['qq']['min'])
        bad['qq']['maxPoints'] = 1
        self.assertTrue(any('maxPoints' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['box']['min'])
        bad['box'] = 'not-an-object'
        self.assertTrue(any('box' in e for e in mw.validate_spec(bad)[0]))

    def test_field_and_matrix_kinds_validation_and_generation(self):
        """contour / vector / matrix：骨架自身必须合法、能生成；缺表达式、越界 levels/points、
        非 2×2 矩阵、非法 samples/scale 必须在生成阶段就报 [错误]，而不是画错。"""
        for kind in ('contour', 'vector', 'matrix'):
            with self.subTest(kind=kind):
                self.assertEqual(mw.validate_spec(copy.deepcopy(mw.KINDS[kind]['min']))[0], [])
                self.assertEqual(self.new(self.input_spec(kind), kind), 0)
        # contour：expr 缺、levels 越界、缺 y 轴、grid points 越界
        bad = copy.deepcopy(mw.KINDS['contour']['min'])
        bad['contour'].pop('expr')
        self.assertTrue(any('expr' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['contour']['min'])
        bad['contour']['levels'] = 0
        self.assertTrue(any('levels' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['contour']['min'])
        bad.pop('y')
        self.assertTrue(any('spec.y' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['contour']['min'])
        bad['x']['points'] = 500
        self.assertTrue(any('points' in e for e in mw.validate_spec(bad)[0]))
        # vector：u/v 缺、scale 未知
        bad = copy.deepcopy(mw.KINDS['vector']['min'])
        bad['vector']['v'] = ''
        self.assertTrue(any('vector.v' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['vector']['min'])
        bad['vector']['scale'] = 'log'
        self.assertTrue(any('scale' in e for e in mw.validate_spec(bad)[0]))
        # matrix：非 2×2、非法 samples、非法 bind
        bad = copy.deepcopy(mw.KINDS['matrix']['min'])
        bad['matrix']['values'] = [[1, 2, 3], [4, 5, 6]]
        self.assertTrue(any('2×2' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['matrix']['min'])
        bad['matrix']['samples'] = [[1]]
        self.assertTrue(any('samples' in e for e in mw.validate_spec(bad)[0]))
        bad = copy.deepcopy(mw.KINDS['matrix']['min'])
        bad['matrix']['bind'] = {'p': [0, 2]}
        self.assertTrue(any('bind' in e for e in mw.validate_spec(bad)[0]))
        # aspect（等比例坐标轴）：非法值在生成阶段就必须报 [错误]；显式值与默认值都要能通过
        for kind, dflt, other in (('contour', 'auto', 'equal'), ('vector', 'auto', 'equal'),
                                  ('matrix', 'equal', 'auto')):
            with self.subTest(kind=kind, aspect='非法值'):
                bad = copy.deepcopy(mw.KINDS[kind]['min'])
                bad['aspect'] = 'stretch'
                self.assertTrue(any('aspect' in e for e in mw.validate_spec(bad)[0]),
                                'aspect 非法值必须报 [错误]，不能静默忽略')
            with self.subTest(kind=kind, aspect='显式覆盖'):
                ok = copy.deepcopy(mw.KINDS[kind]['min'])
                ok['aspect'] = other
                self.assertEqual(mw.validate_spec(ok)[0], [])
            with self.subTest(kind=kind, aspect='默认值'):
                self.assertEqual(mw.ASPECT_DEFAULT[kind], dflt, '渲染器默认值')
                self.assertEqual(mw.KINDS[kind]['min']['aspect'], dflt, '骨架里写死的默认值')
        for wrong in (1, True, ['equal'], ''):
            with self.subTest(aspect=repr(wrong)):
                bad = copy.deepcopy(mw.KINDS['matrix']['min'])
                bad['aspect'] = wrong
                self.assertTrue(any('aspect' in e for e in mw.validate_spec(bad)[0]),
                                'aspect 必须是非空字符串 "auto"/"equal"')
        bad = copy.deepcopy(mw.KINDS['plot']['min'])
        bad['aspect'] = 'equal'
        errs, warns = mw.validate_spec(bad)
        self.assertEqual(errs, [], 'aspect 用在不支持的渲染器上不是错误（向后兼容）')
        self.assertTrue(any('aspect' in w for w in warns), 'aspect 用在不支持的渲染器上要提醒，不静默吞掉')
        self.assertEqual(mw.cmd_check(), 0)

    def test_hidden_vault_cli_switch(self):
        # 保证测试夹具走的正是对外脚本的参数解析路径，而不是仅直接调函数。
        self.assertEqual(mw.main(['--vault', self.root, '--index']), 0)
        self.assertTrue(os.path.exists(self.path('Maps/_widgets/Index.md')))

    def test_nonfinite_everywhere(self):
        for literal in ('NaN', 'Infinity', '-Infinity', '1e999', '-1e999'):
            with self.subTest(literal=literal):
                with self.assertRaises(ValueError):
                    mw.load_json('{"nested": [ {"value": %s} ]}' % literal)
                source = self.input_spec('tree')
                data = mw.read_text(source).replace('"u": 1.1', '"u": ' + literal)
                with open(source, 'w') as f:
                    f.write(data)
                with self.assertRaises(SystemExit):
                    self.new(source, 'bad')
        spec = copy.deepcopy(mw.KINDS['tree']['min'])
        spec['tree']['root']['extra'] = float('nan')
        self.assertTrue(mw.validate_spec(spec)[0])

    def test_registry_duplicates_and_mirrors(self):
        self.new(self.input_spec('bars'), 'bars')
        original = mw.load_registry()
        for key in ('uid', 'kind', 'title'):
            reg = copy.deepcopy(original)
            reg['widgets'][0][key] = 'wrong'
            self.write('Maps/_tools/widgets-index.json', json.dumps(reg))
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mw.cmd_check(), 1)
            self.assertIn('spec.' + key, err.getvalue())
        reg = copy.deepcopy(original)
        reg['widgets'].append(copy.deepcopy(reg['widgets'][0]))
        self.write('Maps/_tools/widgets-index.json', json.dumps(reg))
        with self.assertRaises(SystemExit) as err:
            mw.cmd_check()
        self.assertIn('重复', str(err.exception))

    def test_malicious_registry_cli_is_friendly(self):
        for rel in ('../../outside.json', '/tmp/outside.json', 'Maps/../Books/x.json', None):
            reg = {'widgets': [{'note': self.note, 'json': rel, 'html': 'Maps/_widgets/x.html'}]}
            self.write('Maps/_tools/widgets-index.json', json.dumps(reg))
            result = subprocess.run([os.sys.executable, os.path.join(HERE, 'make_widget.py'),
                                     '--vault', self.root, '--check'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn('[错误]', result.stderr)
            self.assertNotIn('Traceback', result.stderr)

    def test_symlink_and_books_paths_rejected(self):
        self.write('Books/original.md', 'original')
        os.symlink(self.path('Books'), self.path('alias'))
        os.symlink(os.path.dirname(self.root), self.path('outside'))
        for rel in ('alias/original.md', 'outside/escape.md', 'Maps/../Books/original.md',
                    self.path(self.note), 'Books/original.md'):
            self.assertFalse(mw.is_vault_rel(rel), rel)
        os.symlink(self.path('Books'), self.path('Maps/_widgets'))
        self.assertEqual(self.new(self.input_spec('bars'), 'escape'), 1)
        self.assertEqual(mw.read_text(self.path('Books/original.md')), 'original')
        self.assertEqual(os.listdir(self.path('Books')), ['original.md'])

    def test_force_write_failure_rolls_back_all_files(self):
        source = self.input_spec('bars')
        self.new(source, 'rollback')
        paths = [self.path('Maps/_widgets/样本-rollback.' + ext) for ext in ('json', 'html')]
        paths += [mw.REGISTRY, mw.INDEX_MD]
        before = {p: mw.read_text(p) for p in paths}
        data = json.loads(mw.read_text(source)); data['title'] = 'changed'
        with open(source, 'w') as f:
            json.dump(data, f)
        real_replace = os.replace
        for fail_at in (1, 2, 3):
            calls = [0]
            def fail_once(src, dst):
                calls[0] += 1
                if calls[0] == fail_at:
                    raise OSError('injected replace failure')
                return real_replace(src, dst)
            with patch.object(mw.os, 'replace', side_effect=fail_once):
                self.assertEqual(self.new(source, 'rollback', force=True), 1)
            self.assertEqual({p: mw.read_text(p) for p in paths}, before)
            self.assertFalse(any(n.startswith('.widget-') for n in os.listdir(mw.WIDGETS_DIR)))
        self.assertEqual(mw.cmd_check(), 0)

    def test_index_failure_returns_two_after_commit(self):
        with patch.object(mw, 'cmd_index', return_value=1):
            self.assertEqual(self.new(self.input_spec('bars'), 'index-fail'), 2)
        self.assertEqual(len(mw.load_registry()['widgets']), 1)

    def test_theme_light_default_and_system_opt_in(self):
        # 默认白底：HTML 里必须有页面级白底规则，且不再把页面声明成深色
        self.assertEqual(self.new(self.input_spec('plot'), '默认主题'), 0)
        with open(self.path('Maps/_widgets/样本-默认主题.html'), encoding='utf-8') as f:
            html = f.read()
        self.assertIn('data-wg-theme', html)          # 运行时把 spec.theme 落到属性上
        self.assertIn('background:#fff', html)        # 白底页面规则已内联
        self.assertNotIn('content="dark"', html)
        # 非法 theme 必须被拒绝
        bad = copy.deepcopy(mw.KINDS['plot']['min'])
        bad['uid'] = 'N0001.01'
        bad['theme'] = 'dark'
        self.assertTrue(any('theme' in e for e in mw.validate_spec(bad)[0]))
        # system 是唯一的显式跟随系统入口
        opt_in_path = self.input_spec('plot', '-system')
        with open(opt_in_path, encoding='utf-8') as f:
            opt_in = json.load(f)
        opt_in['theme'] = 'system'
        with open(opt_in_path, 'w', encoding='utf-8') as f:
            json.dump(opt_in, f, ensure_ascii=False, indent=1)
            f.write('\n')
        self.assertEqual(mw.validate_spec(opt_in)[0], [])
        self.assertEqual(self.new(opt_in_path, '系统主题'), 0)
        self.assertEqual(mw.cmd_check(), 0)

    def test_dark_page_rules_not_nested_in_wrap_block(self):
        """页面级深色规则必须与 .wg-wrap[data-theme="system"] 平级。

        写进那个声明块内部时，现代浏览器的 CSS 嵌套会把它解释成
        `.wg-wrap[data-theme="system"] html[data-wg-theme="system"]`——永远不匹配，
        于是深色系统下组件四周仍留白边。这类错误肉眼很难发现，所以在这里盯住。
        """
        css = mw.read_text(mw.WIDGETS_CSS)
        lines = css.splitlines()
        start = next(i for i, l in enumerate(lines) if '.wg-wrap[data-theme="system"]{' in l)
        depth, block_end = 0, None
        for i in range(start, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth <= 0:
                block_end = i
                break
        self.assertIsNotNone(block_end, '没找到 .wg-wrap[data-theme="system"] 的闭合括号')
        block = '\n'.join(lines[start:block_end + 1])
        self.assertNotIn('html[data-wg-theme=', block, '页面级深色规则被写进了声明块内部')
        tail = '\n'.join(lines[block_end:])
        self.assertIn('html[data-wg-theme="system"] body', tail)
        self.assertIn('@media (prefers-color-scheme: dark)', css)

    @unittest.skipUnless(shutil.which('node'), 'node not installed')
    def test_javascript_runtime(self):
        subprocess.run(['node', os.path.join(HERE, 'test_widget_runtime.js')], check=True)


class LayoutRootTest(unittest.TestCase):
    """--root / --vault / --layout 与 <root>/widgets.config.json：全在临时项目里跑，不碰真实 vault。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('PI_SCRATCH_DIR'))
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, rel):
        return os.path.join(self.root, rel.replace('/', os.sep))

    def write(self, rel, text):
        path = self.path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def config(self, obj):
        self.write('widgets.config.json', json.dumps(obj, ensure_ascii=False, indent=1) + '\n')

    def cli(self, *argv):
        return subprocess.run([os.sys.executable, os.path.join(HERE, 'make_widget.py')] + list(argv),
                              capture_output=True, text=True)

    def test_layout_without_config_file(self):
        """无配置文件 + --root <tmp>：默认布局仍指向 <tmp>/Maps/_tools 与 <tmp>/Maps/_widgets。"""
        result = self.cli('--root', self.root, '--layout')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.count('\n'), 1, '--layout 的 stdout 必须只有一行 JSON')
        data = json.loads(result.stdout)
        self.assertEqual(data['root'], os.path.abspath(self.root))
        self.assertIsNone(data['configFile'])
        self.assertEqual(data['toolsDir'], 'Maps/_tools')
        self.assertEqual(data['widgetsDir'], 'Maps/_widgets')
        self.assertEqual(data['registry'], 'Maps/_tools/widgets-index.json')
        self.assertEqual(data['indexMd'], 'Maps/_widgets/Index.md')
        self.assertEqual(data['notesIndex'], 'Maps/_tools/notes-index.json')

    def test_check_on_empty_project_without_config_file(self):
        """无配置文件 + --root <tmp>：--check 在临时项目上跑得完，0 个组件就该是 0 错误。"""
        result = self.cli('--root', self.root, '--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('错误 0、提醒 0', result.stdout)
        self.assertNotIn('Traceback', result.stderr)
        self.assertIn('布局：root=%s · tools=Maps/_tools · widgets=Maps/_widgets · config=无'
                      % os.path.abspath(self.root), result.stdout)

    def test_config_file_drives_layout_and_new_output_paths(self):
        """mapsDir/toolsDir/widgetsDir 改写布局：new 的 .json/.html 与注册表都落在配置的位置。"""
        self.config({'mapsDir': 'notes', 'toolsDir': 'notes/_t', 'widgetsDir': 'notes/_w'})
        result = self.cli('--root', self.root, '--layout')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['configFile'], 'widgets.config.json')
        self.assertEqual(data['mapsDir'], 'notes')
        self.assertEqual(data['toolsDir'], 'notes/_t')
        self.assertEqual(data['widgetsDir'], 'notes/_w')
        self.assertEqual(data['registry'], 'notes/_t/widgets-index.json')
        self.assertEqual(data['indexMd'], 'notes/_w/Index.md')
        self.assertEqual(data['notesIndex'], 'notes/_t/notes-index.json')
        self.write('notes/源.md', '# 源\n\n这篇笔记已经写清了要验证的结论。\n')
        spec = copy.deepcopy(mw.KINDS['plot']['min'])
        spec['uid'] = ''
        self.write('in.json', json.dumps(spec, ensure_ascii=False, indent=1) + '\n')
        result = self.cli('--root', self.root, 'new', '--note', 'notes/源.md', '--spec', self.path('in.json'))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        slug = mw.slugify(spec['title'])
        self.assertTrue(os.path.exists(self.path('notes/_w/源-%s.json' % slug)))
        self.assertTrue(os.path.exists(self.path('notes/_w/源-%s.html' % slug)))
        self.assertTrue(os.path.exists(self.path('notes/_w/Index.md')))
        self.assertTrue(os.path.exists(self.path('notes/_t/widgets-index.json')))
        self.assertFalse(os.path.exists(self.path('Maps')), '不该再往默认的 Maps/ 写东西')
        reg = json.loads(mw.read_text(self.path('notes/_t/widgets-index.json')))
        self.assertEqual(reg['widgets'][0]['json'], 'notes/_w/源-%s.json' % slug)
        self.assertEqual(reg['widgets'][0]['html'], 'notes/_w/源-%s.html' % slug)
        self.assertEqual(reg['widgets'][0]['note'], 'notes/源.md')
        result = self.cli('--root', self.root, '--check')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('错误 0、提醒 0', result.stdout)

    def test_hidden_vault_alias_matches_root(self):
        """--vault 仍是 --root 的隐藏别名（同一个临时项目上语义一致）。"""
        self.config({'mapsDir': 'notes'})
        result = self.cli('--vault', self.root, '--layout')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['root'], os.path.abspath(self.root))
        self.assertEqual(data['toolsDir'], 'notes/_tools')
        self.assertEqual(data['widgetsDir'], 'notes/_widgets')

    def test_unknown_config_key_warns_but_exits_zero(self):
        """未知键只 [提醒]（走 stderr，不污染 --layout 的 stdout），退出码 0。"""
        self.config({'mapsDir': 'notes', 'unknownKey': 1})
        result = self.cli('--root', self.root, '--layout')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[提醒]', result.stderr)
        self.assertIn('unknownKey', result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['toolsDir'], 'notes/_tools')

    def test_invalid_config_values_exit_nonzero(self):
        """绝对路径 / 含 .. / 不是对象 / 值不是字符串：一律 [错误] + 非零退出。"""
        for name, cfg in (('绝对路径', {'mapsDir': '/etc'}),
                          ('含 ..', {'widgetsDir': '../outside'}),
                          ('不是对象', ['Maps']),
                          ('值不是字符串', {'toolsDir': 5})):
            with self.subTest(case=name):
                self.write('widgets.config.json', json.dumps(cfg, ensure_ascii=False))
                result = self.cli('--root', self.root, '--layout')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('[错误]', result.stderr)
                self.assertNotIn('Traceback', result.stderr)

    def test_invalid_json_config_exits_nonzero(self):
        self.write('widgets.config.json', '{not json')
        result = self.cli('--root', self.root, '--layout')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('[错误]', result.stderr)
        self.assertNotIn('Traceback', result.stderr)
    def test_config_symlink_escaping_root_exits_nonzero(self):
        """解析后落在 root 之外的 symlink 也要拦（realpath + commonpath 口径）。"""
        outside = tempfile.mkdtemp(dir=os.environ.get('PI_SCRATCH_DIR'))
        self.addCleanup(shutil.rmtree, outside, True)
        os.symlink(outside, self.path('linked'))
        self.config({'widgetsDir': 'linked/widgets'})
        result = self.cli('--root', self.root, '--layout')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn('[错误]', result.stderr)
        self.assertNotIn('Traceback', result.stderr)


    def test_current_vault_layout_is_pinned_without_root_flag(self):
        """回归：不传 --root/--vault 时，StateFarm 当前配置布局仍被正确解析。"""
        result = self.cli('--layout')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data['root'], os.path.dirname(os.path.dirname(HERE)))
        self.assertEqual(data['configFile'], 'widgets.config.json')
        self.assertEqual(data['toolsDir'], '_meta/tools')
        self.assertEqual(data['widgetsDir'], '_widgets')
        self.assertEqual(data['registry'], '_meta/tools/widgets-index.json')
        self.assertEqual(data['indexMd'], '_widgets/Index.md')
        self.assertEqual(data['notesIndex'], '_meta/tools/notes-index.json')


class BoardTest(unittest.TestCase):
    """组件看板（<widgetsDir>/看板.html）：--board / --index 生成的独立单文件派生索引。

    全程用 `--root <tmp>` 子进程跑 make_widget.py，不碰真实 vault。覆盖：空态、卡片内容与数量、
    确定性（无时间戳，两次生成逐字节相同）、零外链、相对预览、`--check` 的提醒语义（看板可选、
    脏了只 [提醒] 不失败）、`--index` 联动、自定义 widgetsDir。
    """

    BOARD = 'Maps/_widgets/看板.html'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('PI_SCRATCH_DIR'))
        self.root = self.tmp.name
        self.note = 'Maps/Notes/样本.md'
        self.write(self.note, '# 样本\n\n这篇笔记已经写清了看板要验证的结论。\n')
        self.write('Maps/_tools/notes-index.json', json.dumps({
            'vault': 'test-vault', '_说明': 'test fixture',
            'notes': [{'file': self.note, 'uid': 'N0001.01', 'node': 'N0001', 'sub': 0,
                       'sub_title': '样本', 'title': '样本'}],
        }, ensure_ascii=False, indent=1) + '\n')

    def tearDown(self):
        self.tmp.cleanup()

    def path(self, rel):
        return os.path.join(self.root, rel.replace('/', os.sep))

    def write(self, rel, text):
        path = self.path(rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    def read(self, rel):
        with open(self.path(rel), encoding='utf-8') as f:
            return f.read()

    def cli(self, *argv):
        return subprocess.run([os.sys.executable, os.path.join(HERE, 'make_widget.py'),
                               '--root', self.root] + list(argv),
                              capture_output=True, text=True)

    def new_widget(self, kind, title):
        spec = copy.deepcopy(mw.KINDS[kind]['min'])
        spec['title'] = title
        spec['uid'] = 'N0001.01'
        spec_file = self.path('specs/%s-%s.json' % (kind, mw.slugify(title)))
        os.makedirs(os.path.dirname(spec_file), exist_ok=True)
        with open(spec_file, 'w', encoding='utf-8') as f:
            json.dump(spec, f, ensure_ascii=False, indent=1)
            f.write('\n')
        result = self.cli('new', '--note', self.note, '--spec', spec_file)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_empty_project_board_has_empty_state(self):
        result = self.cli('--board')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = self.read(self.BOARD)
        self.assertIn('还没有组件', text)
        self.assertIn('共 0 个组件', text)

    def test_board_cards_match_registered_widgets(self):
        self.new_widget('plot', '看板样例：曲线')
        self.new_widget('bars', '看板样例：条形')
        result = self.cli('--board')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = self.read(self.BOARD)
        self.assertEqual(text.count('class="card"'), 2, '卡片数必须等于已登记组件数')
        for title, kind in (('看板样例：曲线', 'plot'), ('看板样例：条形', 'bars')):
            with self.subTest(title=title):
                self.assertIn(title, text)
                self.assertIn('<span class="k">%s</span>' % kind, text)
        self.assertIn('<span class="note">样本</span>', text)      # 卡片里能看到源笔记文件名（不带 .md）
        self.assertIn('共 2 个组件', text)

    def test_board_is_deterministic(self):
        self.new_widget('plot', '看板样例：曲线')
        self.assertEqual(self.cli('--board').returncode, 0)
        first = self.read(self.BOARD)
        self.assertEqual(self.cli('--board').returncode, 0)      # 幂等：内容没变不落盘
        self.assertEqual(self.read(self.BOARD), first, '看板不含时间戳，两次生成必须逐字节相同')
        os.remove(self.path(self.BOARD))                        # 强制重新生成，再比一次
        self.assertEqual(self.cli('--board').returncode, 0)
        self.assertEqual(self.read(self.BOARD), first)

    def test_board_has_no_external_dependencies(self):
        self.new_widget('plot', '看板样例：曲线')
        self.assertEqual(self.cli('--board').returncode, 0)
        text = self.read(self.BOARD)
        for needle in ('http://', 'https://', '//cdn.', '<script src=', '<link href='):
            self.assertNotIn(needle, text)

    def test_board_previews_are_relative_sibling_paths(self):
        self.new_widget('plot', '看板样例：曲线')
        self.new_widget('bars', '看板样例：条形')
        self.assertEqual(self.cli('--board').returncode, 0)
        text = self.read(self.BOARD)
        srcs = re.findall(r'<iframe[^>]*\bsrc="([^"]+)"', text)
        self.assertEqual(len(srcs), 2)
        for src in srcs:
            with self.subTest(src=src):
                self.assertNotIn(':', src)                          # 没有盘符、没有协议头
                self.assertFalse(src.startswith('/'), src)
                self.assertNotIn('..', src)
                # 文件名里的非 ASCII 字符做了百分号编码；解码后必须是同目录下的纯文件名
                decoded = urllib.parse.unquote(src)
                self.assertTrue(decoded.endswith('.html'))
                self.assertEqual(os.path.basename(decoded), decoded)
                self.assertTrue(os.path.exists(self.path('Maps/_widgets/' + decoded)),
                                '预览要指向同目录真实文件')

    def test_board_survives_symlinked_root(self):
        """符号链接路径下的项目也要生成正确的相对预览路径。

        这是真踩过的 bug：vault_path() 返回 realpath（符号链接已解析），而 WIDGETS_DIR 没解析，
        两边前缀不同时 os.path.relpath 会一路往上绕成 `../../../../private/var/...`，看板预览全指向临时目录。
        macOS 的 /var → /private/var、/tmp 都长这样。
        """
        self.new_widget('plot', '看板样例：曲线')
        link = os.path.join(self.root, 'link-to-root')
        try:
            os.symlink(self.root, link)
        except (OSError, NotImplementedError) as e:
            self.skipTest('本机不支持建符号链接：%s' % e)
        # 从**符号链接**这一侧再跑一次（--root 指向链接）：看板里的预览必须仍是同目录纯文件名
        result = subprocess.run([os.sys.executable, os.path.join(HERE, 'make_widget.py'),
                                 '--root', link, '--board'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        text = self.read(self.BOARD)
        srcs = re.findall(r'<iframe[^>]*\bsrc="([^"]+)"', text)
        self.assertEqual(len(srcs), 1, srcs)
        for src in srcs:
            with self.subTest(src=src):
                self.assertNotIn('..', src, '不许出现 ../../ 绕行：%s' % src)
                self.assertNotIn(':private', src)
                decoded = urllib.parse.unquote(src)
                self.assertEqual(os.path.basename(decoded), decoded, src)
                self.assertTrue(os.path.exists(self.path('Maps/_widgets/' + decoded)), src)
    def test_check_treats_board_as_optional_and_only_warns_on_stale(self):
        self.new_widget('plot', '看板样例：曲线')
        self.assertEqual(self.cli('--board').returncode, 0)
        fresh = self.cli('--check')
        self.assertEqual(fresh.returncode, 0, fresh.stdout + fresh.stderr)
        self.assertIn('错误 0、提醒 0', fresh.stdout)
        os.remove(self.path(self.BOARD))                        # 看板可选：没有也不报错、不提它
        without = self.cli('--check')
        self.assertEqual(without.returncode, 0, without.stdout + without.stderr)
        self.assertNotIn('看板.html', without.stdout + without.stderr)
        # 内容改脏（追加一行）后：只提醒、不报错；退出码实测为 0（看板不新鲜不是错误）
        self.assertEqual(self.cli('--board').returncode, 0)
        with open(self.path(self.BOARD), 'a', encoding='utf-8') as f:
            f.write('<!-- 手改测试 -->\n')
        stale = self.cli('--check')
        self.assertEqual(stale.returncode, 0, '看板不新鲜只该 [提醒]，不该把 --check 变成失败')
        self.assertIn('[提醒]', stale.stdout)
        self.assertIn('Maps/_widgets/看板.html 不是最新', stale.stdout)

    def test_index_rebuilds_board_like_board_does(self):
        self.new_widget('plot', '看板样例：曲线')
        self.assertEqual(self.cli('--board').returncode, 0)
        fresh = self.read(self.BOARD)
        with open(self.path(self.BOARD), 'a', encoding='utf-8') as f:
            f.write('<!-- 手改测试 -->\n')
        self.assertNotEqual(self.read(self.BOARD), fresh)
        result = self.cli('--index')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('看板.html', result.stdout)
        self.assertEqual(self.read(self.BOARD), fresh)
        rerun = self.cli('--board')
        self.assertEqual(rerun.returncode, 0, rerun.stdout + rerun.stderr)
        self.assertIn('已是最新', rerun.stdout)
        self.assertEqual(self.read(self.BOARD), fresh)

    def test_board_follows_custom_widgets_dir(self):
        self.write('widgets.config.json', json.dumps({'widgetsDir': 'k/w'}, ensure_ascii=False) + '\n')
        self.new_widget('plot', '看板样例：曲线')
        board = self.path('k/w/看板.html')
        self.assertTrue(os.path.exists(board), 'new 的 --index 联动就该把看板放进自定义目录')
        os.remove(board)
        result = self.cli('--board')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(os.path.exists(board))
        self.assertFalse(os.path.exists(self.path('Maps/_widgets')), '不该往默认 widgetsDir 写看板')
        self.assertIn('看板样例：曲线', self.read('k/w/看板.html'))
class TeachingGateTest(unittest.TestCase):
    """教学质量门禁：控件必须改变主图，不接受只移动 marker/readout 的伪交互。"""

    def teaching(self, controls):
        return {
            'question': '测试问题：控件是否改变主图？',
            'sourceSection': 'test fixture',
            'controlEffect': 'semantic 控件进入主绘制表达式',
            'visualEvidence': '曲线几何或数据随控件改变',
            'controls': controls,
        }

    def base(self, expr):
        return {
            'schema': 'widget/v1', 'kind': 'plot', 'uid': '', 'title': '门禁测试：主图变化',
            'x': {'min': 0, 'max': 1, 'points': 9},
            'controls': [{'key': 'shift', 'type': 'slider', 'label': 'shift',
                          'min': 0, 'max': 1, 'step': 0.1, 'value': 0}],
            'series': [{'label': 'curve', 'expr': expr}],
            'markers': [{'x': 'shift', 'y': 'shift'}],
            'readouts': [{'label': 'shift', 'expr': 'shift', 'fmt': '0.0'}],
        }

    def run_verifier(self, root):
        return subprocess.run(['node', os.path.join(HERE, 'verify_widget_pages.js'), '--root', root],
                              capture_output=True, text=True)

    def test_teaching_fields_are_required(self):
        spec = self.base('x*x')
        err, _ = mw.validate_spec(spec)
        self.assertTrue(any('teaching.question' in x for x in err))
        self.assertTrue(any('teaching.visualEvidence' in x for x in err))

    def test_marker_only_control_is_rejected(self):
        spec = self.base('x*x')
        spec['teaching'] = self.teaching(['shift'])
        err, _ = mw.validate_spec(spec)
        self.assertTrue(any('shift' in x and '主绘制' in x for x in err), err)

    def test_series_control_is_accepted_and_verifier_sees_geometry_change(self):
        spec = self.base('x*x + shift')
        spec['teaching'] = self.teaching(['shift'])
        err, warn = mw.validate_spec(spec)
        self.assertEqual(err, [], warn)
        with tempfile.TemporaryDirectory(dir=os.environ.get('PI_SCRATCH_DIR')) as root:
            path = os.path.join(root, '_widgets', 'curve.json')
            os.makedirs(os.path.join(root, '_widgets'))
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(spec, f, ensure_ascii=False, indent=1)
            result = self.run_verifier(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('通过', result.stdout)

    def test_verifier_rejects_marker_only_fixture(self):
        spec = self.base('x*x')
        spec['teaching'] = self.teaching(['shift'])
        with tempfile.TemporaryDirectory(dir=os.environ.get('PI_SCRATCH_DIR')) as root:
            os.makedirs(os.path.join(root, '_widgets'))
            with open(os.path.join(root, '_widgets', 'marker-only.json'), 'w', encoding='utf-8') as f:
                json.dump(spec, f, ensure_ascii=False, indent=1)
            result = self.run_verifier(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('主绘制', result.stdout + result.stderr)



if __name__ == '__main__':
    unittest.main(verbosity=2)
