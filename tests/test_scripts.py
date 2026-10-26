"""Offline regressions for relocated entry points and benchmark accounting."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'qwen38-27b-Base'
HAUHAU = ROOT / 'qwen38-27b-Uncensored-HauhauCS'
BENCH = ROOT / 'benchmarks/hauhau-local-20261024'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_module('run_matrix', BENCH / 'run_matrix.py')
safe = load_module('run_matrix_safe', BENCH / 'run_matrix_safe.py')
download = load_module('download_qwen', BASE / 'download-qwen38-mirror.py')
hauhau_download = load_module('download_hauhau', HAUHAU / 'download-hauhau-mirror.py')


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='model scripts ')
        self.addCleanup(self.temp.cleanup)
        self.cwd = Path(self.temp.name)
        self.env = dict(os.environ, QWEN_ENV_FILE='/dev/null', HAUHAU_ENV_FILE='/dev/null',
                        VLLM_BIN='/missing/vllm', PORT='18090',
                        MODEL_DIR=str(self.cwd / 'model with spaces'))

    def launch(self, script, *args):
        return subprocess.run(['bash', str(script), *args], cwd=self.cwd,
                              env=self.env, capture_output=True, text=True)

    def test_launchers_resolve_from_other_directory_and_forward_arguments(self):
        primary = self.launch(BASE / 'start-qwen38-bf16-vllm.sh', '--dry-run', '--seed', '42')
        hauhau = self.launch(HAUHAU / 'start-hauhau-vllm.sh',
                             '--dry-run', '--seed', '42')
        self.assertEqual(primary.returncode, 0, primary.stderr)
        self.assertEqual(hauhau.returncode, 0, hauhau.stderr)
        self.assertNotEqual(primary.stdout, hauhau.stdout)
        args = shlex.split(primary.stdout)
        self.assertEqual(args[2], str(self.cwd / 'model with spaces'))
        self.assertEqual(args[args.index('--port') + 1], '18090')
        self.assertIn('Qwen3.8 27b', args)
        self.assertEqual(args[-2:], ['--seed', '42'])
        hauhau_args = shlex.split(hauhau.stdout)
        self.assertEqual(hauhau_args[hauhau_args.index('--quantization') + 1], 'compressed-tensors')
        self.assertIn('qwen3.8-27b-hauhau-aggressive', hauhau_args)
        self.assertNotIn('Qwen3.8 27b', hauhau_args)
        self.assertEqual(hauhau_args[-2:], ['--seed', '42'])

    def test_projects_work_without_sibling_or_root_configuration(self):
        for folder, script, model, port, tp in (
            (BASE, 'start-qwen38-bf16-vllm.sh', 'Qwen3.8-27B', '18089', '4'),
            (HAUHAU, 'start-hauhau-vllm.sh', 'HauhauCS-Qwen3.8-27B-Aggressive-AWQ-MTP', '18090', '2'),
        ):
            with self.subTest(project=folder.name), tempfile.TemporaryDirectory() as temp:
                target = Path(temp) / folder.name
                shutil.copytree(folder, target, ignore=shutil.ignore_patterns('__pycache__', '.env'))
                env = dict(os.environ, VLLM_BIN='/missing/vllm')
                for key in ('MODEL_DIR', 'MODEL_NAME', 'MODEL_ALIAS', 'PORT', 'TENSOR_PARALLEL_SIZE', 'QWEN_ENV_FILE', 'HAUHAU_ENV_FILE'):
                    env.pop(key, None)
                # Root BF16 settings must not leak into Hauhau.
                if folder == HAUHAU:
                    (Path(temp) / '.env').write_text('MODEL_NAME=wrong-model\nPORT=1\n')
                proc = subprocess.run(['bash', str(target / script), '--dry-run'],
                                      cwd='/', env=env, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                args = shlex.split(proc.stdout)
                self.assertEqual(args[2], str(Path(temp) / 'models' / model))
                self.assertEqual(args[args.index('--port') + 1], port)
                self.assertEqual(args[args.index('--tensor-parallel-size') + 1], tp)

    def test_project_configuration_is_loaded(self):
        config = self.cwd / 'custom.env'
        config.write_text('PORT=19000\nMODEL_NAME="my hauhau"\n')
        self.env['HAUHAU_ENV_FILE'] = str(config)
        proc = self.launch(HAUHAU / 'start-hauhau-vllm.sh', '--dry-run')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        args = shlex.split(proc.stdout)
        self.assertEqual(args[args.index('--port') + 1], '19000')
        self.assertIn('my hauhau', args)

    def test_activated_conda_environment_is_preferred(self):
        conda = self.cwd / 'conda env'
        (conda / 'bin').mkdir(parents=True)
        fake = conda / 'bin/vllm'
        fake.write_text('#!/usr/bin/env bash\nexit 0\n')
        fake.chmod(0o755)
        self.env.pop('VLLM_BIN')
        self.env['CONDA_PREFIX'] = str(conda)
        for package, script in ((BASE, 'start-qwen38-bf16-vllm.sh'), (HAUHAU, 'start-hauhau-vllm.sh')):
            proc = self.launch(package / script, '--dry-run')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(shlex.split(proc.stdout)[0], str(fake))

    def test_missing_vllm_exits_with_actionable_error(self):
        proc = self.launch(BASE / 'start-qwen38-bf16-vllm.sh')
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('set VLLM_BIN', proc.stderr)

    def test_real_exec_preserves_exit_status_and_environment(self):
        model = self.cwd / 'model with spaces'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        fake = self.cwd / 'fake vllm'
        fake.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$CUDA_VISIBLE_DEVICES" "$@"\nexit 7\n')
        fake.chmod(0o755)
        self.env.update(VLLM_BIN=str(fake), CUDA_VISIBLE_DEVICES='1,2')
        proc = self.launch(BASE / 'start-qwen38-bf16-vllm.sh', '--seed', '9')
        self.assertEqual(proc.returncode, 7, proc.stderr)
        self.assertEqual(proc.stdout.splitlines()[:3], ['1,2', 'serve', str(model)])
        self.assertEqual(proc.stdout.splitlines()[-2:], ['--seed', '9'])

    def test_setup_environment_resolves_relative_requirements_from_project(self):
        fake = self.cwd / 'fake conda'
        fake.write_text('#!/usr/bin/env bash\npwd\nprintf "%s\\n" "$@"\n')
        fake.chmod(0o755)
        self.env['CONDA_BIN'] = str(fake)
        for package in (BASE, HAUHAU):
            proc = self.launch(package / 'setup-env.sh', '--dry-run')
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.splitlines(), [str(package), 'env', 'create',
                              '--file', str(package / 'environment.yml'), '--dry-run'])


class DownloadTests(unittest.TestCase):
    def test_download_plan_uses_project_paths_without_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, str(BASE / 'download-qwen38-mirror.py'),
                                   '--dry-run'], cwd=folder, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan['model_dir'], str(ROOT / 'models/Qwen3.8-27B'))
            self.assertEqual(plan['files'], 32)
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_hauhau_plan_includes_full_snapshot_and_mtp(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, str(HAUHAU / 'download-hauhau-mirror.py'),
                                   '--dry-run'], cwd=folder, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            plan = json.loads(proc.stdout)
            self.assertEqual(plan['files'], 14)
            self.assertEqual(plan['repo_id'], 'twolven/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-AWQ-MTP')
            for name in ('model.safetensors', 'model-mtp.safetensors', 'tokenizer.json',
                         'video_preprocessor_config.json', 'README.md', 'recipe.yaml'):
                self.assertIn(name, plan['filenames'])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_existing_corruption_requires_explicit_replacement(self):
        from argparse import Namespace
        for module in (download, hauhau_download):
            with self.subTest(module=module.REPO_ID), tempfile.TemporaryDirectory() as temp:
                model_dir = Path(temp)
                entry = {'path': 'model', 'size': 4, 'lfs': {'oid': hashlib.sha256(b'good').hexdigest()}}
                args = Namespace(model_dir=model_dir, force_download=False, endpoint='https://example.invalid')
                downloader = mock.Mock(side_effect=lambda **kw: (model_dir / kw['filename']).write_bytes(b'good'))
                self.assertTrue(module.fetch_file(entry, args, downloader)['verified'])
                self.assertEqual(downloader.call_args.kwargs['revision'], module.REVISION)
                (model_dir / 'model').write_bytes(b'evil')
                downloader.reset_mock()
                with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
                    module.fetch_file(entry, args, downloader)
                downloader.assert_not_called()
                args.force_download = True
                self.assertTrue(module.fetch_file(entry, args, downloader)['verified'])

    def test_verify_only_checks_all_files_and_returns_failure(self):
        for package, script in ((BASE, 'download-qwen38-mirror.py'), (HAUHAU, 'download-hauhau-mirror.py')):
            with self.subTest(project=package.name), tempfile.TemporaryDirectory() as temp:
                model = Path(temp) / 'missing'
                results = Path(temp) / 'results'
                proc = subprocess.run([sys.executable, str(package / script), '--verify-only',
                                       '--model-dir', str(model), '--results-dir', str(results)],
                                      capture_output=True, text=True)
                self.assertEqual(proc.returncode, 1, proc.stderr)
                report = json.loads((results / 'download-verification.json').read_text())
                manifest = json.loads((package / 'metadata/mirror-tree.json').read_text())
                self.assertEqual(set(report['failed']), {e['path'] for e in manifest})
                self.assertFalse(model.exists())

    def test_hash_verification_rejects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model'
            path.write_bytes(b'good')
            entry = {'path': 'model', 'size': 4,
                     'lfs': {'oid': hashlib.sha256(b'good').hexdigest()}}
            self.assertTrue(download.verify(entry, path=path)['verified'])
            path.write_bytes(b'evil')
            with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
                download.verify(entry, path=path)


class BenchmarkTests(unittest.TestCase):
    def test_partial_result_fails_even_when_client_exits_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            config = benchmark.Config(Path(folder))

            def client(command, **kwargs):
                result_dir = Path(command[command.index('--result-dir') + 1])
                (result_dir / 'result.json').write_text(json.dumps({'completed': 19, 'failed': 1}))
                return subprocess.CompletedProcess(command, 0)

            with mock.patch.object(benchmark.subprocess, 'run', side_effect=client), contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.run(benchmark.spec('partial', 'test'), 1, config)
            self.assertEqual(code, 1)
            meta = json.loads(next(Path(folder).glob('runs/*/meta.json')).read_text())
            self.assertEqual(meta['status'], 'failed')

    def test_safe_suite_counts_unattempted_cases_as_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            config = benchmark.Config(Path(folder))
            cases = [benchmark.spec(str(i), 'test') for i in range(3)]
            with mock.patch.object(benchmark, 'wait_ready', side_effect=[True, True, False]), \
                 mock.patch.object(benchmark, 'check_model'), \
                 mock.patch.object(benchmark.subprocess, 'check_output', return_value='test'), \
                 mock.patch.object(benchmark, 'run', return_value=1), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = benchmark.run_suite(config, cases, safe=True)
            self.assertEqual(code, 1)
            summary = json.loads((Path(folder) / 'safe_summary.json').read_text())
            self.assertEqual(summary, {'total': 3, 'attempted': 1, 'succeeded': 0,
                                       'failed': 1, 'skipped': 2})

    def test_safe_import_does_not_replace_full_matrix(self):
        self.assertEqual(len(benchmark.matrix()), 20)
        self.assertEqual(len(safe.safe_matrix()), 23)

    def test_benchmark_dry_runs_need_no_server_or_client(self):
        for name, count in [('run_matrix.py', 20), ('run_matrix_safe.py', 23)]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                output = Path(folder) / 'output'
                proc = subprocess.run([sys.executable, str(BENCH / name), '--dry-run',
                                       '--output-dir', str(output), '--vllm-bin', '/missing/vllm'],
                                      cwd=folder, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(len(json.loads(proc.stdout)['cases']), count)
                self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
