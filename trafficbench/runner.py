import hashlib
import importlib
import importlib.metadata
import json
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # not available on Windows
    resource = None

from .contracts import Context, validate_predictions
from .data import digest, save_table
from .metrics import aggregate, odme_metrics, physics_diagnostics, queue_metrics, state_metrics


def code_files():
    return sorted(p for folder in ['trafficbench', 'solutions', 'config'] for p in Path(folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.json'}) + [p for p in [Path('pyproject.toml'), Path('uv.lock'), Path('.python-version')] if p.exists()]


def code_hash():
    return digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in code_files()})


def solution_hash(name):
    path = Path('solutions') / name
    files = sorted(path.rglob('*.py')) if path.is_dir() else [path.with_suffix('.py')]
    return digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def evaluator_hash():
    return digest({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('trafficbench').glob('*.py'))})


def git(*args):
    result = subprocess.run(['git', *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def verify_bundle(bundle, panel=None):
    manifest = json.loads((bundle / 'manifest.json').read_text())
    ident = manifest['benchmark_id']
    if digest({k:v for k,v in manifest.items() if k != 'benchmark_id'}) != ident:
        raise ValueError('Benchmark manifest changed; prepare a new version')
    for name, expected in manifest['files'].items():
        if panel and not (name.startswith(panel + '/') or name.startswith('network/' + panel + '/')):
            continue
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Benchmark file changed: {name}; prepare a new version')
    return manifest


def report(metadata, rows):
    lines = [f"# {metadata['solution']} — {metadata['status']}", '', f"Task: {metadata['task']} | Benchmark: {metadata['benchmark_id'][:12]} | Profile: {metadata['profile']}", '',
             'Queue and OD synthetic scores are proxies. Physics values are diagnostics. No local overall competition score is available.', '',
             '| Metric | Result | Baseline | Change |', '| --- | ---: | ---: | ---: |']
    for key, value in metadata.get('metrics', {}).items():
        baseline = metadata.get('baseline_metrics', {}).get(key)
        lines.append(f"| {key} | {value:.6f} | {baseline:.6f} | {value-baseline:+.6f} |" if baseline is not None else f'| {key} | {value:.6f} | — | — |')
    lines += ['', f"Elapsed: {metadata.get('seconds', 0):.1f}s. Peak process memory: {metadata.get('peak_memory_mb', 0):.0f} MB."]
    if metadata.get('error'):
        lines += ['', '```', metadata['error'], '```']
    lines += ['', 'Per-panel, regime, condition and window results: metrics.csv. Predictions: predictions/.']
    for warning in metadata.get('warnings', []):
        lines += ['', f'Note: {warning}']
    return '\n'.join(lines) + '\n'


def run(solution, task='state', bundle='data/benchmarks/quick', params=None, runs='runs', compare_baseline=True, panel=None):
    bundle, runs = Path(bundle), Path(runs)
    manifest = verify_bundle(bundle, panel)
    params = params or {}
    selected = [c for c in manifest['cases'] if c['task'] == task and (panel is None or c['panel'] == panel)]
    if not selected:
        raise ValueError(f'No {task} cases in this benchmark')
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + solution + '-' + task + '-' + uuid.uuid4().hex[:6]
    directory = runs / run_id
    directory.mkdir(parents=True)
    versions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
    metadata = {'run_id': run_id, 'solution': solution, 'task': task, 'params': params, 'status':'running', 'timestamp': datetime.now(timezone.utc).isoformat(),
                'benchmark_id':manifest['benchmark_id'], 'profile':manifest['profile'], 'seed':manifest['scheme']['seed'],
                'panels':sorted({c['panel'] for c in selected}),
                'commit':git('rev-parse','HEAD'), 'branch':git('branch','--show-current'), 'dirty':bool(git('status','--porcelain')),
                'code_hash':code_hash(), 'evaluator_hash':evaluator_hash(),
                'baseline_hash':solution_hash('baseline'),
                'python':sys.version, 'platform':platform.platform(), 'dependencies':versions, 'warnings':manifest.get('warnings', [])}
    for p in code_files():
        destination = directory / 'source' / p
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, destination)
    (directory / 'benchmark.json').write_text(json.dumps(manifest, indent=2))
    (directory / 'run.json').write_text(json.dumps(metadata, indent=2))
    rows, failure = [], None
    try:
        module = importlib.import_module(f'solutions.{solution}')
        predict = getattr(module, task, None)
        if not callable(predict):
            raise ValueError(f'solutions/{solution} must define {task}(ctx)')
        previous, train, cache, network = None, pd.DataFrame(), {}, {}
        for case in selected:
            group = case['panel'], case['fold']
            if group != previous:
                # Arrow stores repeated strings compactly; avoid holding the previous
                # panel's large training table while reading the next one.
                train, cache = pd.DataFrame(), {}
                if 'ctx' in locals():
                    del ctx
                train = pd.read_parquet(bundle / case['train'], dtype_backend='pyarrow') if case.get('train') else pd.DataFrame()
                network = {p.stem: pd.read_parquet(p) for p in (bundle / 'network' / case['panel']).glob('*.parquet')}
                cache, previous = {}, group
            tables = {p.stem: pd.read_parquet(p) for p in (bundle / case['path']).glob('*.parquet')}
            targets = tables['targets']
            seed = int(digest([metadata['seed'], case['path']])[:8], 16)
            random.seed(seed)
            np.random.seed(seed)
            ctx = Context(task=task, panel=case['panel'], fold=case['fold'], train=train,
                          observations=tables.get('observations', pd.DataFrame()), targets=targets.copy(deep=True),
                          network={k:v.copy(deep=True) for k,v in network.items()}, params=params.copy(),
                          ramps=tables.get('ramps', pd.DataFrame()), counts=tables.get('counts', pd.DataFrame()).copy(),
                          prior=tables.get('prior', pd.DataFrame()).copy(), cache=cache, seed=seed)
            if task == 'queue':
                origin = pd.Timestamp(case['origin'])
                if not ctx.observations.timestamp.lt(origin).all() or not ctx.observations.timestamp.ge(origin-pd.Timedelta(minutes=60)).all():
                    raise ValueError('Queue history extends outside permitted window')
            print(f'{solution}: {case["panel"]}/{case["fold"]}/{Path(case["path"]).name}', flush=True)
            predictions = validate_predictions(task, targets, predict(ctx))
            save_table(directory / 'predictions' / f'{case["path"]}.parquet', predictions)
            common = {k:case[k] for k in ['task','panel','family','fold']}
            if task == 'state':
                detail = state_metrics(predictions, tables['truth'], network)
                detail += [physics_diagnostics(predictions, network)]
            elif task == 'queue':
                detail = [{**queue_metrics(predictions, tables['truth']), 'condition':case['condition'], 'window_id':case['id']}]
            else:
                measured = odme_metrics(predictions, tables['counts'], tables['prior'], network, tables.get('truth'))
                # Keep observed count fit separate from synthetic-case diagnostics.
                if 'truth' in tables:
                    measured = {('synthetic_' + k if k != 'odme_synthetic_score' else k): v for k,v in measured.items()}
                detail = [{**measured, 'case':case['kind']}]
            rows.extend({**common, **d} for d in detail)
        metadata['metrics'] = aggregate(rows)
        metadata['status'] = 'completed'
    except Exception as exc:
        metadata['status'] = 'failed'
        metadata['error'] = traceback.format_exc()
        failure = exc
    finally:
        metadata['seconds'] = time.perf_counter() - started
        if resource:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata['peak_memory_mb'] = rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)
        pd.DataFrame(rows).to_csv(directory / 'metrics.csv', index=False)
        (directory / 'run.json').write_text(json.dumps(metadata, indent=2))
        (directory / 'report.md').write_text(report(metadata, rows))
        print(f'Results: {directory}', flush=True)
    if failure:
        raise failure
    if compare_baseline and solution != 'baseline':
        # A fixed baseline has no experiment parameters; cache only identical code/data/env.
        baseline = None
        for p in sorted(runs.glob('*/run.json'), reverse=True):
            candidate = json.loads(p.read_text())
            if all(candidate.get(k) == metadata[k] for k in ['task','benchmark_id','panels','evaluator_hash','baseline_hash','dependencies']) and candidate.get('solution') == 'baseline' and candidate.get('status') == 'completed' and not candidate.get('params'):
                baseline = candidate
                break
        if baseline is None:
            path = run('baseline', task, bundle, {}, runs, False, panel)
            baseline = json.loads((path / 'run.json').read_text())
        metadata['baseline_run'] = baseline['run_id']
        metadata['baseline_metrics'] = baseline['metrics']
        (directory / 'run.json').write_text(json.dumps(metadata, indent=2))
        (directory / 'report.md').write_text(report(metadata, rows))
    print(json.dumps(metadata['metrics'], indent=2))
    return directory


def compare(paths):
    records = [json.loads((Path(p) / 'run.json').read_text()) for p in paths]
    if any(r['status'] != 'completed' for r in records):
        raise ValueError('Cannot compare a failed run')
    if len({(r['benchmark_id'], r['task']) for r in records}) != 1:
        raise ValueError('Runs must use the same task and benchmark; quick/full/holdout are not interchangeable')
    if len({r.get('evaluator_hash') for r in records}) != 1:
        raise ValueError('Scoring code changed between these runs; rerun using the same evaluator')
    if len({tuple(r.get('panels', [])) for r in records}) != 1:
        raise ValueError('Runs must cover the same panels; a partial run is not a full benchmark')
    frame = pd.DataFrame([{'run':r['run_id'], **r['metrics'], 'seconds':r['seconds']} for r in records])
    print(frame.to_string(index=False))
    return frame
