"""Package immutable panel bundles and combine hosted-runner results."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import pandas as pd

from .metrics import aggregate
from .runner import report, verify_bundle


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def package(bundle, output):
    bundle, output = Path(bundle), Path(output)
    manifest = verify_bundle(bundle)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Package output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    profile = manifest['profile']
    index = {'profile':profile, 'benchmark_id':manifest['benchmark_id'], 'shards':[]}
    for panel in sorted({c['panel'] for c in manifest['cases']}):
        asset = f'{profile}-{panel}.tar.gz'
        path = output / asset
        with tarfile.open(path, 'w:gz', compresslevel=1) as tar:
            tar.add(bundle / 'manifest.json', arcname='manifest.json')
            for name in sorted(manifest['files']):
                if name.startswith(panel + '/') or name.startswith('network/' + panel + '/'):
                    tar.add(bundle / name, arcname=name, recursive=False)
        if path.stat().st_size >= 2 * 1024**3:
            raise ValueError(f'{asset} exceeds the GitHub release asset limit')
        tasks = sorted({c['task'] for c in manifest['cases'] if c['panel'] == panel})
        index['shards'].append({'panel':panel, 'asset':asset, 'sha256':file_hash(path), 'tasks':tasks})
        print(f'Packed {asset}: {path.stat().st_size / 1024**2:.1f} MiB', flush=True)
    (output / f'{profile}-index.json').write_text(json.dumps(index, indent=2))
    return index


def matrix(index, task):
    return {'include':[{k:s[k] for k in ['panel','asset','sha256']} for s in index['shards'] if task in s['tasks']]}


def unpack(archive, output, expected_hash):
    if file_hash(archive) != expected_hash:
        raise ValueError('Downloaded benchmark checksum does not match the index')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        if any(not member.isfile() for member in tar.getmembers()):
            raise ValueError('Benchmark archive must contain regular files only')
        tar.extractall(output, filter='data')


def combine(root, index, solution, task, output):
    expected = sorted(s['panel'] for s in index['shards'] if task in s['tasks'])
    records = [(p, json.loads(p.read_text())) for p in Path(root).rglob('run.json')]
    chosen = [(p,r) for p,r in records if r['solution'] == solution and r['task'] == task]
    if len(chosen) != len(expected) or sorted(p for _,r in chosen for p in r.get('panels', [])) != expected:
        raise ValueError('Missing, duplicate or unexpected panel results; refusing a partial overall score')
    for _, r in chosen:
        if r['status'] != 'completed' or r['benchmark_id'] != index['benchmark_id'] or len(r['panels']) != 1:
            raise ValueError('Failed panel or mismatched benchmark')
    for key in ['params','evaluator_hash','code_hash','dependencies','commit','profile']:
        if len({json.dumps(r.get(key), sort_keys=True) for _,r in chosen}) != 1:
            raise ValueError(f'Panel runs disagree on {key}')
    if not chosen:
        raise ValueError('No panel results')
    frame = pd.concat([pd.read_csv(p.parent / 'metrics.csv') for p,_ in chosen], ignore_index=True)
    metadata = dict(chosen[0][1])
    metadata.update(run_id='combined-' + metadata['run_id'], panels=expected, metrics=aggregate(frame.to_dict('records')),
                    seconds=sum(r['seconds'] for _,r in chosen), peak_memory_mb=max(r['peak_memory_mb'] for _,r in chosen),
                    panel_runs=[r['run_id'] for _,r in chosen])
    metadata.pop('baseline_run', None)
    if solution != 'baseline':
        baseline_rows = []
        for _,r in chosen:
            matches = [(p,b) for p,b in records if b['run_id'] == r.get('baseline_run') and b['status'] == 'completed']
            if len(matches) != 1:
                raise ValueError('Missing baseline result')
            baseline_rows.append(pd.read_csv(matches[0][0].parent / 'metrics.csv'))
        metadata['baseline_metrics'] = aggregate(pd.concat(baseline_rows).to_dict('records'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'metrics.csv', index=False)
    (output / 'run.json').write_text(json.dumps(metadata, indent=2))
    (output / 'report.md').write_text(report(metadata, frame.to_dict('records')))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('matrix')
    p.add_argument('--index', required=True)
    p.add_argument('--task', required=True)
    p = sub.add_parser('unpack')
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--sha256', required=True)
    p = sub.add_parser('combine')
    p.add_argument('--root', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--solution', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'matrix':
        print(json.dumps(matrix(json.loads(Path(args.index).read_text()), args.task)))
    elif args.command == 'unpack':
        unpack(args.archive, args.output, args.sha256)
    else:
        combine(args.root, json.loads(Path(args.index).read_text()), args.solution, args.task, args.output)


if __name__ == '__main__':
    main()
