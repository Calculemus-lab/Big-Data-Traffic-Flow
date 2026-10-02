import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='One command to test a traffic-flow idea.')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Prepare a fixed benchmark once, from zip or directory')
    p.add_argument('--data', required=True)
    p.add_argument('--profile', choices=['quick','full','holdout'], default='quick')
    p.add_argument('--output')
    p.add_argument('--scheme', default='config/cv_scheme.yaml')
    p = sub.add_parser('new', help='Create a solution package for a new experiment')
    p.add_argument('name')
    p.add_argument('--task', choices=['state','queue','odme'], default='state')
    p = sub.add_parser('run', help='Run an experiment and compare with the baseline')
    p.add_argument('solution', nargs='?', default='baseline')
    p.add_argument('--task', choices=['state','queue','odme'], default='state')
    p.add_argument('--profile', choices=['quick','full','holdout'], default='quick')
    p.add_argument('--bundle')
    p.add_argument('--params', default='{}', help='JSON object passed to ctx.params')
    p.add_argument('--runs', default='runs')
    p.add_argument('--panel', help='Only this panel, primarily for distributed CI')
    p = sub.add_parser('package-ci', help='Package a prepared benchmark for GitHub-hosted runners')
    p.add_argument('--bundle', default='data/benchmarks/quick')
    p.add_argument('--output', required=True)
    p = sub.add_parser('compare', help='Compare saved runs on the same benchmark')
    p.add_argument('runs', nargs='+')
    p = sub.add_parser('summary', help='Export all run metadata to a CSV')
    p.add_argument('--runs', default='runs')
    p.add_argument('--output', default='runs/summary.csv')
    p = sub.add_parser('assemble', help='Strictly assemble three task prediction files into one submission')
    for name in ['state','queue','odme','key','template','output']:
        p.add_argument('--' + name, required=True)
    p = sub.add_parser('validate', help='Validate a final CSV against the exact upload template')
    p.add_argument('submission')
    p.add_argument('--template', required=True)
    p = sub.add_parser('fixture', help='Create a tiny synthetic release for CI; not a performance benchmark')
    p.add_argument('--output', default='data/fixture')
    args = parser.parse_args()
    if args.command == 'prepare':
        from .prepare import prepare
        prepare(args.data, args.output or f'data/benchmarks/{args.profile}', args.profile, args.scheme)
    elif args.command == 'new':
        if not re.fullmatch(r'[a-z][a-z0-9_]*', args.name):
            parser.error('Use lowercase letters, digits and underscores; start with a letter')
        path = Path('solutions') / args.name
        if path.with_suffix('.py').exists():
            parser.error(f'{path.with_suffix(".py")} already exists')
        if path.exists():
            parser.error(f'{path} already exists')
        path.mkdir()
        (path / '__init__.py').write_text(f'from .model import {args.task}\n')
        (path / 'model.py').write_text(
            f'"""Implement the {args.task} approach here; add helper modules as needed."""\n'
            'from solutions import baseline\n\n\n'
            f'def {args.task}(ctx):\n'
            f'    out = baseline.{args.task}(ctx)\n'
            '    # Replace or improve the baseline. Parameters are in ctx.params.\n'
            '    return out\n'
        )
        print(f'Created {path}/. Run: bench run {args.name} --task {args.task}')
    elif args.command == 'run':
        from .runner import run
        params = json.loads(args.params)
        if not isinstance(params, dict):
            parser.error('--params must be a JSON object')
        if not re.fullmatch(r'[a-z][a-z0-9_]*', args.solution):
            parser.error('Invalid solution name')
        run(args.solution, args.task, args.bundle or f'data/benchmarks/{args.profile}', params, args.runs, panel=args.panel)
    elif args.command == 'package-ci':
        from .ci import package
        package(args.bundle, args.output)
    elif args.command == 'compare':
        from .runner import compare
        compare(args.runs)
    elif args.command == 'summary':
        import pandas as pd
        rows = []
        for p in sorted(Path(args.runs).glob('*/run.json')):
            r = json.loads(p.read_text())
            rows.append({**{k:r.get(k) for k in ['run_id','timestamp','solution','task','status','profile','benchmark_id','commit','seconds']}, **r.get('metrics', {}), 'path':str(p.parent)})
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(args.output)
    elif args.command == 'assemble':
        from .submission import assemble
        assemble(args.state, args.queue, args.odme, args.key, args.template, args.output)
    elif args.command == 'validate':
        from .submission import validate_submission
        validate_submission(args.submission, args.template)
    elif args.command == 'fixture':
        from .fixture import make_fixture
        make_fixture(args.output)


if __name__ == '__main__':
    main()
