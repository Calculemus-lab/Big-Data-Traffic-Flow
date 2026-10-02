"""Freeze benchmark inputs and answers once; reuse them for every experiment."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import VERSION
from .contracts import KEYS, VALUES
from .data import Release, digest, save_table
from .metrics import operator


def select_state_truth(targets, truth):
    keys = ['timestamp', 'station_id', 'link_id']
    selected = targets.merge(truth[keys + ['speed_kmh', 'flow_vph', 'pct_observed']], on=keys, how='left', validate='one_to_one')
    if selected[['speed_kmh', 'flow_vph']].isna().any().any() or not selected.pct_observed.ge(75).all():
        raise ValueError('Template target has missing or ineligible ground truth')
    return selected[KEYS['state'] + VALUES['state']]


def proxy_windows(truth, panel, fold, network, config, seed):
    """Observation-based proxies, using the release's [T-60,T) history convention."""
    speed = truth.pivot_table(index='timestamp', columns='link_id', values='speed_kmh', aggfunc='mean')
    speed = speed.reindex(pd.date_range(speed.index.min(), speed.index.max(), freq='5min'))
    quality = truth.assign(eligible=truth.pct_observed.ge(75) & truth.speed_kmh.notna()).pivot_table(index='timestamp', columns='link_id', values='eligible', aggfunc='all').reindex(index=speed.index, columns=speed.columns).fillna(False).astype(bool)
    free = network['links'].set_index('link_id').free_speed_kmh.reindex(speed.columns)
    if free.isna().any():
        raise ValueError('Missing queue threshold')
    queued = speed.le(.6 * free, axis=1).to_numpy() & quality.to_numpy()
    valid = quality.to_numpy()
    rng = np.random.default_rng(seed)
    candidates = rng.permutation(np.arange(12, len(speed) - 6))
    selected = {'queue_onset': [], 'queue_ongoing': []}
    windows = []
    for i in candidates:
        history_q = queued[i-12:i]
        future_q = queued[i+1:i+7]
        if valid[i-12:i].mean() < config['minimum_coverage'] or valid[i+1:i+7].mean() < config['minimum_coverage'] or not future_q.any():
            continue
        # Match the downloaded contract's established-queue definition.
        condition = 'queue_ongoing' if (history_q.sum(axis=0) >= 2).any() else 'queue_onset'
        previous = selected[condition]
        if len(previous) >= config['windows_per_condition'] or any(abs(i-j)*5 < config['minimum_spacing_minutes'] for j in previous):
            continue
        previous.append(int(i))
        origin = speed.index[i]
        wid = f'proxy_{panel}_{fold}_{condition}_{len(previous):02d}'
        history = truth[truth.timestamp.ge(origin - pd.Timedelta(minutes=60)) & truth.timestamp.lt(origin)].copy()
        history['window_id'] = wid
        target = pd.MultiIndex.from_product([[wid], speed.index[i+1:i+7], speed.columns], names=KEYS['queue']).to_frame(index=False)
        answer = target.copy()
        answer['queue_true'] = future_q.ravel().astype(int)
        answer['eligible'] = valid[i+1:i+7].ravel()
        windows.append(({'id': wid, 'origin': origin.isoformat(), 'condition': condition}, history, target, answer))
        if all(len(v) == config['windows_per_condition'] for v in selected.values()):
            break
    # Never silently compare experiments on different condition coverage.
    if any(not v for v in selected.values()):
        raise ValueError(f'{panel}/{fold}: no windows for both queue conditions; enlarge evaluation dates')
    return sorted(windows, key=lambda w: w[0]['id'])


def prepare(data, output, profile='quick', scheme='config/cv_scheme.yaml'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'{output} is not empty. Reuse it with bench run, or prepare into a new directory.')
    config = yaml.safe_load(Path(scheme).read_text())
    if profile not in config['profiles']:
        raise ValueError(f'Unknown profile: {profile}')
    release = Release(data)
    corridors = json.loads(release.raw('config/corridors.json'))['panels']
    families = {p['corridor_id']: p['family_id'] for p in corridors}
    spec = config['profiles'][profile]
    panels = sorted(families) if spec['panels'] == 'all' else spec['panels']
    manifest = {'version': VERSION, 'profile': profile, 'scheme': config, 'data_fingerprint': release.fingerprint, 'cases': [], 'warnings': []}
    output.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        print(f'Preparing {panel}', flush=True)
        network = release.network(panel)
        for name, table in network.items():
            save_table(output / 'network' / panel / f'{name}.parquet', table)
        template = release.csv(f'task1/{panel}/train/sample_submission_state.csv')[KEYS['state']]
        for fold in spec['folds']:
            dates = config['folds'][fold]
            start, end = dates['start'], dates['end']
            train_start = config['train_start']
            if spec.get('train_days'):
                train_start = (pd.Timestamp(start) - pd.Timedelta(days=spec['train_days'])).strftime('%Y-%m-%d')
            if spec.get('eval_days'):
                end = min(end, (pd.Timestamp(start) + pd.Timedelta(days=spec['eval_days'])).strftime('%Y-%m-%d'))
            base = Path(panel) / fold
            train = release.states(panel, 'train', 'mainline_states', train_start, start)
            save_table(output / base / 'train.parquet', train)
            del train
            truth = release.states(panel, 'train', 'mainline_states', start, end)
            observed = release.states(panel, 'train', 'mainline_states_masked', start, end)
            target = template[template.timestamp.ge(pd.Timestamp(start, tz='UTC')) & template.timestamp.lt(pd.Timestamp(end, tz='UTC'))].copy()
            if set(target.mask_regime) != {'R1', 'R2', 'R3'}:
                raise ValueError(f'{panel}/{fold} lacks targets for all three regimes')
            # Verify that ALL measured target channels are hidden in model inputs.
            target_observed = target.merge(observed, on=['timestamp','station_id','link_id'], validate='one_to_one')
            channels = [c for c in ['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm'] if c in target_observed]
            if not target_observed[channels].isna().all().all():
                raise ValueError('Released state targets are not completely masked')
            case_dir = base / 'state'
            tables = {'observations': observed, 'targets': target, 'truth': select_state_truth(target, truth),
                      'ramps': release.states(panel, 'train', 'ramp_states', start, end)}
            for name, table in tables.items():
                save_table(output / case_dir / f'{name}.parquet', table)
            manifest['cases'].append({'task': 'state', 'panel': panel, 'family': families[panel], 'fold': fold, 'path': str(case_dir), 'train': str(base / 'train.parquet')})
            if panel not in config['queue']['excluded_panels']:
                windows = proxy_windows(truth, panel, fold, network, config['queue'], config['seed'])
                counts = {condition: sum(w[0]['condition'] == condition for w in windows) for condition in ['queue_onset','queue_ongoing']}
                if min(counts.values()) < config['queue']['windows_per_condition']:
                    manifest['warnings'].append(f'{panel}/{fold}: only {counts}; conditions still weighted equally')
                for info, history, targets, answer in windows:
                    case_dir = base / info['id']
                    for name, table in {'observations': history, 'targets': targets, 'truth': answer}.items():
                        save_table(output / case_dir / f'{name}.parquet', table)
                    manifest['cases'].append({'task': 'queue', 'panel': panel, 'family': families[panel], 'fold': fold, 'path': str(case_dir), 'train': str(base / 'train.parquet'), **info})
            del truth, observed, tables
        # ODME has one released demand period: it does not share the temporal folds.
        prior = release.csv(f'task4/{panel}/train/synthetic_weak_prior.csv')
        counts = release.csv(f'task4/{panel}/train/synthetic_link_counts.csv')
        targets = release.csv(f'task4/{panel}/train/sample_submission_path_flow.csv')[KEYS['odme']]
        A = operator(network, targets, counts)
        b = prior.set_index('path_id').reindex(targets.path_id).path_flow.to_numpy(float)
        rng = np.random.default_rng(config['seed'])
        for index in range(config['odme']['synthetic_cases'] + 1):
            tables = {'prior': prior, 'counts': counts, 'targets': targets}
            kind = 'released_counts' if index == 0 else f'synthetic_{index}'
            if index:
                f = b * rng.lognormal(0, config['odme']['demand_log_sigma'], len(b))
                synthetic_counts = counts.copy()
                synthetic_counts['count'] = np.maximum(0, (A @ f) * (1 + rng.normal(0, config['odme']['count_noise_fraction'], len(counts))))
                answer = targets.copy()
                answer['path_flow'] = f
                tables = {**tables, 'counts': synthetic_counts, 'truth': answer}
            case_dir = Path(panel) / 'odme' / kind
            for name, table in tables.items():
                save_table(output / case_dir / f'{name}.parquet', table)
            manifest['cases'].append({'task':'odme', 'panel':panel, 'family':families[panel], 'fold':'odme', 'path':str(case_dir), 'kind':kind})
    # Content hashes detect accidental edits and allow portable, identical benchmark bundles.
    import hashlib
    manifest['files'] = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.rglob('*.parquet'))}
    manifest['benchmark_id'] = digest(manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Ready: {output} ({len(manifest["cases"])} cases, {manifest["benchmark_id"][:12]})')
    return output
