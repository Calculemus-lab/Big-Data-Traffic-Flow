"""Small deterministic synthetic release for testing plumbing, not model quality."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import KEYS
from .data import save_table


def make_fixture(output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Fixture output must be empty')
    panel = 'D12_I5_N'
    (root / 'config').mkdir(parents=True)
    (root / 'config/corridors.json').write_text(json.dumps({'panels':[{'corridor_id':panel,'family_id':'D12_I5'}]}))
    network = root / 'corridors' / panel / 'network'
    network.mkdir(parents=True)
    links = pd.DataFrame({'link_id':['L01','L02'], 'length_km':[1.,1.], 'lanes':[2.,4.], 'free_speed_kmh':[100.,100.], 'capacity_vph':[4000.,8000.], 'critical_density':[40.,80.], 'k_jam':[200.,400.]})
    links.to_csv(network / 'links.csv', index=False)
    links.to_csv(network / 'fd_parameters.csv', index=False)
    paths = pd.DataFrame({'path_id':['P01','P02'], 'origin_zone':['Z0','Z0'], 'destination_zone':['Z1','Z2']})
    paths.to_csv(network / 'path_set.csv', index=False)
    pd.DataFrame({'path_id':['P01','P02','P02'], 'link_id':['L01','L01','L02']}).to_csv(network / 'path_link_incidence.csv', index=False)
    rng = np.random.default_rng(42)
    targets = []
    for day in pd.date_range('2030-11-03', '2030-12-07', freq='D'):
        stamp = pd.date_range(day, periods=288, freq='5min', tz='UTC')
        frame = pd.MultiIndex.from_product([stamp, ['L01','L02']], names=['timestamp','link_id']).to_frame(index=False)
        frame['station_id'] = frame.link_id.map({'L01':'S01','L02':'S02'})
        hour = frame.timestamp.dt.hour
        frame['speed_kmh'] = np.where(hour.between(8,9), 40., 95.)
        frame['flow_vph'] = np.where(frame.link_id.eq('L01'), 2000., 4000.)
        frame['occupancy'] = .2
        frame['density_occ_linear_vehpkm'] = 30.
        frame['pct_observed'] = 100
        frame['is_score_eligible'] = 1
        name = 'synthetic_mainline_' + day.strftime('%Y_%m_%d') + '.parquet'
        save_table(root / 'corridors' / panel / 'train/mainline_states' / name, frame)
        regime = ['R1','R2','R3'][day.day % 3]
        observed = frame.copy()
        observed['mask_regime'] = regime
        mask = rng.random(len(frame)) < {'R1':.2,'R2':.3,'R3':.5}[regime]
        target = observed.loc[mask, ['timestamp','station_id','link_id','mask_regime']].copy()
        target['panel'] = panel
        targets.append(target[KEYS['state']])
        # Extra blanks simulate queue horizons: deliberately absent from task targets.
        extra = frame.timestamp.dt.hour.eq(12)
        observed.loc[mask | extra, ['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm']] = np.nan
        save_table(root / 'corridors' / panel / 'train/mainline_states_masked' / ('mask_regime='+regime) / name, observed)
        ramps = frame[['timestamp','station_id','flow_vph','pct_observed']].copy()
        ramps['ramp_link_id'] = 'R01'
        save_table(root / 'corridors' / panel / 'train/ramp_states' / name.replace('mainline','ramp'), ramps)
    task1 = root / 'task1' / panel / 'train'
    task1.mkdir(parents=True)
    pd.concat(targets).to_csv(task1 / 'sample_submission_state.csv', index=False)
    task4 = root / 'task4' / panel / 'train'
    task4.mkdir(parents=True)
    prior = paths.assign(panel=panel, departure_time='TRAIN_PM', path_flow=[900.,1200.])
    prior.to_csv(task4 / 'synthetic_weak_prior.csv', index=False)
    prior.to_csv(task4 / 'sample_submission_path_flow.csv', index=False)
    pd.DataFrame({'panel':panel, 'link_id':['L01','L02'], 'count':[2200.,1300.]}).to_csv(task4 / 'synthetic_link_counts.csv', index=False)
    print(f'Synthetic CI fixture: {root}')
    return root
