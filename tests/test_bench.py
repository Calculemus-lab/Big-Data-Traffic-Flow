import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trafficbench.contracts import KEYS, validate_predictions
from trafficbench.fixture import make_fixture
from trafficbench.metrics import aggregate, queue_metrics, state_metrics
from trafficbench.prepare import prepare
from trafficbench.runner import compare, run, verify_bundle
from trafficbench.submission import COLUMNS, assemble, validate_submission


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('benchmark')
    make_fixture(root / 'release')
    prepare(root / 'release', root / 'bundle')
    return root / 'bundle'


def test_prepared_inputs_and_truth_are_separate(bundle):
    manifest = verify_bundle(bundle)
    for case in manifest['cases']:
        folder = bundle / case['path']
        targets = pd.read_parquet(folder / 'targets.parquet')
        assert set(targets) == set(KEYS[case['task']])
        if case.get('train'):
            train = pd.read_parquet(bundle / case['train'])
            assert train.timestamp.max() < pd.Timestamp('2030-12-01', tz='UTC')
        if case['task'] == 'state':
            obs = pd.read_parquet(folder / 'observations.parquet')
            truth = pd.read_parquet(folder / 'truth.parquet')
            assert len(truth) == len(targets)
            assert obs.speed_kmh.isna().sum() > len(targets)  # non-target horizon blanks
            hidden = targets.merge(obs, on=['timestamp','station_id','link_id'])
            assert hidden[['speed_kmh','flow_vph','occupancy','density_occ_linear_vehpkm']].isna().all().all()
        if case['task'] == 'queue':
            history = pd.read_parquet(folder / 'observations.parquet')
            origin = pd.Timestamp(case['origin'])
            assert history.timestamp.max() < origin
            assert history.timestamp.min() >= origin - pd.Timedelta(minutes=60)
            assert targets.timestamp.min() == origin + pd.Timedelta(minutes=5)
            assert targets.timestamp.max() == origin + pd.Timedelta(minutes=30)
            assert targets.timestamp.nunique() == 6


def test_state_formula_per_lane_and_regime():
    targets = pd.DataFrame({'panel':['A']*3,'timestamp':pd.date_range('2030-01-01',periods=3,tz='UTC'), 'station_id':['S']*3,'link_id':['L']*3,'mask_regime':['R1','R2','R3']})
    truth = targets.assign(speed_kmh=100., flow_vph=2000.)
    pred = targets.assign(speed_kmh=90., flow_vph=2400.)
    network = {'fd_parameters':pd.DataFrame({'link_id':['L'],'lanes':[4.]})}
    rows = state_metrics(pred, truth, network)
    assert all(r['flow_per_lane_rmse'] == 100 for r in rows)
    assert all(r['state_score'] == pytest.approx(.54*.6 + .46*(1-100/600)) for r in rows)


def test_equal_family_and_condition_weighting():
    rows = [dict(fold='a',panel='A_N',family='A',condition='onset',queue_proxy_iou=1.)]*20
    rows += [dict(fold='a',panel='A_N',family='A',condition='ongoing',queue_proxy_iou=0.)]
    rows += [dict(fold='a',panel='B_N',family='B',condition='onset',queue_proxy_iou=0.)]
    assert aggregate(rows)['queue_proxy_iou'] == .25


def test_queue_ignores_unknown_truth_and_scores_iou():
    keys = pd.DataFrame({'window_id':['w']*3,'timestamp':pd.date_range('2030-01-01',periods=3,tz='UTC'),'link_id':['L']*3})
    pred = keys.assign(queue_pred=[1,1,1])
    truth = keys.assign(queue_true=[1,0,0],eligible=[True,True,False])
    assert queue_metrics(pred, truth)['queue_proxy_iou'] == .5


@pytest.mark.parametrize('bad', ['missing','duplicate','unknown','nan','binary'])
def test_reject_invalid_predictions(bad):
    targets = pd.DataFrame({'window_id':['w','w'],'timestamp':['2030-01-01','2030-01-02'],'link_id':['L','L']})
    pred = targets.assign(queue_pred=[0.,1.])
    if bad == 'missing': pred = pred.iloc[:1]
    if bad == 'duplicate': pred.iloc[1] = pred.iloc[0]
    if bad == 'unknown': pred.loc[1,'link_id'] = 'bad'
    if bad == 'nan': pred.loc[1,'queue_pred'] = np.nan
    if bad == 'binary': pred.loc[1,'queue_pred'] = .5
    with pytest.raises(ValueError): validate_predictions('queue',targets,pred)


@pytest.mark.parametrize('task', ['state','queue','odme'])
def test_baselines_end_to_end(bundle, tmp_path, task):
    path = run('baseline', task, bundle, runs=tmp_path, compare_baseline=False)
    metadata = json.loads((path / 'run.json').read_text())
    assert metadata['status'] == 'completed'
    assert metadata['metrics']
    assert list((path / 'predictions').rglob('*.parquet'))
    assert (path / 'source/solutions/baseline.py').exists()


def test_failure_is_logged(bundle, tmp_path):
    with pytest.raises(ValueError, match='must define'):
        run('example', 'queue', bundle, runs=tmp_path)
    metadata = json.loads(next(tmp_path.glob('*/run.json')).read_text())
    assert metadata['status'] == 'failed'
    assert 'must define' in metadata['error']


def test_compare_rejects_different_benchmarks(tmp_path):
    paths = []
    for ident in ['quick','full']:
        p = tmp_path / ident
        p.mkdir()
        (p / 'run.json').write_text(json.dumps({'status':'completed','benchmark_id':ident,'task':'state'}))
        paths.append(p)
    with pytest.raises(ValueError, match='same task and benchmark'): compare(paths)


def submission_files(tmp_path):
    state = pd.DataFrame({'panel':['A'],'timestamp':['2031-03-01T00:00:00Z'],'station_id':['001'],'link_id':['L'],'mask_regime':['R1'],'speed_kmh':[80.],'flow_vph':[2000.]})
    queue = pd.DataFrame({'window_id':['w'],'timestamp':['2031-03-01T00:05:00Z'],'link_id':['L'],'queue_pred':[1]})
    odme = pd.DataFrame({'panel':['A'],'departure_time':['PM'],'path_id':['P'],'origin_zone':['O'],'destination_zone':['D'],'path_flow':[25.]})
    for task, frame in [('state',state),('queue',queue),('odme',odme)]: frame.to_csv(tmp_path / f'{task}.csv',index=False)
    key = pd.concat([state[KEYS['state']].assign(task='state',submission_id='1'), queue[KEYS['queue']].assign(task='queue',submission_id='2'), odme[KEYS['odme']].assign(task='odme',submission_id='3')],ignore_index=True).fillna('')
    key.to_csv(tmp_path / 'key.csv',index=False)
    template = key[['submission_id','task']].copy()
    for col in COLUMNS[2:]: template[col] = 0
    template.to_csv(tmp_path / 'template.csv',index=False)
    return [str(tmp_path / f'{name}.csv') for name in ['state','queue','odme','key','template','output']]


def test_submission_assembly(tmp_path):
    args = submission_files(tmp_path)
    assemble(*args)
    assert validate_submission(args[-1],args[-2]) == 3
    frame = pd.read_csv(args[-1])
    assert frame.flow_vph.tolist() == [2000,0,0]
    assert frame.queue_pred.tolist() == [0,1,0]
    assert frame.path_flow.tolist() == [0,0,25]


def test_submission_streams_across_chunk_boundaries(tmp_path, monkeypatch):
    from trafficbench import submission
    monkeypatch.setattr(submission, 'CHUNK', 1)
    args = submission_files(tmp_path)
    assemble(*args)
    assert validate_submission(args[-1], args[-2]) == 3


def test_duplicate_predictions_across_chunks(tmp_path, monkeypatch):
    from trafficbench import submission
    monkeypatch.setattr(submission, 'CHUNK', 1)
    args = submission_files(tmp_path)
    frame = pd.read_csv(args[2])
    pd.concat([frame, frame]).to_csv(args[2], index=False)
    with pytest.raises(ValueError, match='Duplicate odme'):
        assemble(*args)


@pytest.mark.parametrize('damage', ['missing','extra','duplicate','zones'])
def test_assembly_rejects_incomplete_or_wrong_keys(tmp_path, damage):
    args = submission_files(tmp_path)
    table = pd.read_csv(args[2])
    if damage == 'missing': table = table.iloc[:0]
    if damage == 'extra': table = pd.concat([table,table.assign(path_id='extra')])
    if damage == 'duplicate': table = pd.concat([table,table])
    if damage == 'zones': table['destination_zone'] = 'wrong'
    table.to_csv(args[2],index=False)
    with pytest.raises(ValueError): assemble(*args)
    assert not Path(args[-1]).exists()


@pytest.mark.parametrize('damage', ['id','nonfinite','unused','binary','negative','missing','duplicate'])
def test_final_csv_rejects_invalid_values(tmp_path, damage):
    args = submission_files(tmp_path)
    assemble(*args)
    frame = pd.read_csv(args[-1])
    if damage == 'id': frame.loc[0,'submission_id'] = 99
    if damage == 'nonfinite': frame.loc[0,'speed_kmh'] = np.inf
    if damage == 'unused': frame.loc[0,'path_flow'] = 1
    if damage == 'binary': frame.loc[1,'queue_pred'] = .5
    if damage == 'negative': frame.loc[2,'path_flow'] = -1
    if damage == 'missing': frame = frame.iloc[:2]
    if damage == 'duplicate': frame.loc[1,'submission_id'] = 1
    frame.to_csv(args[-1],index=False)
    with pytest.raises(ValueError): validate_submission(args[-1],args[-2])


def test_hosted_ci_package_run_and_combine(bundle, tmp_path):
    from trafficbench.ci import combine, package, unpack
    from trafficbench.runner import run
    index = package(bundle, tmp_path / 'packages')
    shard = index['shards'][0]
    unpack(tmp_path / 'packages' / shard['asset'], tmp_path / 'shard', shard['sha256'])
    path = run('example', 'state', tmp_path / 'shard', runs=tmp_path / 'runs', panel=shard['panel'])
    original = json.loads((path / 'run.json').read_text())
    merged = combine(tmp_path / 'runs', index, 'example', 'state', tmp_path / 'combined')
    assert merged['metrics']['state_score'] == pytest.approx(original['metrics']['state_score'])
    assert merged['baseline_metrics']['state_score'] == pytest.approx(original['baseline_metrics']['state_score'])
    assert merged['panels'] == ['D12_I5_N']
    index['shards'].append({'panel':'missing', 'tasks':['state']})
    with pytest.raises(ValueError, match='Missing, duplicate'):
        combine(tmp_path / 'runs', index, 'example', 'state', tmp_path / 'bad')


def test_hosted_ci_rejects_changed_download(tmp_path):
    from trafficbench.ci import unpack
    path = tmp_path / 'corrupt.tar.gz'
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='checksum'):
        unpack(path, tmp_path / 'out', 'wrong')


def test_comparison_rejects_partial_panel_run(tmp_path):
    paths = []
    for i, panels in enumerate([['A'], ['A','B']]):
        path = tmp_path / str(i)
        path.mkdir()
        (path / 'run.json').write_text(json.dumps({'status':'completed','benchmark_id':'same','task':'state','evaluator_hash':'same','panels':panels}))
        paths.append(path)
    with pytest.raises(ValueError, match='same panels'):
        compare(paths)
