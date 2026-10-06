import json

import numpy as np
import pandas as pd
import pytest
import torch

from solutions.st_cnn import model
from solutions.st_cnn.grid import Profile, clean, features, lanes_for, locate, period, series_order, slot, to_grid
from solutions.st_cnn.net import Imputer, receptive_steps
from trafficbench.contracts import KEYS, Context
from trafficbench.fixture import make_fixture
from trafficbench.prepare import prepare
from trafficbench.runner import run

FAST = {'steps': 3, 'batch': 2, 'window': 48, 'width': 8, 'device': 'cpu', 'log_every': 0}


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp('st_cnn')
    make_fixture(root / 'release')
    prepare(root / 'release', root / 'bundle')
    return root / 'bundle'


def context(bundle, params=None):
    case = bundle / 'D12_I5_N/december/state'
    network = {p.stem: pd.read_parquet(p) for p in (bundle / 'network/D12_I5_N').glob('*.parquet')}
    return Context(task='state', panel='D12_I5_N', fold='december',
                   train=pd.read_parquet(bundle / 'D12_I5_N/december/train.parquet', dtype_backend='pyarrow'),
                   observations=pd.read_parquet(case / 'observations.parquet'),
                   targets=pd.read_parquet(case / 'targets.parquet'), network=network,
                   params={**FAST, **(params or {})}, seed=7)


def zero_net(*_):
    """A network whose output is zero, so predictions equal the profile."""
    net = Imputer(7, 4)
    torch.nn.init.zeros_(net.out.weight)
    torch.nn.init.zeros_(net.out.bias)
    return net.eval()


def weekend_offset(frame):
    weekend = pd.to_datetime(frame.timestamp, utc=True).dt.dayofweek.ge(5).to_numpy()
    return frame.assign(speed_kmh=frame.speed_kmh.to_numpy(float) + np.where(weekend, 10., 0.))


def frame(rows):
    out = pd.DataFrame(rows, columns=['timestamp', 'station_id', 'link_id', 'speed_kmh', 'flow_vph', 'pct_observed'])
    out['timestamp'] = pd.to_datetime(out.timestamp, utc=True)
    return out


def test_runs_end_to_end_through_the_harness(bundle, tmp_path):
    path = run('st_cnn', 'state', bundle, params=FAST, runs=tmp_path, compare_baseline=False)
    metadata = json.loads((path / 'run.json').read_text())
    assert metadata['status'] == 'completed'
    assert 0 < metadata['metrics']['state_score'] <= 1
    assert (path / 'source/solutions/st_cnn/net.py').exists()


def test_predictions_cover_targets_and_are_valid(bundle):
    ctx = context(bundle)
    out = model.state(ctx)
    assert out[['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime']].equals(ctx.targets)
    assert np.isfinite(out[['speed_kmh', 'flow_vph']]).all().all()
    assert (out[['speed_kmh', 'flow_vph']] >= 0).all().all()


def test_same_seed_gives_same_predictions(bundle):
    first, second = model.state(context(bundle)), model.state(context(bundle))
    pd.testing.assert_frame_equal(first, second)


def test_profile_path_matches_truth_on_a_repeating_week(bundle, monkeypatch):
    """With a zero network the output is the profile; on periodic data it must equal truth."""
    monkeypatch.setattr(model, 'fit', zero_net)
    ctx = context(bundle, {'smooth': 0})
    ctx.train = weekend_offset(pd.read_parquet(bundle / 'D12_I5_N/december/train.parquet'))
    ctx.observations = weekend_offset(ctx.observations)
    truth = weekend_offset(pd.read_parquet(bundle / 'D12_I5_N/december/state/truth.parquet'))
    out = model.state(ctx).merge(truth, on=KEYS['state'], suffixes=('', '_true'), validate='one_to_one')
    assert len(out) == len(truth)
    np.testing.assert_allclose(out.speed_kmh, out.speed_kmh_true, atol=1e-6)
    np.testing.assert_allclose(out.flow_vph, out.flow_vph_true, rtol=1e-9)


def test_eval_pool_never_contains_target_cells(bundle, monkeypatch):
    seen = {}

    def spy(pools, p, seed):
        seen['pools'] = pools
        return zero_net()

    monkeypatch.setattr(model, 'fit', spy)
    ctx = context(bundle)
    model.state(ctx)
    assert len(seen['pools']) == 2
    grid, _, times = seen['pools'][1]
    targets = clean(ctx.targets)
    series = series_order(ctx.network, [clean(ctx.train), clean(ctx.observations), targets])
    t, s = locate(targets, times, series)
    assert (t >= 0).all() and np.isnan(grid[:, t, s]).all()
    model.state(context(bundle, {'use_eval': False}))
    assert len(seen['pools']) == 1


def test_seed_parameter_changes_the_fit(bundle):
    first, other = model.state(context(bundle)), model.state(context(bundle, {'seed': 1}))
    assert not np.allclose(first.speed_kmh, other.speed_kmh)


def test_predictions_follow_keys_not_row_order(bundle):
    ctx = context(bundle)
    shuffled = context(bundle)
    shuffled.targets = shuffled.targets.sample(frac=1, random_state=0).reset_index(drop=True)
    keys = ['timestamp', 'station_id', 'link_id']
    expected = model.state(ctx).sort_values(keys).reset_index(drop=True)
    actual = model.state(shuffled).sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, actual)


def test_grid_round_trip_in_corridor_order():
    rows = [('2030-01-01 00:00', 'S2', 'B', 50., 3000., 100), ('2030-01-01 00:05', 'S1', 'A', 90., 1000., 100),
            ('2030-01-01 00:10', 'S1', 'A', 80., 1200., 60)]
    data = clean(frame(rows))
    network = {'lwr_mainline_topology': pd.DataFrame({'link_id': ['A', 'B'], 'order_index': [1, 0]})}
    series = series_order(network, [data])
    assert list(series.get_level_values('link_id')) == ['B', 'A']
    times = period(data)
    grid = to_grid(data, times, series, np.array([3., 2.]))
    assert grid.shape == (2, 3, 2)
    assert grid[0, 0, 0] == 50 and grid[1, 0, 0] == 1000   # per-lane flow
    assert grid[0, 1, 1] == 90 and grid[1, 1, 1] == 500
    assert np.isnan(grid[:, 2, 1]).all()                    # pct_observed < 75 is not evidence
    t, s = locate(data, times, series)
    assert t.tolist() == [0, 1, 2] and s.tolist() == [0, 1, 1]


def test_locate_marks_unknown_and_out_of_range_keys():
    data = clean(frame([('2030-01-01 00:00', 'S1', 'A', 90., 1000., 100)]))
    series = series_order({}, [data])
    other = clean(frame([('2030-01-01 00:00', 'S9', 'Z', 0., 0., 100), ('2030-01-02 00:00', 'S1', 'A', 0., 0., 100)]))
    t, s = locate(other, period(data), series)
    assert t.tolist() == [-1, -1] and s.tolist() == [-1, -1]


def test_profile_averages_by_weekday_and_falls_back():
    times = pd.date_range('2030-01-07', periods=14 * 288, freq='5min', tz='UTC')   # two Mondays onward
    monday = np.asarray(times.dayofweek == 0)
    grid = np.full((2, len(times), 3), np.nan)
    grid[:, :, 0] = np.where(monday, 10., 20.)
    grid[:, monday, 1] = slot(times)[monday]                                      # Mondays only
    values = Profile(grid, times, smooth=0)(times)
    assert values.shape == grid.shape
    assert np.allclose(values[:, monday, 0], 10) and np.allclose(values[:, ~monday, 0], 20)
    np.testing.assert_allclose(values[:, :, 1], np.broadcast_to(slot(times), (2, len(times))))  # time of day
    assert np.allclose(values[:, :, 2], np.nanmean(grid, axis=(1, 2))[:, None])                  # corridor mean


def test_profile_smoothing_continues_across_midnight():
    times = pd.date_range('2030-01-07', periods=7 * 288, freq='5min', tz='UTC')   # Monday to Sunday
    grid = np.zeros((2, len(times), 1))
    grid[:, 287, 0] = 30.                                                         # Monday 23:55
    values = Profile(grid, times, smooth=1)(times)
    np.testing.assert_allclose(values[:, 286:289, 0], 10)                         # Mon 23:50 to Tue 00:00
    assert (values[:, :286, 0] == 0).all() and (values[:, 289:, 0] == 0).all()


def test_profile_rejects_empty_history():
    times = pd.date_range('2030-01-01', periods=288, freq='5min', tz='UTC')
    with pytest.raises(ValueError, match='No usable'):
        Profile(np.full((2, len(times), 1), np.nan), times)


def test_lanes_follow_link_ids_not_row_order():
    series = pd.MultiIndex.from_arrays([['S1', 'S2'], ['A', 'B']], names=['station_id', 'link_id'])
    assert lanes_for(series, {'fd_parameters': pd.DataFrame({'link_id': ['B', 'A'], 'lanes': [5, 2]})}).tolist() == [2., 5.]
    with pytest.raises(ValueError, match='lane'):
        lanes_for(series, {'fd_parameters': pd.DataFrame({'link_id': ['A'], 'lanes': [2]})})


def test_hidden_cells_follow_regime_rates():
    rng = np.random.default_rng(0)
    visible = rng.random((200, 200)) < .7
    rates = set()
    for _ in range(60):
        hidden = model.hide(visible, rng)
        assert not (hidden & ~visible).any()
        rate = hidden.sum() / visible.sum()
        assert min(abs(rate - r) for r in model.RATES) < .02
        rates.add(round(rate, 1))
    assert rates == set(model.RATES)


def test_training_batches_only_show_unhidden_cells():
    rng = np.random.default_rng(0)
    times = pd.date_range('2030-01-01', periods=96, freq='5min', tz='UTC')
    values = rng.normal(100, 10, (2, 96, 6))
    values[:, rng.random((96, 6)) < .3] = np.nan
    profile = np.full_like(values, 90.)
    norm = {'resid': np.array([5., 50.]), 'mean': np.zeros(2), 'std': np.ones(2)}
    x, y, m = next(model.batches([(values, profile, times)], {'batch': 3, 'window': 96, 'norm': norm}, rng))
    visible = ~np.isnan(values[0])
    for xi, yi, hidden in zip(x.numpy(), y.numpy(), m.numpy().astype(bool)):
        shown = visible & ~hidden
        assert hidden.any() and not (hidden & ~visible).any()
        np.testing.assert_array_equal(xi[2].astype(bool), shown)
        assert (xi[:2][:, ~shown] == 0).all()
        np.testing.assert_allclose(xi[0][shown], ((values[0] - 90) / 5)[shown], rtol=1e-5)
        np.testing.assert_allclose(xi[1][shown], ((values[1] - 90) / 50)[shown], rtol=1e-5)
        np.testing.assert_allclose(yi[:, visible], (values - profile)[:, visible], rtol=1e-5)


def test_features_zero_hidden_residuals():
    times = pd.date_range('2030-01-01', periods=4, freq='5min', tz='UTC')
    values = np.full((2, 4, 3), 7.)
    visible = np.zeros((4, 3), bool)
    visible[1, 2] = True
    norm = {'resid': np.array([2., 4.]), 'mean': np.zeros(2), 'std': np.ones(2)}
    x = features(values, visible, np.ones((2, 4, 3)), times, norm)
    assert x.shape == (7, 4, 3) and x.dtype == np.float32
    assert x[0, 1, 2] == 3 and x[1, 1, 2] == 1.5
    assert (x[:2][:, ~visible] == 0).all() and x[2].sum() == 1


def test_score_loss_matches_competition_formula():
    target = torch.zeros(1, 2, 2, 2)
    pred = target.clone()
    pred[:, 0] = 5.    # speed error 5 km/h
    pred[:, 1] = 60.   # flow error 60 veh/h/lane
    mask = torch.ones(1, 2, 2)
    mask[0, 0, 0] = 0
    pred[0, :, 0, 0] = 1e4      # unmasked cells must not count
    loss = model.score_loss(pred, target, mask)
    assert loss.item() == pytest.approx(.54 * 5 / 25 + .46 * 60 / 600)


def test_chunked_prediction_matches_a_direct_pass():
    torch.manual_seed(0)
    net = Imputer(7, 4).eval()
    times = pd.date_range('2030-01-01', periods=300, freq='5min', tz='UTC')
    rng = np.random.default_rng(0)
    hidden = rng.random((300, 5)) < .5
    values = np.where(hidden, np.nan, rng.normal(0, 10, (2, 300, 5)))
    profile = rng.normal(0, 1, (2, 300, 5))
    norm = {'resid': np.array([3., 70.]), 'mean': np.zeros(2), 'std': np.ones(2)}
    with torch.no_grad():
        x = torch.from_numpy(features(values, ~hidden, profile, times, norm))[None]
        direct = net(x)[0].numpy() * norm['resid'][:, None, None] + profile
    chunked = model.predict(net, values, profile, times, {'norm': norm}, chunk=70)
    assert receptive_steps() * 2 < 70 * 2 and np.isfinite(chunked).all()
    np.testing.assert_allclose(chunked, direct, atol=1e-4)


def test_rejects_invalid_parameters(bundle):
    with pytest.raises(ValueError, match='positive'):
        model.state(context(bundle, {'steps': 0}))
