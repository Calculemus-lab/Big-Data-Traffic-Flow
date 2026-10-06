import numpy as np
import pandas as pd
import torch
import torch.nn as nn

HISTORY, HORIZON, STEP = 12, 6, pd.Timedelta(minutes=5)


def to_grids(frame, link_ids, free_speed, times):
    row = pd.Index(times).get_indexer(pd.DatetimeIndex(frame.timestamp))
    col = pd.Index(link_ids).get_indexer(frame.link_id.astype(str))
    keep = (row >= 0) & (col >= 0)
    row, col = row[keep], col[keep]
    speed = np.full((len(times), len(link_ids)), np.nan)
    occupancy, observed = speed.copy(), np.zeros(speed.shape)
    speed[row, col] = frame.speed_kmh.to_numpy(dtype=float, na_value=np.nan)[keep]
    occupancy[row, col] = frame.occupancy.to_numpy(dtype=float, na_value=np.nan)[keep]
    observed[row, col] = frame.pct_observed.to_numpy(dtype=float, na_value=0)[keep]
    known = (observed >= 75) & ~np.isnan(speed)          # unknown is not "no queue"
    ratio = speed / free_speed
    queued = known & (ratio <= 0.6)
    return np.nan_to_num(ratio), np.nan_to_num(occupancy), known, queued


def slot_of(times):
    times = pd.DatetimeIndex(times)
    return (times.hour * 12 + times.minute // 5).to_numpy()


class GCN(nn.Module):
    def __init__(self, n_features, a_ahead, a_behind, hidden=64, layers=2):
        super().__init__()
        self.a_ahead, self.a_behind = a_ahead, a_behind
        self.encode = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.own = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.ahead = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.behind = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)])
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):                                  # x: (batch, links, features)
        h = self.encode(x)
        for own, ahead, behind in zip(self.own, self.ahead, self.behind):
            h = torch.relu(own(h) + ahead(self.a_ahead @ h) + behind(self.a_behind @ h))
        return self.head(h)                                # (batch, links, 6 future steps)


def features(ratio, occupancy, known, origin_slots, weekend, profile):
    history = np.stack([ratio, occupancy, known.astype(float)], -1)            # batch, 12, links, 3
    history = history.transpose(0, 2, 1, 3).reshape(len(ratio), ratio.shape[2], -1)
    angle = 2 * np.pi * origin_slots / 288
    clock = np.stack([np.sin(angle), np.cos(angle), weekend.astype(float)], -1)
    clock = np.repeat(clock[:, None, :], ratio.shape[2], axis=1)
    future_slots = (origin_slots[:, None] + np.arange(1, HORIZON + 1)) % 288   # batch, 6
    usual = profile[future_slots].transpose(0, 2, 1)                           # batch, links, 6
    return torch.tensor(np.concatenate([history, clock, usual], -1), dtype=torch.float32)


def fit(ctx):
    params = ctx.params
    links = ctx.network['links'].set_index('link_id')
    link_ids = sorted(links.index.astype(str))
    free_speed = links.free_speed_kmh.reindex(link_ids).to_numpy(float)
    stamps = pd.DatetimeIndex(ctx.train.timestamp)
    times = pd.date_range(stamps.min(), stamps.max(), freq=STEP)
    ratio, occupancy, known, queued = to_grids(ctx.train, link_ids, free_speed, times)
    slots, weekend = slot_of(times), np.asarray(times.dayofweek >= 5)

    hits, seen = np.zeros((288, len(link_ids))), np.zeros((288, len(link_ids)))
    np.add.at(hits, slots, queued)
    np.add.at(seen, slots, known)
    profile = hits / np.maximum(seen, 1)

    # Graph: who is ahead of / behind each link.
    index = {link: i for i, link in enumerate(link_ids)}
    a_ahead, a_behind = torch.zeros(len(link_ids), len(link_ids)), torch.zeros(len(link_ids), len(link_ids))
    for row in ctx.network['lwr_mainline_topology'].itertuples():
        for nxt in str(row.outgoing_link_ids).split(';'):
            if row.link_id in index and nxt in index:
                a_ahead[index[row.link_id], index[nxt]] = 1
                a_behind[index[nxt], index[row.link_id]] = 1
    a_ahead = a_ahead / a_ahead.sum(1, keepdim=True).clamp(min=1)
    a_behind = a_behind / a_behind.sum(1, keepdim=True).clamp(min=1)

    torch.manual_seed(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    model = GCN(HISTORY * 3 + 3 + HORIZON, a_ahead, a_behind, layers=int(params.get('layers', 2)))
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    # Queued cells are rare (about 7%), so they count more in the loss.
    loss_fn = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(float(params.get('pos_weight', 3.0))))
    origins = np.arange(HISTORY, len(times) - HORIZON)       # every example lies wholly inside ctx.train
    model.train()
    for epoch in range(int(params.get('epochs', 4))):
        order = rng.permutation(origins)
        total, batches = 0.0, 0
        for start in range(0, len(order), 64):
            batch = order[start:start + 64]
            window = batch[:, None] + np.arange(-HISTORY, 0)
            x = features(ratio[window], occupancy[window], known[window], slots[batch], weekend[batch], profile)
            future = batch[:, None] + np.arange(1, HORIZON + 1)
            y = torch.tensor(queued[future].transpose(0, 2, 1), dtype=torch.float32)
            mask = torch.tensor(known[future].transpose(0, 2, 1), dtype=torch.float32)   # learn from known cells only
            loss = (loss_fn(model(x), y) * mask).sum() / mask.sum().clamp(min=1)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total, batches = total + loss.item(), batches + 1
        print(f'epoch {epoch + 1}: loss {total / batches:.4f}', flush=True)
    model.eval()
    return {'model': model, 'link_ids': link_ids, 'free_speed': free_speed, 'profile': profile}


def queue(ctx):
    if 'queue_gnn' not in ctx.cache:                        # fit once per panel/fold
        ctx.cache['queue_gnn'] = fit(ctx)
    state = ctx.cache['queue_gnn']
    out = ctx.targets.copy()
    target_times = pd.DatetimeIndex(out.timestamp)
    origin = target_times.min() - STEP                      # T; history is [T-60, T)
    times = pd.date_range(origin - HISTORY * STEP, origin - STEP, freq=STEP)
    ratio, occupancy, known, _ = to_grids(ctx.observations, state['link_ids'], state['free_speed'], times)
    x = features(ratio[None], occupancy[None], known[None], slot_of([origin]), np.array([origin.dayofweek >= 5]), state['profile'])
    with torch.no_grad():
        probability = torch.sigmoid(state['model'](x))[0].numpy()             # links, 6
    step = ((target_times - origin) // STEP - 1).to_numpy()
    link = pd.Index(state['link_ids']).get_indexer(out.link_id.astype(str))
    chance = np.where(link >= 0, probability[np.maximum(link, 0), step], 0.0)
    out['queue_pred'] = (chance >= float(ctx.params.get('threshold', 0.4))).astype(int)
    return out