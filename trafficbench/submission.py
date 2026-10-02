"""Strict, disk-backed submission assembly; missing predictions are errors."""
import itertools
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .contracts import KEYS, VALUES, normalize, validate_predictions

COLUMNS = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']
CHUNK = 100_000


def chunks(path):
    path = Path(path)
    if path.suffix == '.parquet':
        for batch in pq.ParquetFile(path).iter_batches(batch_size=CHUNK):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, dtype=str, chunksize=CHUNK, keep_default_na=False)


def canonical(frame, keys):
    result = normalize(frame[keys])
    if 'timestamp' in result:
        result['timestamp'] = result.timestamp.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return result.astype(str)


def validate_submission(submission, template):
    count = 0
    with tempfile.TemporaryDirectory(prefix='trafficbench-ids-') as tmp:
        db = sqlite3.connect(str(Path(tmp) / 'ids.sqlite'))
        try:
            db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY)')
            for actual, expected in itertools.zip_longest(chunks(submission), chunks(template)):
                if actual is None or expected is None or len(actual) != len(expected):
                    raise ValueError('Submission row count differs from template')
                if list(actual.columns) != COLUMNS or list(expected.columns) != COLUMNS:
                    raise ValueError(f'Submission must have exactly these columns, in order: {COLUMNS}')
                if not actual[['submission_id','task']].astype(str).reset_index(drop=True).equals(expected[['submission_id','task']].astype(str).reset_index(drop=True)):
                    raise ValueError('Submission IDs, order or task assignments differ from template')
                if actual.submission_id.eq('').any() or not actual.task.isin(VALUES).all():
                    raise ValueError('Empty ID or unknown task')
                try:
                    db.executemany('INSERT INTO ids VALUES (?)', [(str(v),) for v in actual.submission_id])
                except sqlite3.IntegrityError as exc:
                    raise ValueError('Duplicate submission IDs') from exc
                numeric = actual[COLUMNS[2:]].apply(pd.to_numeric, errors='raise')
                if not np.isfinite(numeric.to_numpy()).all():
                    raise ValueError('Submission contains empty or nonfinite values')
                if not numeric.loc[actual.task.eq('queue'), 'queue_pred'].isin([0,1]).all():
                    raise ValueError('queue_pred must be binary')
                if (numeric.loc[actual.task.eq('odme'), 'path_flow'] < 0).any():
                    raise ValueError('path_flow must be nonnegative')
                for task, values in VALUES.items():
                    unused = [c for c in COLUMNS[2:] if c not in values]
                    if numeric.loc[actual.task.eq(task), unused].ne(0).any().any():
                        raise ValueError(f'{task}: unused output columns must be zero')
                count += len(actual)
        finally:
            db.close()
    if not count:
        raise ValueError('Empty submission')
    print(f'Valid submission: {count:,} rows')
    return count


def assemble(state, queue, odme, key, template, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f'{output} already exists; choose a new output filename')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='trafficbench-assemble-', dir=output.parent) as tmp:
        db = sqlite3.connect(str(Path(tmp) / 'predictions.sqlite'))
        staged = Path(tmp) / 'submission.csv'
        try:
            for task, path in [('state',state),('queue',queue),('odme',odme)]:
                keys, values = KEYS[task], VALUES[task]
                definitions = [f'"{k}" TEXT NOT NULL' for k in keys] + [f'"{v}" REAL NOT NULL' for v in values]
                db.execute(f'CREATE TABLE {task} ({", ".join(definitions)}, used INTEGER DEFAULT 0, PRIMARY KEY ({", ".join(keys)}))')
                for frame in chunks(path):
                    clean = validate_predictions(task, frame, frame)
                    table = canonical(clean, keys)
                    for col in values:
                        table[col] = clean[col].to_numpy(float)
                    try:
                        db.executemany(f'INSERT INTO {task} ({", ".join(keys+values)}) VALUES ({", ".join("?" for _ in keys+values)})', table.itertuples(index=False, name=None))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f'Duplicate {task} prediction keys across chunks') from exc
            first = True
            for frame in chunks(key):
                if not frame.task.isin(VALUES).all():
                    raise ValueError('Unknown task in submission key')
                out = frame[['submission_id','task']].copy()
                for col in COLUMNS[2:]:
                    out[col] = 0.
                for task, keys in KEYS.items():
                    selected = frame.task.eq(task)
                    if not selected.any():
                        continue
                    wanted = canonical(frame.loc[selected], keys).reset_index(drop=True)
                    wanted['position'] = np.arange(len(wanted))
                    wanted.to_sql('wanted', db, if_exists='replace', index=False)
                    join = ' AND '.join(f'p."{k}" = w."{k}"' for k in keys)
                    result = db.execute(f'SELECT p.rowid, {", ".join("p."+v for v in VALUES[task])} FROM wanted w LEFT JOIN {task} p ON {join} ORDER BY w.position').fetchall()
                    if any(any(v is None for v in row) for row in result):
                        raise ValueError(f'Missing {task} predictions, or incorrect natural keys/zones')
                    out.loc[selected, VALUES[task]] = np.array([row[1:] for row in result])
                    db.executemany(f'UPDATE {task} SET used=1 WHERE rowid=?', [(row[0],) for row in result])
                out[COLUMNS].to_csv(staged, index=False, mode='w' if first else 'a', header=first)
                first = False
            for task in VALUES:
                if db.execute(f'SELECT COUNT(*) FROM {task} WHERE used=0').fetchone()[0]:
                    raise ValueError(f'Extra {task} predictions not present in submission key')
            validate_submission(staged, template)
            staged.replace(output)
        finally:
            db.close()
    print(f'Wrote {output}')
