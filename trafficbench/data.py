"""Read a downloaded zip or an extracted release without changing the source."""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .contracts import normalize


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class Release:
    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None
        if self.zip:
            names = self.zip.namelist()
            matches = [n for n in names if n.endswith("config/corridors.json")]
            if len(matches) != 1:
                raise ValueError("Expected one release root in the archive")
            self.prefix = matches[0][:-len("config/corridors.json")]
            self.names = sorted(n[len(self.prefix):] for n in names if n.startswith(self.prefix) and not n.endswith("/"))
            self.fingerprint = digest([(i.filename, i.CRC, i.file_size) for i in self.zip.infolist()])
        else:
            if (self.path / "kaggle_public").exists():
                self.path /= "kaggle_public"
            if not (self.path / "config/corridors.json").exists():
                raise ValueError("Point --data at the competition zip or kaggle_public directory")
            self.names = sorted(str(p.relative_to(self.path)) for p in self.path.rglob("*") if p.is_file())
            h = hashlib.sha256()
            for name in self.names:
                h.update(name.encode())
                with (self.path / name).open("rb") as f:
                    for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                        h.update(chunk)
            self.fingerprint = h.hexdigest()

    def raw(self, name):
        return self.zip.read(self.prefix + name) if self.zip else (self.path / name).read_bytes()

    def csv(self, name):
        frame = pd.read_csv(io.BytesIO(self.raw(name)), dtype=str, keep_default_na=False)
        identifiers = {"panel", "station_id", "link_id", "ramp_link_id", "path_id", "origin_zone", "destination_zone", "departure_time", "mask_regime", "window_id", "timestamp"}
        for col in frame:
            if col not in identifiers:
                try:
                    frame[col] = pd.to_numeric(frame[col].replace("", float("nan")))
                except (ValueError, TypeError):
                    pass
        return normalize(frame)

    def network(self, panel):
        prefix = f"corridors/{panel}/network/"
        return {Path(n).stem: self.csv(n) for n in self.names if n.startswith(prefix) and n.endswith(".csv")}

    def states(self, panel, split, kind, start, end):
        prefix = f"corridors/{panel}/{split}/{kind}/"
        frames = []
        for name in self.names:
            if not name.startswith(prefix) or not name.endswith(".parquet"):
                continue
            date = re.search(r"(20\d\d)_([01]\d)_([0-3]\d)\.parquet$", name)
            if date and not start[:10] <= "-".join(date.groups()) < end[:10]:
                continue
            frame = normalize(pq.ParquetFile(io.BytesIO(self.raw(name))).read().to_pandas())
            frame = frame[frame.timestamp.ge(pd.Timestamp(start, tz="UTC")) & frame.timestamp.lt(pd.Timestamp(end, tz="UTC"))]
            if not frame.empty:
                frames.append(frame)
        if not frames:
            raise ValueError(f"No {kind} records for {panel}/{split} in [{start}, {end})")
        return pd.concat(frames, ignore_index=True)


def save_table(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
