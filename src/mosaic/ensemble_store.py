"""Disk-backed completed runs and bounded-memory queries for ensemble views."""
from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np


class _Moments:
    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.0

    def step(self, value):
        if value is None:
            return
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)

    def finalize(self):
        return json.dumps([self.mean, math.sqrt(max(0, self.m2) / (self.n - 1))
                           if self.n > 1 else 0.0])


class RunSequence(Sequence):
    """Read-only sequence; indexing loads one run, iteration streams records."""
    def __init__(self, store, assignments=False):
        self.store = store
        self.assignments = assignments

    def __len__(self):
        return self.store.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        with self.store.connect() as db:
            row = db.execute('SELECT run_id, assignment FROM runs WHERE seq=?'
                             if self.assignments else
                             'SELECT summary FROM runs WHERE seq=?', (index + 1,)).fetchone()
        return ((row[0], self.store.read_assignment(row[1]))
                if self.assignments else json.loads(row[0]))

    def __iter__(self):
        with self.store.connect() as db:
            query = ('SELECT run_id, assignment FROM runs' if self.assignments
                     else 'SELECT summary FROM runs')
            for row in db.execute(query + ' WHERE seq<=? ORDER BY seq', (len(self),)):
                yield ((row[0], self.store.read_assignment(row[1]))
                       if self.assignments else json.loads(row[0]))


class EnsembleStore:
    def __init__(self, path, columns, precinct_count):
        self.path = path
        self.binary_path = path.with_name('ensemble_assignments.bin')
        self.binary_path.touch(exist_ok=False)
        self.precinct_count = precinct_count
        self.count = 0
        self.fields = {col: f'v{i}' for i, col in enumerate(columns)
                       if col not in ('run_id', 'phase')}
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            extra = ''.join(f', {v} REAL' for v in self.fields.values())
            db.execute('CREATE TABLE runs (seq INTEGER PRIMARY KEY, run_id TEXT UNIQUE, '
                       'assignment INTEGER, summary TEXT' + extra + ')')
            db.execute('CREATE TABLE metadata (precinct_count INTEGER, dtype TEXT)')
            db.execute('INSERT INTO metadata VALUES (?, ?)', (precinct_count, '<i4'))
            if 'score' in self.fields:
                db.execute(f'CREATE INDEX score_order ON runs ({self.fields["score"]}, seq)')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute('PRAGMA cache_size=-2048')
            db.execute('PRAGMA temp_store=FILE')
            db.create_aggregate('moments', 1, _Moments)
            db.create_function('shown', 3, lambda v, scale, places:
                               float(np.round(v * scale, places)) if v is not None else None)
            yield db
            db.commit()
        finally:
            db.close()

    def add(self, run_id, districts, row):
        if districts.shape != (self.precinct_count,):
            raise ValueError('Assignment length does not match the ensemble map')
        values = []
        for col in self.fields:
            value = row.get(col)
            values.append(float(value) if isinstance(value, (int, float, np.number))
                          and math.isfinite(value) else None)
        body = json.dumps(row, default=lambda v: v.item())
        offset = self.count * self.precinct_count * 4
        with self.binary_path.open('r+b') as binary:
            binary.seek(offset)
            binary.write(districts.astype('<i4').tobytes())
        args = [self.count + 1, run_id, offset, body, *values]
        with self.connect() as db:
            db.execute('INSERT INTO runs VALUES (' + ','.join('?' for _ in args) + ')', args)
        self.count += 1

    def read_assignment(self, offset):
        with self.binary_path.open('rb') as binary:
            binary.seek(offset)
            data = binary.read(self.precinct_count * 4)
        if len(data) != self.precinct_count * 4:
            raise OSError('Incomplete ensemble assignment file')
        return np.frombuffer(data, dtype='<i4').copy()

    def best_run(self):
        col = self.fields['score']
        with self.connect() as db:
            row = db.execute(f'SELECT run_id FROM runs WHERE {col} IS NOT NULL AND seq<=? '
                             f'ORDER BY {col}, seq LIMIT 1', (self.count,)).fetchone()
        return row[0] if row else None

    def bounds(self, col, scale=1, places=2, count=None):
        field = self.fields.get(col)
        if field is None:
            return (0.0, 1.0)
        with self.connect() as db:
            lo, hi = db.execute(f'SELECT MIN({field}), MAX({field}) FROM runs WHERE seq<=?',
                                (self.count if count is None else count,)).fetchone()
        if lo is None:
            return (0.0, 1.0)
        return float(np.round(lo * scale, places)), float(np.round(hi * scale, places))

    def histogram(self, col, max_bins=40, count=None):
        field = self.fields.get(col)
        if field is None:
            return None
        with self.connect() as db:
            where = f'{field} IS NOT NULL AND seq<=?'
            args = (self.count if count is None else count,)
            n, lo, hi, moments, integral = db.execute(
                f'SELECT COUNT(*), MIN({field}), MAX({field}), moments({field}), '
                f'MIN({field}=CAST({field} AS INTEGER)) FROM runs WHERE {where}', args).fetchone()
            if not n:
                return None
            mean, sd = json.loads(moments)

            def quantile(q):
                rank = (n - 1) * q
                i = int(rank)
                vals = [r[0] for r in db.execute(
                    f'SELECT {field} FROM runs WHERE {where} ORDER BY {field} LIMIT 2 OFFSET ?',
                    (*args, i))]
                return vals[0] + (rank - i) * (vals[-1] - vals[0])

            median = quantile(.5)
            if integral and hi - lo < max_bins and max(abs(lo), abs(hi)) < 2**52:
                edges = np.arange(lo - .5, hi + 1.5)
            elif lo == hi:
                delta = abs(lo) * .01 or .5
                edges = np.array([lo - delta, hi + delta])
            else:
                iqr = quantile(.75) - quantile(.25)
                width = 2 * iqr / np.cbrt(n)
                sturges = math.ceil(math.log2(n) + 1)
                fd = math.ceil(min(max_bins, (hi - lo) / width)) if width > 0 else 1
                edges = np.linspace(lo, hi, min(max_bins, max(sturges, fd)) + 1)
            bins = len(edges) - 1
            # Match numpy's rightmost-edge inclusion, without loading all values.
            db.create_function('bin_index', 1, lambda v:
                               min(bins - 1, max(0, int(
                                   np.searchsorted(edges, v, side='right') - 1))))
            counts = np.zeros(bins, dtype=np.int64)
            for index, count in db.execute(
                    f'SELECT bin_index({field}), COUNT(*) FROM runs WHERE {where} GROUP BY 1',
                    args):
                counts[index] = count
        return dict(n=n, lo=lo, hi=hi, mean=mean, sd=sd, median=median,
                    integral=bool(integral), edges=edges, counts=counts)

    def scatter(self, xcol, ycol, limit=2000, count=None):
        xfield, yfield = self.fields.get(xcol), self.fields.get(ycol)
        if xfield is None or yfield is None:
            return None
        with self.connect() as db:
            where = f'{xfield} IS NOT NULL AND {yfield} IS NOT NULL AND seq<=?'
            args = (self.count if count is None else count,)
            n = db.execute(f'SELECT COUNT(*) FROM runs WHERE {where}', args).fetchone()[0]
            if not n:
                return None
            chosen = set(np.linspace(0, n - 1, min(n, limit), dtype=np.int64))
            points, ids = [], []
            mx = my = xx = yy = xy = 0.0
            xmin = ymin = float('inf')
            xmax = ymax = -float('inf')
            for i, (rid, x, y) in enumerate(db.execute(
                    f'SELECT run_id, {xfield}, {yfield} FROM runs WHERE {where} ORDER BY seq',
                    args)):
                dx, dy = x - mx, y - my
                mx += dx / (i + 1)
                my += dy / (i + 1)
                xx += dx * (x - mx)
                yy += dy * (y - my)
                xy += dx * (y - my)
                xmin, xmax = min(xmin, x), max(xmax, x)
                ymin, ymax = min(ymin, y), max(ymax, y)
                if i in chosen:
                    points.append((x, y))
                    ids.append(rid.removeprefix('run_'))
        r = min(1.0, max(-1.0, xy / math.sqrt(xx * yy))) if n >= 3 and xx > 0 and yy > 0 else None
        slope = xy / xx if xx > 0 else 0.0
        return dict(n=n, points=np.asarray(points), ids=ids, r=r, slope=slope,
                    intercept=my - slope * mx, xbounds=np.array([xmin, xmax]),
                    ybounds=np.array([ymin, ymax]))

    def roster(self, criteria, sort_col=None, descending=False, page=0, page_size=100,
               count=None):
        where, args = ['seq<=?'], [self.count if count is None else count]
        for col, lo, hi, pin_lo, pin_hi, scale, places in criteria:
            field = self.fields[col]
            where.append(f'{field} IS NOT NULL')
            for pinned, op, value in ((pin_lo, '>=', lo), (pin_hi, '<=', hi)):
                if not pinned:
                    where.append(f'shown({field}, ?, ?) {op} ?')
                    args.extend((scale, places, value))
        clause = ' AND '.join(where)
        field = self.fields.get(sort_col, 'seq')
        sort_args = []
        for col, _, _, _, _, scale, places in criteria:
            if col == sort_col:
                field = f'shown({field}, ?, ?)'
                sort_args = [scale, places]
                break
        direction = 'DESC' if descending else 'ASC'
        with self.connect() as db:
            total = db.execute(f'SELECT COUNT(*) FROM runs WHERE {clause}', args).fetchone()[0]
            page = min(max(0, page), max(0, (total - 1) // page_size))
            rows = [json.loads(r[0]) for r in db.execute(
                f'SELECT summary FROM runs WHERE {clause} ORDER BY {field} {direction}, seq '
                'LIMIT ? OFFSET ?', (*args, *sort_args, page_size, page * page_size))]
        return rows, total, page

    def export_assignments(self, path, id_col, precinct_ids):
        """Transpose disk records in small precinct stripes, then stream CSV rows.

        Normal stripes use at most 8 MiB of assignment data. Extremely wide
        ensembles stream one precinct in batches rather than allocate a whole row.
        """
        n = self.count
        budget = 8 * 1024 * 1024
        stripe = max(1, min(256, budget // max(4 * n, 1)))
        with self.connect() as db, path.open('w', newline='', encoding='utf-8') as out, \
                self.binary_path.open('rb') as binary:
            import io
            cell = io.StringIO(newline='')
            csv.writer(cell).writerow([id_col])
            out.write(cell.getvalue().removesuffix('\r\n'))
            for (rid,) in db.execute('SELECT run_id FROM runs WHERE seq<=? ORDER BY seq', (n,)):
                cell.seek(0)
                cell.truncate()
                csv.writer(cell).writerow([rid])
                out.write(',' + cell.getvalue().removesuffix('\r\n'))
            out.write('\n')
            for start in range(0, len(precinct_ids), stripe):
                height = min(stripe, len(precinct_ids) - start)
                if n * 4 <= budget:
                    tile = np.empty((n, height), dtype='<i4')
                    for i in range(n):
                        binary.seek((i * self.precinct_count + start) * 4)
                        blob = binary.read(height * 4)
                        if len(blob) != height * 4:
                            raise OSError('Incomplete ensemble assignment file')
                        tile[i] = np.frombuffer(blob, dtype='<i4')
                    for j in range(height):
                        cell.seek(0)
                        cell.truncate()
                        csv.writer(cell).writerow([precinct_ids[start + j]])
                        out.write(cell.getvalue().removesuffix('\r\n'))
                        for chunk in range(0, n, 4096):
                            values = tile[chunk:chunk + 4096, j].tolist()
                            out.write(',' + ','.join(map(str, values)))
                        out.write('\n')
                else:
                    cell.seek(0)
                    cell.truncate()
                    csv.writer(cell).writerow([precinct_ids[start]])
                    out.write(cell.getvalue().removesuffix('\r\n'))
                    for i in range(n):
                        binary.seek((i * self.precinct_count + start) * 4)
                        blob = binary.read(4)
                        if len(blob) != 4:
                            raise OSError('Incomplete ensemble assignment file')
                        out.write(',' + str(int.from_bytes(blob, 'little', signed=True)))
                    out.write('\n')
