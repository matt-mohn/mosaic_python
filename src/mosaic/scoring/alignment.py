"""
Alignment — how close the proposed plan stays to a reference ("alternative")
plan loaded from a CSV. This is the least-change / core-retention axis: a real
legal criterion in several states (keep the new map close to the enacted one).

District NUMBERING is irrelevant — the score depends only on the partition the
reference induces, not its labels.

Direction is one-sided BY DESIGN. It measures how each reference district spreads
across the proposed districts (splitting), never the reverse (proposed-district
purity). So merging/coarsening is unpenalized: a reference district absorbed whole
into a neighbour reads as cohesive, and the degenerate "everything in one proposed
district" map scores a perfect 0. That is correct for a least-change axis — it
punishes breaking old communities apart, not collapsing them together — and is
safe in practice because population balance and the other weighted terms forbid
the degenerate. If symmetric core-retention were ever wanted, add the reverse
Herfindahl (proposed-district purity) as a separate term.

How it scores: cohesion (Herfindahl)
------------------------------------
For each reference district a, look at how its mass spreads across the proposed
districts (fractions f_{a,p} summing to 1) and measure how concentrated it stayed:

    cohesion_a = SUM_p f_{a,p}^2          # = probability two of a's residents are
                                          #   still in the same proposed district
    penalty    = 100 * SUM_a w_a * (1 - cohesion_a) / SUM_a w_a

cohesion is 1.0 when the district survived whole and falls as its residents are
split across more proposed districts. Because it uses the FULL distribution (not
just the biggest surviving chunk), it correctly distinguishes a clean split from
a shatter — a district halved cleanly scores better than one whose second half is
pulverized. No label matching is needed (it's relabel-invariant by construction),
so there are no Hungarian-matching discontinuities.

w_a is the reference district's mass in the chosen weight (population for plain
alignment; a party's votes for partisan alignment). District NUMBERING and the
gain/loss of seats fall out naturally: a brand-new proposed district has no
reference row, and a reference district cleanly absorbed into a neighbour reads
as cohesive (its people stayed together).

Penalty form: float in [0, 100], 0 = best (every reference district intact).
Baked x100 puts it in the same band as Polsby-Popper / Reock, so weight_alignment
is directly comparable to those (default 25).

Per-iteration cost: one numba pass builds the (n_alt x n_prop) overlap matrix +
a length-k reduction. Same cost class as Reock; cheaper than before (no Hungarian).

History: earlier versions used a "biggest surviving chunk" retention with an
optimal (Hungarian) label match and a convex exponent to encode a spread-the-
change preference. That was dropped — it was blind to displaced-voter cohesion
(clean split == shatter), had a near-flat gradient near alignment, and an NC/GA
bake-off showed Herfindahl reaches materially tighter alignment at equal effort.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from numba import njit

log = logging.getLogger("mosaic")


class AlignmentError(ValueError):
    """Raised when the reference plan CSV cannot be loaded/joined."""


@dataclass
class AlignmentData:
    """Reference plan, aligned to gdf row order.

    alt_assignment      -- (n,) int64 zero-indexed reference district per precinct
                           (int64 so the hot-path overlap pass never re-copies it)
    n_alt_districts     -- number of districts in the reference plan
    filename            -- source CSV name (for GUI display)
    alt_dem_by_district -- (n_alt,) D two-party votes per reference district, or
                           None if no election data was available at load. Used
                           to select which reference districts a party "wins"
                           (Ask 2) without recomputing per iteration.
    alt_gop_by_district -- (n_alt,) R two-party votes per reference district, or None.
    alt_labels          -- (n_alt,) original district numbers from the CSV, indexed
                           by the densified 0..n_alt-1 id. Lets "Infer from
                           alignment" renumbering adopt the reference's own
                           numbers instead of the densified indices.
    """
    alt_assignment: np.ndarray
    n_alt_districts: int
    filename: str = ""
    alt_dem_by_district: Optional[np.ndarray] = None
    alt_gop_by_district: Optional[np.ndarray] = None
    alt_labels: Optional[np.ndarray] = None


def precompute_alignment_data(
    path: str | Path,
    *,
    gdf: gpd.GeoDataFrame,
    gdf_id_col: str,
    csv_id_col: str,
    csv_district_col: str,
    dem_votes: Optional[np.ndarray] = None,
    gop_votes: Optional[np.ndarray] = None,
) -> AlignmentData:
    """Load a reference plan CSV and align it to gdf row order.

    Mirrors hot_start's id-matching (GEOID-as-string join + reorder) but
    deliberately drops the contiguity, population-tolerance, and matching-
    district-count checks: a reference map is an external artifact and may
    legitimately have a different district count (that is the whole point of
    the spread-vs-concentrate behaviour). All we require is a clean join.

    Raises AlignmentError with a user-facing message on any join failure.
    """
    path = Path(path)
    if not path.is_file():
        raise AlignmentError(f"File not found: {path}")

    # ID column as string so leading zeros survive (GEOID '08001...' must not
    # be coerced to int and silently break the join against the shapefile).
    try:
        df = pd.read_csv(path, dtype={csv_id_col: str})
    except Exception as e:
        raise AlignmentError(f"Could not read CSV: {e}") from e

    for col, kind in ((csv_id_col, "ID"), (csv_district_col, "district")):
        if col not in df.columns:
            raise AlignmentError(
                f"CSV does not have the selected {kind} column '{col}'. "
                f"Available: {list(df.columns)}"
            )

    n = len(gdf)
    if len(df) != n:
        raise AlignmentError(
            f"CSV has {len(df)} rows; shapefile has {n} precincts. "
            f"Row count must match exactly."
        )

    gdf_ids = gdf[gdf_id_col].astype(str).str.strip().values
    csv_ids = df[csv_id_col].astype(str).str.strip().values
    if set(gdf_ids) != set(csv_ids):
        missing = set(gdf_ids) - set(csv_ids)
        extra = set(csv_ids) - set(gdf_ids)
        bits = []
        if missing:
            bits.append(
                f"shapefile has {len(missing)} IDs not in CSV "
                f"(e.g. {sorted(missing)[:3]})"
            )
        if extra:
            bits.append(
                f"CSV has {len(extra)} IDs not in shapefile "
                f"(e.g. {sorted(extra)[:3]})"
            )
        raise AlignmentError(
            "Precinct IDs in reference CSV do not match shapefile: "
            + "; ".join(bits)
        )

    # Reorder CSV rows to gdf order so the assignment aligns to score inputs.
    order_map = {pid: i for i, pid in enumerate(gdf_ids)}
    df = df.assign(
        _pos=df[csv_id_col].astype(str).str.strip().map(order_map)
    ).sort_values("_pos")
    raw = df[csv_district_col].values

    if not np.issubdtype(raw.dtype, np.number):
        try:
            raw = raw.astype(np.int64)
        except (ValueError, TypeError) as e:
            raise AlignmentError(f"'district' column is not numeric: {e}") from e
    if np.isnan(raw.astype(np.float64)).any():
        raise AlignmentError("'district' column contains NaN / missing values")

    raw = raw.astype(np.int64)
    # Densify labels to a contiguous 0..k-1 range; we don't care what the
    # reference plan's labels are, only the partition they induce.
    # int64 (not int32): alt_assignment is static and feeds the numba overlap
    # pass every scored proposal, so storing it in the dtype the kernel wants
    # lets score_alignment's astype(copy=False) skip a per-iteration copy.
    # Keep the original labels (uniq) alongside the densified ids so the GUI can
    # renumber a plan to the reference's own district numbers ("Infer from
    # alignment"); the score itself only needs the partition.
    uniq, alt = np.unique(raw, return_inverse=True)
    alt = alt.astype(np.int64)
    alt_labels = uniq.astype(np.int64)
    n_alt = int(alt.max()) + 1

    # Per-reference-district two-party vote totals (if election data is present),
    # so the GUI can restrict scoring to the districts a party wins (Ask 2)
    # without recomputing per iteration. Votes are in gdf row order, same as alt.
    alt_dem = alt_gop = None
    if dem_votes is not None and gop_votes is not None \
            and len(dem_votes) == n and len(gop_votes) == n:
        alt_dem = np.bincount(alt, weights=np.asarray(dem_votes, dtype=np.float64),
                              minlength=n_alt)
        alt_gop = np.bincount(alt, weights=np.asarray(gop_votes, dtype=np.float64),
                              minlength=n_alt)

    log.info(
        f"Alignment reference loaded: {n} precincts, {n_alt} districts "
        f"(from {path.name}){' with votes' if alt_dem is not None else ''}"
    )
    return AlignmentData(
        alt_assignment=alt, n_alt_districts=n_alt, filename=path.name,
        alt_dem_by_district=alt_dem, alt_gop_by_district=alt_gop,
        alt_labels=alt_labels,
    )


@njit(cache=True)
def _overlap_matrix(
    alt: np.ndarray,
    prop: np.ndarray,
    pops: np.ndarray,
    n_alt: int,
    n_prop: int,
) -> np.ndarray:
    """Population-overlap matrix M[a, p] = pop in (alt district a AND proposed
    district p). One pass over precincts.
    """
    m = np.zeros((n_alt, n_prop), dtype=np.float64)
    for i in range(alt.shape[0]):
        m[alt[i], prop[i]] += pops[i]
    return m


def score_alignment(
    assignment: np.ndarray,
    alt_assignment: np.ndarray,
    weights: np.ndarray,
    n_alt_districts: int,
    n_districts: int,
    district_mask: Optional[np.ndarray] = None,
    return_components: bool = False,
):
    """Herfindahl cohesion penalty against the reference plan.

    Args:
        assignment:       (n,) proposed district indices 0..n_districts-1
        alt_assignment:   (n,) reference district indices 0..n_alt_districts-1
        weights:          (n,) per-precinct weight cohesion is measured in.
                          Population for plain alignment; a party's votes for
                          partisan alignment (Ask 1).
        n_alt_districts:  number of reference districts
        n_districts:      number of proposed districts
        district_mask:    optional (n_alt,) bool selecting which reference
                          districts to score (Ask 2). None = all. Unselected
                          reference districts simply don't contribute.
        return_components: also return (mean_cohesion_pct, min_cohesion_pct)

    Returns:
        penalty in [0, 100] (0 = identical), or
        (penalty, mean_cohesion_pct, min_cohesion_pct) when return_components=True.
    """
    # copy=False: skip the allocation when the input already has the kernel's
    # dtype. alt_assignment is stored int64 at load, so it never re-copies; the
    # proposed assignment / weights copy only if they arrive in another dtype.
    m = _overlap_matrix(
        alt_assignment.astype(np.int64, copy=False),
        assignment.astype(np.int64, copy=False),
        weights.astype(np.float64, copy=False),
        n_alt_districts,
        n_districts,
    )
    pop_a = m.sum(axis=1)                       # (n_alt,) weighted ref-district mass

    # Restrict to the selected reference districts (Ask 2).
    if district_mask is not None:
        rows = np.where(district_mask)[0]
    else:
        rows = np.arange(n_alt_districts)
    if len(rows) == 0:
        return (0.0, 100.0, 100.0) if return_components else 0.0

    sub = m[rows, :]                            # (R, n_prop)
    sub_pop = pop_a[rows]
    total_pop = sub_pop.sum()
    if total_pop <= 0:
        return (0.0, 100.0, 100.0) if return_components else 0.0

    # Cohesion_a = sum_p f_ap^2 = P(two of district a's residents still together).
    # Full-distribution, label-free: a clean split scores better than a shatter.
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(sub_pop[:, None] > 0, sub / sub_pop[:, None], 0.0)
    cohesion = np.sum(f * f, axis=1)            # (R,) in [0, 1]
    penalty = 100.0 * np.sum(sub_pop * (1.0 - cohesion)) / total_pop

    if not return_components:
        return float(penalty)

    mean_ret = 100.0 * np.sum(sub_pop * cohesion) / total_pop
    nonempty = sub_pop > 0
    min_ret = 100.0 * float(cohesion[nonempty].min()) if nonempty.any() else 100.0
    return float(penalty), float(mean_ret), float(min_ret)
