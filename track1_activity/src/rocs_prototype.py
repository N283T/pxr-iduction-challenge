"""Utilities for fold-safe ROCS prototype feature construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

ScoreMap = Mapping[str | int, Sequence[float]]


def select_extreme_prototypes(
    compound_ids: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    n_active: int,
    n_inactive: int,
) -> tuple[list[int], list[int]]:
    """Select high- and low-activity prototype compound IDs from a train fold."""
    train_ids = compound_ids[train_idx]
    train_y = y[train_idx]
    active_order = np.argsort(-train_y, kind="stable")[:n_active]
    inactive_order = np.argsort(train_y, kind="stable")[:n_inactive]
    active = [int(train_ids[i]) for i in active_order]
    inactive = [int(train_ids[i]) for i in inactive_order]
    return active, inactive


def _scores_for_queries(score_map: ScoreMap | None, query_ids: Sequence[int]) -> np.ndarray:
    rows: list[list[float]] = []
    if score_map is None:
        return np.zeros((0, 3), dtype=np.float32)
    for qid in query_ids:
        value = score_map.get(qid)
        if value is None:
            value = score_map.get(str(qid))
        if value is None or len(value) < 3:
            continue
        rows.append([float(value[0]), float(value[1]), float(value[2])])
    if not rows:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def _top_mean(values: np.ndarray, k: int) -> float:
    if values.size == 0:
        return 0.0
    kk = min(k, values.size)
    top = np.partition(values, -kk)[-kk:]
    return float(np.mean(top))


def _weighted_knn_pec50(
    combo_scores: np.ndarray,
    query_ids: Sequence[int],
    present_mask: np.ndarray,
    query_targets: Mapping[int, float],
    k: int,
) -> float:
    if combo_scores.size == 0:
        return 0.0
    present_query_ids = [int(q) for q, ok in zip(query_ids, present_mask, strict=True) if ok]
    if not present_query_ids:
        return 0.0
    kk = min(k, combo_scores.size)
    order = np.argsort(-combo_scores, kind="stable")[:kk]
    weights = combo_scores[order].astype(np.float64)
    targets = np.asarray([query_targets[present_query_ids[i]] for i in order], dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 1e-12:
        return float(np.mean(targets))
    return float(np.sum(weights * targets) / denom)


def complete_score_maps(
    target_ids: Sequence[int],
    score_maps: Mapping[int, ScoreMap],
) -> dict[int, ScoreMap]:
    """Return score maps for every target, using an empty map for uncovered targets."""
    return {int(target_id): score_maps.get(int(target_id), {}) for target_id in target_ids}


def build_dense_query_features(
    target_ids: Sequence[int],
    score_maps: Mapping[int, ScoreMap],
    query_ids: Sequence[int],
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    """Build query-by-score ROCS features with fixed columns per query ID."""
    query_ids = [int(q) for q in query_ids]
    names = [
        f"{prefix}_q{qid}_{score_name}"
        for qid in query_ids
        for score_name in ("shape", "color", "combo")
    ]
    rows: list[list[float]] = []
    for target_id in target_ids:
        score_map = score_maps.get(int(target_id), {})
        row: list[float] = []
        for qid in query_ids:
            value = score_map.get(qid) or score_map.get(str(qid))
            if value is None or len(value) < 3:
                row.extend([0.0, 0.0, 0.0])
            else:
                row.extend([float(value[0]), float(value[1]), float(value[2])])
        rows.append(row)
    X = np.asarray(rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def build_prototype_features(
    target_ids: Sequence[int],
    score_maps: Mapping[int, ScoreMap],
    query_ids: Sequence[int],
    query_targets: Mapping[int, float],
    prefix: str,
    top_ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> tuple[np.ndarray, list[str]]:
    """Build fixed-width summary features from query-specific ROCS scores.

    ``score_maps`` maps target compound ID to a per-query map where values are
    ``[shape_tanimoto, color_tanimoto, tanimoto_combo]``. Missing query scores
    are treated as absent, not zero, for top-k and mean summaries.
    """
    score_names = ("shape", "color", "combo")
    names: list[str] = []
    for score_name in score_names:
        names.extend(
            [
                f"{prefix}_{score_name}_max",
                f"{prefix}_{score_name}_mean",
                f"{prefix}_{score_name}_std",
            ]
        )
        names.extend(f"{prefix}_{score_name}_top{k}_mean" for k in top_ks)
    names.extend(f"{prefix}_knn_k{k}_pec50" for k in top_ks)
    names.extend([f"{prefix}_query_count", f"{prefix}_query_coverage"])

    query_ids = [int(q) for q in query_ids]
    rows: list[list[float]] = []
    for target_id in target_ids:
        raw = score_maps.get(int(target_id))
        scores = _scores_for_queries(raw, query_ids)
        present: list[bool] = []
        if raw is None:
            present = [False] * len(query_ids)
        else:
            for qid in query_ids:
                present.append(qid in raw or str(qid) in raw)
        present_mask = np.asarray(present, dtype=bool)

        row: list[float] = []
        for col in range(3):
            values = scores[:, col] if scores.size else np.zeros(0, dtype=np.float32)
            if values.size:
                row.extend([float(np.max(values)), float(np.mean(values)), float(np.std(values))])
            else:
                row.extend([0.0, 0.0, 0.0])
            row.extend(_top_mean(values, k) for k in top_ks)
        combo_scores = scores[:, 2] if scores.size else np.zeros(0, dtype=np.float32)
        row.extend(
            _weighted_knn_pec50(combo_scores, query_ids, present_mask, query_targets, k)
            for k in top_ks
        )
        query_count = float(scores.shape[0])
        row.extend([query_count, query_count / max(len(query_ids), 1)])
        rows.append(row)

    return np.asarray(rows, dtype=np.float32), names
