"""Query-only V5 candidate evaluation for P1, P2, and P3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .clustering import fit_robust_attractor
from .insertion import BoundaryManifest, materialize_views
from .interfaces import Candidate, ClusterStructure, TextEmbeddingOracle
from .occupancy import cosine_distance_to_centers, fixed_structure_coverage


@dataclass
class EvaluationBundle:
    record: dict[str, Any]
    structures: dict[str, ClusterStructure]
    view_embeddings: dict[str, np.ndarray]
    clean_embeddings: np.ndarray


def candidate_seed(base_seed: int, candidate_key: str, task: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{task}\0{candidate_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _fixed_pairs(size: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    left = rng.integers(0, size, size=count, dtype=np.int64)
    right = rng.integers(0, size - 1, size=count, dtype=np.int64)
    right += right >= left
    return left, right


def _diagnostics(clean: np.ndarray, triggered: np.ndarray, seed: int) -> dict[str, float]:
    if len(triggered) != len(clean):
        repeats = len(triggered) // len(clean)
        clean = np.tile(clean, (repeats, 1))
    paired = np.maximum(0.0, 1.0 - np.sum(clean * triggered, axis=1))
    left, right = _fixed_pairs(len(triggered), min(4096, max(1, len(triggered) * 4)), seed)
    triggered_pairs = np.maximum(0.0, 1.0 - np.sum(triggered[left] * triggered[right], axis=1))
    clean_pairs = np.maximum(0.0, 1.0 - np.sum(clean[left] * clean[right], axis=1))
    return {
        "displacement_q05": float(np.quantile(paired, 0.05)),
        "displacement_mean": float(np.mean(paired)),
        "global_contraction_q90_ratio": float(
            np.quantile(triggered_pairs, 0.90) / max(np.quantile(clean_pairs, 0.90), 1e-12)
        ),
    }


def _support_diagnostics(centers: np.ndarray, benign_reference: np.ndarray, k: int = 10) -> dict[str, float]:
    distances = cosine_distance_to_centers(centers, benign_reference)
    kth = np.partition(distances, min(k - 1, distances.shape[1] - 1), axis=1)[:, min(k - 1, distances.shape[1] - 1)]
    return {
        "support_k10_min": float(np.min(kth)),
        "support_k10_median": float(np.median(kth)),
        "support_k10_max": float(np.max(kth)),
    }


def _logical_groups(view_embeddings: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    random_values = []
    for view, values in view_embeddings.items():
        if view.startswith("random_r"):
            random_values.append(values)
        else:
            result[view] = values
    if random_values:
        result["random"] = np.concatenate(random_values, axis=0)
    return result

class CandidateEvaluator:
    def __init__(
        self,
        *,
        oracle: TextEmbeddingOracle,
        frame: pd.DataFrame,
        clean_embeddings: np.ndarray,
        benign_probe: np.ndarray,
        benign_reference: np.ndarray,
        manifest: BoundaryManifest,
        role: str,
        task: str,
        config: Mapping[str, Any],
        active_minimum_coverage: float,
        active_maximum_outlier_rate: float,
    ) -> None:
        self.oracle = oracle
        self.frame = frame.reset_index(drop=True)
        self.clean = np.asarray(clean_embeddings, dtype=np.float32)
        self.benign_probe = np.asarray(benign_probe, dtype=np.float32)
        self.benign_reference = np.asarray(benign_reference, dtype=np.float32)
        self.manifest = manifest
        self.role = role
        self.task = task
        self.config = config
        self.active_minimum_coverage = float(active_minimum_coverage)
        self.active_maximum_outlier_rate = float(active_maximum_outlier_rate)

    def _views(self, candidate: Candidate) -> dict[str, np.ndarray]:
        insertion = self.config["insertion"]
        texts = materialize_views(
            self.frame,
            candidate.trigger,
            self.task,
            role=self.role,
            manifest=self.manifest,
            random_replicates=int(insertion["random_replicates"]),
            separator=str(insertion["separator"]),
        )
        return {view: self.oracle.encode(values) for view, values in texts.items()}

    def _fit(self, candidate: Candidate, views: dict[str, np.ndarray]) -> dict[str, ClusterStructure]:
        seed = candidate_seed(int(self.config["seed"]), candidate.key, self.task)
        logical = _logical_groups(views)
        if self.task == "conditional":
            return {
                name: fit_robust_attractor(
                    values,
                    self.benign_probe,
                    self.config,
                    seed=seed + index * 100003,
                    minimum_coverage=self.active_minimum_coverage,
                    maximum_outlier_rate=self.active_maximum_outlier_rate,
                )
                for index, (name, values) in enumerate(sorted(logical.items()))
            }
        if self.task == "shared":
            pooled = np.concatenate(list(views.values()), axis=0)
            return {
                "shared": fit_robust_attractor(
                    pooled,
                    self.benign_probe,
                    self.config,
                    seed=seed,
                    minimum_coverage=self.active_minimum_coverage,
                    maximum_outlier_rate=self.active_maximum_outlier_rate,
                )
            }
        pooled = np.concatenate(list(views.values()), axis=0)
        return {
            self.task: fit_robust_attractor(
                pooled,
                self.benign_probe,
                self.config,
                seed=seed,
                minimum_coverage=self.active_minimum_coverage,
                maximum_outlier_rate=self.active_maximum_outlier_rate,
            )
        }

    def _position_records(
        self, views: Mapping[str, np.ndarray], structures: Mapping[str, ClusterStructure]
    ) -> tuple[dict[str, dict[str, float | int]], float]:
        confidence = float(self.config["validation"]["confidence"])
        records: dict[str, dict[str, float | int]] = {}
        if self.task == "shared":
            structure = structures["shared"]
            for view, values in views.items():
                records[view] = fixed_structure_coverage(
                    values, structure.centers, structure.radii, confidence=confidence
                )
        elif self.task == "conditional":
            for view, values in views.items():
                logical = "random" if view.startswith("random_r") else view
                structure = structures[logical]
                records[view] = fixed_structure_coverage(
                    values, structure.centers, structure.radii, confidence=confidence
                )
        else:
            structure = structures[self.task]
            for view, values in views.items():
                records[view] = fixed_structure_coverage(
                    values, structure.centers, structure.radii, confidence=confidence
                )
        worst = min(float(record["coverage"]) for record in records.values())
        return records, worst

    def evaluate(self, candidate: Candidate, *, retain_embeddings: bool = False) -> EvaluationBundle:
        base = {
            "candidate_key": candidate.key,
            "token_ids": candidate.key,
            "trigger": candidate.trigger,
            "actual_token_length": int(candidate.actual_token_length),
            "task": self.task,
            "role": self.role,
            "exact_token_roundtrip": bool(candidate.exact_token_roundtrip),
        }
        try:
            views = self._views(candidate)
            structures = self._fit(candidate, views)
        except (ValueError, RuntimeError, FloatingPointError) as error:
            record = {
                **base,
                "evaluation_error": f"{type(error).__name__}: {error}",
                "occupancy_auc": 10.0,
                "cmax": 10.0,
                "cavg": 10.0,
                "coverage": 0.0,
                "worst_position_coverage": 0.0,
                "outlier_rate": 1.0,
                "cluster_count": 0,
                "minimum_cluster_mass": 0.0,
                "lambda_star": 0.0,
                "constraint_violations": {
                    "fit": 1.0,
                    "coverage": self.active_minimum_coverage,
                    "outlier_rate": 1.0 - self.active_maximum_outlier_rate,
                },
            }
            return EvaluationBundle(record, {}, {}, self.clean if retain_embeddings else np.empty((0, 0)))

        position_records, worst_position_coverage = self._position_records(views, structures)
        values = list(structures.values())
        coverage = min(value.coverage for value in values)
        outlier_rate = max(value.outlier_rate for value in values)
        minimum_mass = min(float(np.min(value.masses)) for value in values)
        cmax = max(value.cmax for value in values)
        cavg = max(value.cavg for value in values)
        occupancy_auc = max(value.occupancy_auc for value in values)
        lambda_star = min(value.lambda_star for value in values)
        per_position_required = float(self.config["structure"]["minimum_per_position_coverage"])
        minimum_cluster_mass = float(self.config["structure"]["minimum_cluster_inlier_mass"])
        violations = {
            "coverage": max(0.0, self.active_minimum_coverage - coverage),
            "position_coverage": max(0.0, per_position_required - worst_position_coverage),
            "outlier_rate": max(0.0, outlier_rate - self.active_maximum_outlier_rate),
            "cluster_mass": max(0.0, minimum_cluster_mass - minimum_mass),
        }
        all_triggered = np.concatenate(list(views.values()), axis=0)
        diagnostics = _diagnostics(self.clean, all_triggered, candidate_seed(int(self.config["seed"]), candidate.key, self.task))
        all_centers = np.concatenate([value.centers for value in values], axis=0)
        support = _support_diagnostics(all_centers, self.benign_reference)
        if self.task == "shared":
            clean_for_region = np.tile(self.clean, (len(views), 1))
            clean_region = fixed_structure_coverage(
                clean_for_region,
                structures["shared"].centers,
                structures["shared"].radii,
                confidence=float(self.config["validation"]["confidence"]),
            )["coverage"]
        else:
            clean_coverages = []
            for name, structure in structures.items():
                repeats = int(np.ceil(len(structure.assignments) / len(self.clean)))
                clean_values = np.tile(self.clean, (repeats, 1))[: len(structure.assignments)]
                clean_coverages.append(
                    fixed_structure_coverage(
                        clean_values,
                        structure.centers,
                        structure.radii,
                        confidence=float(self.config["validation"]["confidence"]),
                    )["coverage"]
                )
            clean_region = max(map(float, clean_coverages))
        record = {
            **base,
            "evaluation_error": "",
            "occupancy_auc": occupancy_auc,
            "cmax": cmax,
            "cavg": cavg,
            "coverage": coverage,
            "worst_position_coverage": worst_position_coverage,
            "outlier_rate": outlier_rate,
            "cluster_count": max(value.cluster_count for value in values),
            "total_conditional_cluster_count": sum(value.cluster_count for value in values),
            "minimum_cluster_mass": minimum_mass,
            "lambda_star": lambda_star,
            "attraction_gain": coverage - float(clean_region),
            "clean_region_coverage": float(clean_region),
            "constraint_violations": violations,
            "position_records": position_records,
            "structure_summaries": {name: value.summary() for name, value in structures.items()},
            **diagnostics,
            **support,
        }
        return EvaluationBundle(
            record,
            structures,
            views if retain_embeddings else {},
            self.clean if retain_embeddings else np.empty((0, 0), dtype=np.float32),
        )


def flatten_record(record: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in record.items():
        if isinstance(value, (dict, list, tuple)):
            import json

            result[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            result[key] = value
    return result
