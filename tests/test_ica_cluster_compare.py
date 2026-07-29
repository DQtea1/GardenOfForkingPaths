import math

import numpy as np
import pandas as pd
import pytest

from src.ica_cluster_compare import compare_ica_clusters


def _metasamples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "IC01": [0.0, 2.0, 4.0, 6.0, 1.0, 3.0],
            "IC02": [0.0, 0.0, 0.0, 0.0, 3.0, 5.0],
        },
        index=[f"S{i}" for i in range(1, 7)],
    )


def _row(rows: list[dict], component: str, a: str, b: str) -> dict:
    return next(
        row
        for row in rows
        if row["component"] == component
        and row["cluster_a"] == a
        and row["cluster_b"] == b
    )


def test_all_pairs_methods_distances_and_exports(tmp_path):
    result = compare_ica_clusters(
        _metasamples(),
        {3: np.array([1, 1, 2, 2, 3, 3])},
        tmp_path,
        min_cluster_size=2,
        clustering_method="kmeans",
    )

    assert result["settings"]["clustering_method"] == "kmeans"
    per_k = result["perK"]["3"]
    assert per_k["n_clusters"] == 3
    assert per_k["n_pairs"] == 3
    assert all(len(rows) == 3 * 2 for rows in per_k["componentwise"].values())

    centroid = _row(per_k["componentwise"]["centroid"], "IC01", "1", "2")
    assert centroid["representative_a"] == pytest.approx(1.0)
    assert centroid["representative_b"] == pytest.approx(5.0)
    assert centroid["signed_difference"] == pytest.approx(-4.0)
    assert centroid["absolute_difference"] == pytest.approx(4.0)
    assert centroid["sd_a"] == pytest.approx(math.sqrt(2.0))

    # Deux observations équidistantes : le premier identifiant est le médoïde,
    # conformément à la règle de départage déterministe documentée.
    medoid = _row(per_k["componentwise"]["medoid"], "IC01", "1", "2")
    assert medoid["representative_sample_a"] == "S1"
    assert medoid["representative_sample_b"] == "S3"
    assert medoid["signed_difference"] == pytest.approx(-4.0)

    cohen = _row(per_k["componentwise"]["cohen"], "IC01", "1", "2")
    assert cohen["pooled_sd"] == pytest.approx(math.sqrt(2.0))
    assert cohen["effect_size"] == pytest.approx(-4.0 / math.sqrt(2.0))
    assert cohen["hedges_correction"] == pytest.approx(1.0 - 3.0 / 7.0)
    assert cohen["hedges_g"] == pytest.approx(
        cohen["effect_size"] * cohen["hedges_correction"]
    )
    assert cohen["absolute_difference"] == pytest.approx(4.0)
    assert cohen["absolute_effect_size"] == pytest.approx(
        abs(cohen["effect_size"])
    )

    assert len(per_k["global"]) == 3
    for pair in per_k["global"]:
        contributions = [
            row
            for row in per_k["contributions"]
            if row["cluster_a"] == pair["cluster_a"]
            and row["cluster_b"] == pair["cluster_b"]
        ]
        assert pair["distance"] ** 2 == pytest.approx(
            sum(row["quadratic_contribution"] for row in contributions)
        )
        if pair["distance"] > 0:
            assert sum(row["relative_contribution"] for row in contributions) == (
                pytest.approx(1.0)
            )

    export_dir = tmp_path / "cluster_comparisons" / "k3"
    assert (export_dir / "componentwise_centroid.csv").exists()
    assert (export_dir / "componentwise_medoid.csv").exists()
    assert (export_dir / "componentwise_cohen.csv").exists()
    assert (export_dir / "global_standardized_distance.csv").exists()
    assert (export_dir / "global_component_contributions.csv").exists()


def test_non_calculable_cohen_and_global_component_are_explicit(tmp_path):
    matrix = pd.DataFrame(
        {
            "IC_constant": [1.0, 1.0, 1.0],
            "IC_missing": [0.0, np.nan, 2.0],
        },
        index=["S1", "S2", "S3"],
    )
    result = compare_ica_clusters(
        matrix,
        {2: np.array([1, 2, 2])},
        tmp_path,
        min_cluster_size=2,
    )
    per_k = result["perK"]["2"]

    constant = _row(
        per_k["componentwise"]["cohen"], "IC_constant", "1", "2"
    )
    assert constant["effect_size"] is None
    assert "moins de 2 valeurs valides" in constant["warning"]
    assert "petit cluster" in constant["warning"]

    settings = per_k["global_settings"]
    assert settings["n_components_total"] == 2
    assert "IC_constant" in settings["excluded_components"]
    assert any(row["warning"] for row in per_k["global"])


def test_rejects_misaligned_labels(tmp_path):
    with pytest.raises(ValueError, match="labels"):
        compare_ica_clusters(
            _metasamples(),
            {3: np.array([1, 1, 2])},
            tmp_path,
        )


def test_every_tested_k_is_computed(tmp_path):
    result = compare_ica_clusters(
        _metasamples(),
        {
            2: np.array([1, 1, 1, 2, 2, 2]),
            3: np.array([1, 1, 2, 2, 3, 3]),
        },
        tmp_path,
        min_cluster_size=2,
    )

    assert set(result["perK"]) == {"2", "3"}
    assert result["perK"]["2"]["n_pairs"] == 1
    assert result["perK"]["3"]["n_pairs"] == 3
