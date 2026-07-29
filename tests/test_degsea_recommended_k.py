from types import SimpleNamespace

import pandas as pd

from src import report
from src import run_pipeline


class _Log:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _context(tmp_path, *, all_k="n"):
    args = SimpleNamespace(
        run_degsea="y",
        gsea_collections={"h": "unused.gmt"},
        gsea_gene_sets=None,
        degsea_all_k=all_k,
        degsea_mode="ova",
        gsea_permutations=10,
        gsea_heatmap_pval=0.05,
        seed=0,
        min_cluster_size=10,
        k_criterion="pac",
    )
    branch = SimpleNamespace(
        k_values=(3, 4, 5),
        k_final=3,
        result=object(),
    )
    return SimpleNamespace(
        args=args,
        log=_Log(),
        outdir=tmp_path,
        primary=branch,
        eff_n_jobs=1,
        raw=object(),
        nes=None,
        degsea_by_k={},
    )


def test_degsea_single_k_targets_recommended_not_lowest_pac(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, all_k="n")
    calls = []
    heatmaps = []
    matrix = pd.DataFrame({"C1": [1.0]}, index=["PATH"])

    monkeypatch.setattr(
        run_pipeline.mt,
        "suggest_k",
        lambda result, min_cluster_size, method: 4,
    )
    monkeypatch.setattr(
        run_pipeline,
        "_degsea",
        lambda c, k, gene_sets, output_subdir: (
            calls.append((k, output_subdir)) or {"h": matrix}
        ),
    )
    monkeypatch.setattr(
        run_pipeline.pl,
        "plot_gsea_ova_heatmap",
        lambda *args, **kwargs: heatmaps.append((args, kwargs)),
    )

    run_pipeline._degsea_all_k(context)

    assert calls == [(4, "")]
    assert set(context.degsea_by_k) == {4}
    assert len(heatmaps) == 1
    # La synthèse principale est sur k_final=3 : ne pas lui fournir les NES de
    # la partition recommandée k=4, qui ne sont pas alignés sur ses clusters.
    assert context.nes is None


def test_degsea_all_k_keeps_all_partitions_and_recommended_heatmap(
    tmp_path, monkeypatch
):
    context = _context(tmp_path, all_k="y")
    calls = []
    heatmap_k = []

    monkeypatch.setattr(
        run_pipeline.mt,
        "suggest_k",
        lambda result, min_cluster_size, method: 4,
    )

    def fake_degsea(c, k, gene_sets, output_subdir):
        calls.append((k, output_subdir))
        return {"h": pd.DataFrame({f"C{k}": [float(k)]}, index=["PATH"])}

    monkeypatch.setattr(run_pipeline, "_degsea", fake_degsea)
    monkeypatch.setattr(
        run_pipeline.pl,
        "plot_gsea_ova_heatmap",
        lambda matrix, *args, **kwargs: heatmap_k.append(float(matrix.iloc[0, 0])),
    )

    run_pipeline._degsea_all_k(context)

    assert calls == [(3, "k3"), (4, "k4"), (5, "k5")]
    assert set(context.degsea_by_k) == {3, 4, 5}
    assert heatmap_k == [4.0]
    assert float(context.nes.iloc[0, 0]) == 3.0


def test_report_prefers_recommended_k_when_available():
    assert report._degsea_default_k(3, 4, {4: {}}) == 4
    assert report._degsea_default_k(3, 4, {3: {}, 4: {}, 5: {}}) == 4
    assert report._degsea_default_k(3, 4, {3: {}}) == 3
