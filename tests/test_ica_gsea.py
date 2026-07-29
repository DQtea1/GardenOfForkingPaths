from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src import ica_gsea
from src import run_pipeline
from src.report import _ica_metagene_gsea_payload


def _gsea_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Term": ["NEGATIVE_PATH", "POSITIVE_PATH"],
            "NES": [-1.8, 2.1],
            "NOM p-val": [0.01, 0.02],
            "FDR q-val": [0.03, 0.04],
            "Lead_genes": ["A;B", "C;D"],
        }
    )


def test_run_ica_metagene_gsea_covers_every_component_and_collection(
    tmp_path, monkeypatch
):
    metagenes = pd.DataFrame(
        [[-2.0, 0.2, 1.5], [1.0, -3.0, 0.5]],
        index=["IC01", "IC02"],
        columns=["A", "B", "C"],
    )
    collections = {}
    for name in ("h", "c2"):
        path = tmp_path / f"{name}.gmt"
        path.write_text("PATH\ttest\tA\tB\tC\n")
        collections[name] = str(path)

    calls = []

    def fake_prerank(scores, gene_sets, **kwargs):
        calls.append((str(scores.name), Path(gene_sets).stem))
        return _gsea_table()

    monkeypatch.setattr(ica_gsea, "gsea_prerank_scores", fake_prerank)
    results = ica_gsea.run_ica_metagene_gsea(
        metagenes,
        collections,
        tmp_path / "m6",
        min_size=1,
        n_jobs=1,
    )

    assert set(calls) == {
        ("IC01", "h"),
        ("IC01", "c2"),
        ("IC02", "h"),
        ("IC02", "c2"),
    }
    assert set(results) == {"IC01", "IC02"}
    assert all(set(results[component]) == {"h", "c2"} for component in results)
    manifest = pd.read_csv(
        tmp_path / "m6" / "metagene_gsea" / "gsea_run_manifest.csv"
    )
    assert len(manifest) == 4
    assert set(manifest["status"]) == {"ok"}


def test_pipeline_runs_gsea_for_every_selected_ica_dimension(
    tmp_path, monkeypatch
):
    selected = (6, 8, 10, 12)
    decompositions = {
        dimension: SimpleNamespace(
            metagenes=pd.DataFrame(
                [[-dimension, dimension]],
                index=["IC01"],
                columns=["A", "B"],
            )
        )
        for dimension in selected
    }
    gmt = tmp_path / "toy.gmt"
    gmt.write_text("PATH\ttest\tA\tB\n")
    calls = []

    def fake_run(metagenes, gene_sets, outdir, **kwargs):
        calls.append(int(Path(outdir).name.removeprefix("m")))
        return {str(component): {} for component in metagenes.index}

    monkeypatch.setattr(run_pipeline.ig, "run_ica_metagene_gsea", fake_run)
    context = SimpleNamespace(
        ica_result=SimpleNamespace(
            persisted_dimensions=selected,
            decompositions=decompositions,
            dimension_roles={
                6: ("mstd",),
                8: ("nearest_lower",),
                10: ("nearest_higher",),
                12: ("best_mean_stability",),
            },
        ),
        args=SimpleNamespace(
            run_ica_gsea="y",
            gsea_collections={"toy": str(gmt)},
            gsea_gene_sets=None,
            gsea_permutations=10,
            ica_gsea_min_size=1,
            ica_gsea_max_size=10,
            seed=0,
        ),
        log=SimpleNamespace(info=lambda *args, **kwargs: None,
                            warning=lambda *args, **kwargs: None),
        outdir=tmp_path,
        eff_n_jobs=1,
        ica_metagene_gsea={},
    )

    run_pipeline._ica_metagene_gsea(context)

    assert calls == list(selected)
    assert set(context.ica_metagene_gsea) == set(selected)


def test_report_payload_keeps_signed_nes_fdr_and_leading_edge():
    payload = _ica_metagene_gsea_payload(
        {"IC01": {"hallmark": _gsea_table()}}
    )

    negative, positive = payload["IC01"]["hallmark"]
    assert negative == {
        "term": "NEGATIVE_PATH",
        "NES": -1.8,
        "pvalue": 0.01,
        "padj": 0.03,
        "leadingEdge": "A;B",
    }
    assert positive["NES"] == 2.1
    assert positive["leadingEdge"] == "C;D"
