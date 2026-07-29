"""Configuration : le YAML doit subir exactement les mêmes contrôles que la CLI.

Le défaut historique : les valeurs du YAML étaient recopiées telles quelles dans
le ``Namespace``, sans passer par les ``choices`` ni les ``type`` d'argparse. Une
faute de frappe survivait jusqu'aux workers joblib — après le prétraitement et le
scan ICA — ou passait totalement inaperçue (``run_ica: maybe`` désactivait l'ICA
en silence).
"""

import textwrap

import pytest

from gardenofforks import config as cf


def _write(tmp_path, body: str, *, counts: str = "counts.csv"):
    """Écrit un YAML de test et le fichier de counts qu'il référence."""
    (tmp_path / counts).write_text("gene,s1,s2\nA,1,2\n")
    path = tmp_path / "config.yaml"
    path.write_text(
        f"counts: {tmp_path / counts}\n" + textwrap.dedent(body), encoding="utf-8"
    )
    return path


def _load(tmp_path, body: str, *extra_argv):
    return cf.load_config(["--config", str(_write(tmp_path, body)), *extra_argv])


# ----------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "line, needle",
    [
        ("base: hierarchcal", "hierarchical"),      # choices + suggestion
        ("metric: peason", "pearson"),
        ("run_ica: maybe", "run_ica"),              # sinon : ICA sautée en silence
        ("linkage: avarage", "linkage"),
        ("norm_method: vsst", "norm_method"),
        ("k_criterion: pack", "k_criterion"),
    ],
)
def test_yaml_values_are_checked_against_choices(tmp_path, line, needle):
    with pytest.raises(cf.ConfigError) as excinfo:
        _load(tmp_path, line)
    assert needle in str(excinfo.value)


def test_yaml_values_are_coerced_to_the_declared_type(tmp_path):
    args = _load(tmp_path, "k_max: '9'\nprop_genes: '0.5'")
    assert args.k_max == 9 and isinstance(args.k_max, int)
    assert args.prop_genes == 0.5 and isinstance(args.prop_genes, float)


def test_non_numeric_value_for_numeric_option_is_rejected(tmp_path):
    with pytest.raises(cf.ConfigError, match="n_resamples"):
        _load(tmp_path, "n_resamples: beaucoup")


@pytest.mark.parametrize(
    "body, needle",
    [
        ("k_min: 8\nk_max: 3", "plage de k"),
        ("prop_samples: 1.7", "prop_samples"),
        ("prop_genes: 0", "prop_genes"),
        ("k_min: 3\nk_max: 6\nk_final: 9", "k_final"),
        ("ica_n_components_min: 10\nica_n_components_max: 4", "ica_n_components_max"),
        ("ica_top_dimensions: 7", "ica_top_dimensions"),
        ("n_jobs: 0", "n_jobs"),
        ("outlier_min_explained_var: 1.0", "outlier_min_explained_var"),
        ("purity_threshold: 1.5", "purity_threshold"),
    ],
)
def test_cross_field_incoherences_are_refused(tmp_path, body, needle):
    with pytest.raises(cf.ConfigError, match=needle):
        _load(tmp_path, body)


def test_missing_counts_file_is_caught_before_any_computation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("counts: /introuvable/nulle_part.csv\n")
    with pytest.raises(cf.ConfigError, match="introuvable"):
        cf.load_config(["--config", str(path)])


def test_every_error_is_reported_in_one_pass(tmp_path):
    """Corriger un YAML ne doit pas demander une exécution par faute."""
    with pytest.raises(cf.ConfigError) as excinfo:
        _load(tmp_path, "base: hierarchcal\nmetric: peason\nk_min: 8\nk_max: 3")
    message = str(excinfo.value)
    for needle in ("base", "metric", "plage de k"):
        assert needle in message


def test_already_normalized_conflicts_with_raw_count_steps(tmp_path):
    with pytest.raises(cf.ConfigError, match="already-normalized"):
        _load(tmp_path, "already_normalized: true\nrun_degsea: y")


# ------------------------------------------------------------------ précédence
def test_cli_overrides_yaml_which_overrides_defaults(tmp_path):
    default_k_max = cf.build_parser().parse_args([]).k_max
    args = _load(tmp_path, "k_max: 9\nseed: 7")
    assert args.k_max == 9 and args.seed == 7        # YAML > défaut

    args = _load(tmp_path, "k_max: 9", "--k-max", "11")
    assert args.k_max == 11                          # CLI > YAML
    assert default_k_max != 11


def test_cli_wins_even_when_it_restates_the_default(tmp_path):
    """Cas limite qui justifiait l'ancien bricolage sur `parser._actions`.

    Passer explicitement une valeur qui vaut le défaut doit tout de même
    l'emporter sur le YAML : on ne peut donc pas déduire la provenance en
    comparant au défaut.
    """
    default_k_max = cf.build_parser().parse_args([]).k_max
    args = _load(tmp_path, f"k_max: 9", "--k-max", str(default_k_max))
    assert args.k_max == default_k_max


def test_load_config_does_not_mutate_the_parser(tmp_path):
    """L'ancienne détection écrasait les défauts du parser, en place."""
    before = vars(cf.build_parser().parse_args([]))
    _load(tmp_path, "k_max: 9")
    after = vars(cf.build_parser().parse_args([]))
    assert before == after


# ------------------------------------------------------------- clés inconnues
def test_typo_in_a_key_is_fatal_with_a_suggestion(tmp_path):
    with pytest.raises(cf.ConfigError, match="k_max"):
        _load(tmp_path, "k_maxx: 12")


def test_unrecognised_key_without_close_match_only_warns(tmp_path, caplog):
    """Le YAML sert aussi de brouillon pour des options à venir."""
    args = _load(tmp_path, "human_pathways: /quelque/part.gmt")
    assert args.k_max == cf.build_parser().parse_args([]).k_max
    assert "human_pathways" in caplog.text


# --------------------------------------------------------- blocs structurés
def test_gsea_collections_accepts_both_documented_forms(tmp_path):
    gmt = tmp_path / "h.gmt"
    gmt.write_text("SET\tna\tA\tB\n")
    args = _load(
        tmp_path,
        f"""
        gsea_collections:
          h: {gmt}
          c2:
            path: {gmt}
            enabled: false
        """,
    )
    assert args.gsea_collections == {"h": str(gmt)}   # c2 désactivée


def test_legacy_flat_load_keys_still_populate_collections(tmp_path):
    gmt = tmp_path / "sig.gmt"
    gmt.write_text("SET\tna\tA\n")
    args = _load(tmp_path, f"load_signatures_select: {gmt}")
    assert args.gsea_collections == {"signatures_select": str(gmt)}


def test_ordinal_variables_accept_dict_and_list(tmp_path):
    args = _load(tmp_path, "ordinal_variables:\n  stade: [I, II, III]")
    assert args.ordinal_variables == {"stade": ["I", "II", "III"]}
    args = _load(tmp_path, "ordinal_variables: [stade, grade]")
    assert args.ordinal_variables == {"stade": None, "grade": None}


def test_yaml_booleans_are_accepted_for_y_n_options(tmp_path):
    """PyYAML lit `yes`/`no` comme des booléens — ne pas les rejeter."""
    args = _load(tmp_path, "run_ica: yes\nrun_degsea: no")
    assert args.run_ica == "y" and args.run_degsea == "n"


# --------------------------------------------- collections : point unique
def test_collections_or_fallback_expands_user_paths_consistently():
    assert cf.collections_or_fallback({"h": "~/x.gmt"}, None)["h"].startswith("/")
    fallback = cf.collections_or_fallback({}, "~/hallmarks.gmt")
    assert list(fallback) == ["hallmarks"]
    assert fallback["hallmarks"].startswith("/")
    assert cf.collections_or_fallback({}, None) == {}


def test_clinical_experiments_requires_the_four_contrast_keys():
    spec = {"exp": {"design": "~x", "contrast": "x", "control": "a"}}
    with pytest.raises(cf.ConfigError, match="test"):
        cf.clinical_experiments(spec)
    spec["exp"]["test"] = "b"
    assert set(cf.clinical_experiments(spec)) == {"exp"}


def test_clinical_experiments_honours_the_enabled_switch():
    spec = {"enabled": "n", "exp": {"design": "~x", "contrast": "x",
                                    "control": "a", "test": "b"}}
    assert cf.clinical_experiments(spec) == {}


def test_clinical_experiment_name_cannot_escape_its_directory():
    spec = {"../evasion": {"design": "~x", "contrast": "x",
                           "control": "a", "test": "b"}}
    with pytest.raises(cf.ConfigError, match="invalide"):
        cf.clinical_experiments(spec)


# ------------------------------------------------------- configs réelles
@pytest.mark.parametrize("name", ["config_SARAH.yaml", "config_PREDIMEL.yaml"])
def test_shipped_configs_are_structurally_valid(name):
    """Les YAML livrés doivent rester lisibles (hors chemins propres à la machine)."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "config" / name
    raw = cf.load_yaml(path)
    known = set(vars(cf.build_parser().parse_args([])))
    consumed: set[str] = set()
    cf._extract_structured(raw, consumed)
    assert cf._check_unknown_keys(raw, known, consumed) == []
