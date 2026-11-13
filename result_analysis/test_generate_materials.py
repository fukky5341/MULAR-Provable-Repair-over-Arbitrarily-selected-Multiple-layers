from pathlib import Path

import pandas as pd

from result_analysis.generate_materials import (
    PAPER_MATERIAL_PATHS,
    _accuracy_category_spacing,
    _accuracy_legend_kwargs,
    _accuracy_legend_fontsize,
    _accuracy_drop_y_label,
    _accuracy_plot_fontsize,
    _accuracy_x_label,
    _display_series_name,
    _layer_count_name,
    _matched_accuracy_drop_pair,
    _matched_runtime_pair,
    _pairwise_accuracy_fontsize,
    _runtime_legend_kwargs,
    _runtime_legend_fontsize,
    _runtime_axis_label_fontsize,
    _runtime_y_label,
    _runtime_y_label_fontsize,
    _runtime_tick_fontsize,
    _runtime_series_name,
    _series_color_index,
    _write_accuracy_boxplots,
    _write_network_success_tables,
    _write_network_summary,
    _write_pairwise_accuracy_drop_figures,
    _write_pairwise_accuracy_win_tables,
    _write_pairwise_success_tables,
    _write_pairwise_runtime_figures,
    _write_rq2_prepared_outcome_tables,
    _write_runtime_figures,
    pairwise_accuracy_wins,
    pairwise_success_asymmetry,
    prepared_outcome_matrix,
    success_count_matrix,
)


def test_paper_material_contract_has_only_documented_rq_outputs() -> None:
    assert len(PAPER_MATERIAL_PATHS) == 37
    assert {
        path.split("/", 1)[0] for path in PAPER_MATERIAL_PATHS
    } == {"rq1", "rq2", "rq3", "rq4", "rq5"}
    assert sum(path.startswith("rq1/") for path in PAPER_MATERIAL_PATHS) == 9
    assert sum(path.startswith("rq2/") for path in PAPER_MATERIAL_PATHS) == 8
    assert sum(path.startswith("rq3/") for path in PAPER_MATERIAL_PATHS) == 6
    assert sum(path.startswith("rq4/") for path in PAPER_MATERIAL_PATHS) == 8
    assert sum(path.startswith("rq5/") for path in PAPER_MATERIAL_PATHS) == 6


def test_layer_count_names_use_singular_and_plural() -> None:
    assert _layer_count_name("layers_16") == "1 layer"
    assert _layer_count_name("layers_12_14_16") == "3 layers"


def test_rq5_variant_display_names_use_mular_subscripts() -> None:
    assert _display_series_name("pareto10") == r"MULAR$_{\mathrm{Arachne}}$"
    assert _display_series_name("pareto10_random") == r"MULAR$_{\mathrm{Random}}$"


def test_approach_colors_use_global_order() -> None:
    assert _series_color_index("prepared", 0) == 2
    assert _series_color_index("prorepair", 0) == 3
    assert _series_color_index("lastlayer", 0) == 1
    assert _series_color_index("ours", 0) == 4
    assert _series_color_index("layers_16", 7) == 7


def test_accuracy_x_labels_distinguish_dimensions_and_boxes() -> None:
    assert _accuracy_x_label("ndims") == "#Dim"
    assert _accuracy_x_label("buggy_points_num") == "#P"


def test_accuracy_legend_locations_are_selected_by_rq_and_network() -> None:
    assert _accuracy_legend_kwargs("rq1", "mnist_256x4") == {"loc": "lower right"}
    assert _accuracy_legend_kwargs("rq2", "mnist 9x100") == {"loc": "lower left"}
    assert _accuracy_legend_kwargs("rq2", "mnist_256x4") == {"loc": "lower left"}
    assert _accuracy_legend_kwargs("rq2", "cifar10_cnn1") == {"loc": "lower right"}
    assert _accuracy_legend_kwargs("rq5", "cifar10_cnn1") == {"loc": "upper right"}
    assert _accuracy_legend_kwargs("rq5", "mnist 9x100") == {"loc": "lower left"}
    assert _accuracy_legend_kwargs("rq5", "mnist_256x4") == {"loc": "lower left"}
    assert _accuracy_legend_kwargs("rq3", "mnist 9x100") == {
        "loc": "upper left",
        "bbox_to_anchor": (1.01, 1.0),
    }


def test_rq5_accuracy_legends_use_larger_font() -> None:
    assert _accuracy_legend_fontsize("rq5") == 18
    assert _accuracy_legend_fontsize("rq2") == 16


def test_rq5_accuracy_text_uses_larger_font() -> None:
    assert _accuracy_plot_fontsize("rq5") == 18
    assert _accuracy_plot_fontsize("rq2") == 16


def test_rq2_pairwise_accuracy_text_is_four_points_larger() -> None:
    assert _pairwise_accuracy_fontsize("rq2") == 18
    assert _pairwise_accuracy_fontsize("rq5") == 18
    assert _pairwise_accuracy_fontsize("rq3") == 14


def test_rq1_rq2_and_rq5_accuracy_labels_use_short_form() -> None:
    assert _accuracy_drop_y_label("rq1") == "Acc. (original - repaired)"
    assert _accuracy_drop_y_label("rq2") == "Acc. (original - repaired)"
    assert _accuracy_drop_y_label("rq5") == "Acc. (original - repaired)"
    assert _accuracy_drop_y_label("rq3") == (
        "Accuracy degradation (original - repaired)"
    )


def test_accuracy_box_groups_are_closer_for_polytopes() -> None:
    assert _accuracy_category_spacing("buggy_points_num") == 0.70
    assert _accuracy_category_spacing("ndims") == 0.82


def test_runtime_legend_locations_are_selected_by_rq() -> None:
    assert _runtime_legend_kwargs("rq1") == {"loc": "upper right"}
    assert _runtime_legend_kwargs("rq2") == {"loc": "upper left"}
    assert _runtime_legend_kwargs("rq2", "cifar10_cnn1") == {
        "loc": "upper right"
    }
    assert _runtime_legend_kwargs("rq3") == {"loc": "upper left"}
    assert _runtime_legend_kwargs("rq4") == {"loc": "upper left"}
    assert _runtime_legend_kwargs("rq5") == {"loc": "upper left"}
    assert _runtime_legend_kwargs("other") == {
        "loc": "upper left",
        "bbox_to_anchor": (1.01, 1.0),
    }


def test_runtime_legends_use_configured_fonts() -> None:
    assert _runtime_legend_fontsize("rq1") == 18
    assert _runtime_legend_fontsize("rq5") == 20
    assert _runtime_legend_fontsize("rq2") == 22


def test_runtime_axis_labels_use_configured_fonts() -> None:
    assert _runtime_axis_label_fontsize("rq2") == 22
    assert _runtime_axis_label_fontsize("rq5") == 22
    assert _runtime_axis_label_fontsize("rq1") == 18


def test_rq1_rq2_and_rq5_runtime_y_labels_are_shorter() -> None:
    assert _runtime_y_label("rq1") == "Mean Time"
    assert _runtime_y_label_fontsize("rq1") == 18
    assert _runtime_y_label("rq2") == "Mean Time"
    assert _runtime_y_label_fontsize("rq2") == 22
    assert _runtime_y_label("rq5") == "Mean Time"
    assert _runtime_y_label_fontsize("rq5") == 22


def test_runtime_ticks_use_configured_fonts() -> None:
    assert _runtime_tick_fontsize("rq1") == 18
    assert _runtime_tick_fontsize("rq2") == 22
    assert _runtime_tick_fontsize("rq5") == 22


def test_rq3_runtime_series_names_use_layer_counts() -> None:
    assert _runtime_series_name("rq3", "layers_16") == "1 layer"
    assert _runtime_series_name("rq3", "layers_12_14_16") == "3 layers"
    assert _runtime_series_name("rq4", "layers_12_14_16") == "[12, 14, 16]"


def test_network_summary_reports_experiment_architectures(tmp_path) -> None:
    repository = Path(__file__).resolve().parents[1]
    _write_network_summary(repository, tmp_path)
    summary = pd.read_csv(tmp_path / "network_summary.csv").set_index("network")
    assert summary.loc["MNIST 9x100", ["layers", "neurons", "parameters"]].tolist() == [
        9,
        810,
        150210,
    ]
    assert summary.loc["MNIST 256x4", ["layers", "neurons", "parameters"]].tolist() == [
        5,
        1034,
        400906,
    ]
    assert summary.loc["CIFAR-10 CNN1", ["layers", "neurons", "parameters"]].tolist() == [
        4,
        6254,
        210998,
    ]
    assert summary.loc["GTSRB CNN", ["layers", "neurons", "parameters"]].tolist() == [
        4,
        6287,
        214331,
    ]


def test_success_count_matrix_has_approaches_as_rows_and_settings_as_columns() -> None:
    summary = pd.DataFrame(
        [
            {"approach": "ours", "ndims": 7, "successes": 18},
            {"approach": "ours", "ndims": 5, "successes": 20},
            {"approach": "prepared", "ndims": 5, "successes": 17},
            {"approach": "prepared", "ndims": 7, "successes": 15},
        ]
    )
    matrix = success_count_matrix(summary, "ndims")
    assert list(matrix.columns) == ["Approach", 5, 7]
    assert list(matrix["Approach"]) == [
        "APRNN",
        "ProRepair",
        "PREPARED",
        "LLR",
        "MULAR",
    ]
    ours = matrix[matrix["Approach"] == "MULAR"].iloc[0]
    assert ours[5] == "20"
    assert ours[7] == "18"
    assert matrix[matrix["Approach"] == "APRNN"].iloc[0][5] == "0"


def test_success_count_matrix_supports_layer_set_rows() -> None:
    summary = pd.DataFrame(
        [
            {"layer_set_label": "layers_7", "buggy_points_num": 1, "successes": 20},
            {"layer_set_label": "layers_5_7", "buggy_points_num": 1, "successes": 18},
            {"layer_set_label": "layers_7", "buggy_points_num": 2, "successes": 15},
            {"layer_set_label": "layers_5_7", "buggy_points_num": 2, "successes": 19},
        ]
    )
    matrix = success_count_matrix(
        summary,
        "buggy_points_num",
        row_column="layer_set_label",
        row_order=["layers_7", "layers_5_7"],
        row_names={"layers_7": "[7]", "layers_5_7": "[5, 7]"},
        left_header="Repairable layers",
    )
    assert list(matrix.columns) == ["Repairable layers", 1, 2]
    assert list(matrix["Repairable layers"]) == ["[7]", "[5, 7]"]
    assert matrix.iloc[1][2] == "19"


def test_rq2_success_table_uses_num_box_and_plain_n_headers(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "model_name": "cifar10_cnn1",
                "approach": "ours",
                "buggy_points_num": n,
                "successes": 10,
            }
            for n in [12, 14, 16, 18, 20]
        ]
    )
    _write_network_success_tables(
        summary,
        tmp_path,
        rq="rq2",
        x_column="buggy_points_num",
        x_symbol=None,
        left_header=r"\numBox",
    )
    tex = (tmp_path / "rq2_success_counts_cifar10_cnn1.tex").read_text()
    assert r"\numBox & 12 & 14 & 16 & 18 & 20 \\" in tex


def test_rq2_prepared_outcome_tables_break_down_failure_modes(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "cifar10_cnn1",
                "approach": "prepared",
                "buggy_points_num": num_polys,
                "result": result,
                "failure_reason": reason,
            }
            for num_polys, result, reason in [
                (6, "success", pd.NA),
                (6, "failed", "memory_limit"),
                (8, "failed", "gurobi_time_limit"),
                (8, "incomplete", "log_ended_before_result"),
            ]
        ]
    )
    matrix = prepared_outcome_matrix(rows).set_index("Outcome")
    assert matrix.loc["Successful", 6] == 1
    assert matrix.loc["Memory limit", 6] == 1
    assert matrix.loc["Gurobi time limit", 8] == 1
    assert matrix.loc["Log ended before result", 8] == 1
    assert matrix.loc["Total unsuccessful", 6] == 1
    assert matrix.loc["Total unsuccessful", 8] == 2

    paths = _write_rq2_prepared_outcome_tables(rows, tmp_path)
    assert {path.name for path in paths} == {
        "rq2_prepared_outcomes_cifar10_cnn1.csv",
        "rq2_prepared_outcomes_cifar10_cnn1.tex",
    }
    tex = (tmp_path / "rq2_prepared_outcomes_cifar10_cnn1.tex").read_text()
    assert r"\numBox & 6 & 8" in tex
    assert r"\prepared" in tex


def test_rq1_success_table_uses_num_dims_and_plain_dimension_headers(tmp_path) -> None:
    dimensions = [5, 6, 7, 8, 9, 10, 12, 14]
    summary = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "approach": "ours",
                "ndims": dimension,
                "successes": 20,
            }
            for dimension in dimensions
        ]
    )
    _write_network_success_tables(
        summary,
        tmp_path,
        rq="rq1",
        x_column="ndims",
        x_symbol=None,
        left_header=r"\numDims",
    )
    tex = (tmp_path / "rq1_success_counts_mnist_9x100.tex").read_text()
    assert r"\numDims & 5 & 6 & 7 & 8 & 9 & 10 & 12 & 14 \\" in tex
    assert r"\aprnn &" in tex
    assert r"\prorepair &" in tex
    assert r"\prepared &" in tex
    assert r"\lastlayerTool &" in tex
    assert r"\tool &" in tex


def test_accuracy_boxplots_are_written_separately_per_network(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {"model_name": model, "result": "success", "accuracy_drop": drop, "ndims": dim, "evaluation_series": approach}
            for model in ["mnist 9x100", "mnist_256x4"]
            for approach in ["prepared", "ours"]
            for dim in [5, 6]
            for drop in [0.01, 0.02]
        ]
    )
    paths = _write_accuracy_boxplots(rows, tmp_path, x="ndims", prefix="rq1")
    assert len(paths) == 2
    assert {path.name for path in paths} == {
        "rq1_accuracy_drop_mnist_9x100.png",
        "rq1_accuracy_drop_mnist_256x4.png",
    }


def test_runtime_figures_are_written_separately_per_network(tmp_path) -> None:
    summary = pd.DataFrame(
        [
            {
                "model_name": model,
                "evaluation_series": approach,
                "ndims": dim,
                "mean_runtime_sec_success": runtime,
            }
            for model in ["mnist 9x100", "mnist_256x4"]
            for approach in ["prepared", "ours"]
            for dim, runtime in [(5, 1.0), (6, 2.0)]
        ]
    )
    paths = _write_runtime_figures(summary, tmp_path, x="ndims", prefix="rq1")
    assert {path.name for path in paths} == {
        "rq1_runtime_mnist_9x100.png",
        "rq1_runtime_mnist_256x4.png",
    }


def test_pairwise_accuracy_drop_uses_only_matched_successes() -> None:
    rows = pd.DataFrame(
        [
            {"evaluation_series": label, "inp_eps": 1, "buggy_points_num": 2, "run_index": run, "repair_data_ids": ids, "result": result, "accuracy_drop": drop}
            for label, run, ids, result, drop in [
                ("prepared", 1, "1,2", "success", 0.20),
                ("ours", 1, "1,2", "success", 0.05),
                ("prepared", 2, "3,4", "success", 0.30),
                ("ours", 2, "3,4", "failed", None),
            ]
        ]
    )
    matched = _matched_accuracy_drop_pair(rows, "prepared", "ours")
    assert len(matched) == 1
    assert matched.iloc[0]["left_drop"] == 0.20
    assert matched.iloc[0]["right_drop"] == 0.05


def test_pairwise_success_counts_row_success_and_column_non_success(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": approach,
                "inp_eps": 1,
                "buggy_points_num": 2,
                "run_index": run,
                "repair_data_ids": ids,
                "result": result,
            }
            for approach, run, ids, result in [
                ("prepared", 1, "1,2", "success"),
                ("prepared", 2, "3,4", "success"),
                ("lastlayer", 1, "1,2", "failed"),
                ("lastlayer", 2, "3,4", "success"),
                ("ours", 1, "1,2", "incomplete"),
                ("ours", 2, "3,4", "failed"),
            ]
        ]
    )
    matrix = pairwise_success_asymmetry(rows).set_index("Approach")
    assert matrix.loc["PREPARED", "LLR"] == 1
    assert matrix.loc["LLR", "PREPARED"] == 0
    assert matrix.loc["PREPARED", "MULAR"] == 2
    assert matrix.loc["LLR", "MULAR"] == 1
    assert matrix.loc["MULAR", "MULAR"] == 0

    paths = _write_pairwise_success_tables(rows, tmp_path)
    assert {path.name for path in paths} == {
        "rq2_pairwise_success_mnist_9x100.csv",
        "rq2_pairwise_success_mnist_9x100.tex",
    }
    tex = (tmp_path / "rq2_pairwise_success_mnist_9x100.tex").read_text()
    assert r"Approach & \prepared & \lastlayerTool & \tool" in tex
    assert r"\prepared & -- &" in tex
    assert r"\prepared" in tex
    assert r"\lastlayerTool" in tex
    assert r"\tool" in tex


def test_rq3_pairwise_success_tables_use_layer_count_labels(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": layer_set,
                "inp_eps": 1,
                "buggy_points_num": 2,
                "run_index": run,
                "repair_data_ids": ids,
                "result": result,
            }
            for layer_set, run, ids, result in [
                ("layers_16", 1, "1,2", "failed"),
                ("layers_16", 2, "3,4", "success"),
                ("layers_14_16", 1, "1,2", "success"),
                ("layers_14_16", 2, "3,4", "success"),
            ]
        ]
    )
    label_names = {"layers_16": "1 layer", "layers_14_16": "2 layers"}
    matrix = pairwise_success_asymmetry(
        rows,
        label_names=label_names,
        left_header="Repairable layers",
    ).set_index("Repairable layers")
    assert matrix.loc["2 layers", "1 layer"] == 1
    assert matrix.loc["1 layer", "2 layers"] == 0

    paths = _write_pairwise_success_tables(
        rows,
        tmp_path,
        rq="rq3",
        label_names=label_names,
        left_header="Repairable layers",
    )
    assert {path.name for path in paths} == {
        "rq3_pairwise_success_mnist_9x100.csv",
        "rq3_pairwise_success_mnist_9x100.tex",
    }
    tex = (tmp_path / "rq3_pairwise_success_mnist_9x100.tex").read_text()
    assert "Repairable layers & 1 layer & 2 layers" in tex
    assert "1 layer & -- & 0" in tex


def test_rq3_pairwise_accuracy_win_tables_use_matched_successes(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": layer_set,
                "inp_eps": 1,
                "buggy_points_num": 2,
                "run_index": run,
                "repair_data_ids": ids,
                "result": result,
                "repaired_accuracy": accuracy,
            }
            for layer_set, run, ids, result, accuracy in [
                ("layers_16", 1, "1,2", "success", 0.90),
                ("layers_16", 2, "3,4", "success", 0.80),
                ("layers_16", 3, "5,6", "success", 0.70),
                ("layers_16", 4, "7,8", "failed", None),
                ("layers_14_16", 1, "1,2", "success", 0.80),
                ("layers_14_16", 2, "3,4", "success", 0.90),
                ("layers_14_16", 3, "5,6", "success", 0.70),
                ("layers_14_16", 4, "7,8", "success", 0.95),
            ]
        ]
    )
    label_names = {"layers_16": "1 layer", "layers_14_16": "2 layers"}
    matrix = pairwise_accuracy_wins(
        rows,
        label_names=label_names,
        left_header="Repairable layers",
    ).set_index("Repairable layers")
    assert matrix.loc["1 layer", "2 layers"] == 1
    assert matrix.loc["2 layers", "1 layer"] == 1
    assert matrix.loc["1 layer", "1 layer"] == 0

    paths = _write_pairwise_accuracy_win_tables(
        rows,
        tmp_path,
        rq="rq3",
        label_names=label_names,
        left_header="Repairable layers",
    )
    assert {path.name for path in paths} == {
        "rq3_pairwise_accuracy_wins_mnist_9x100.csv",
        "rq3_pairwise_accuracy_wins_mnist_9x100.tex",
    }
    tex = (tmp_path / "rq3_pairwise_accuracy_wins_mnist_9x100.tex").read_text()
    assert "Repairable layers & 1 layer & 2 layers" in tex
    assert "1 layer & -- & 1" in tex


def test_pairwise_accuracy_figures_are_separate_for_each_pair(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": series,
                "inp_eps": 1,
                "buggy_points_num": 1,
                "run_index": 1,
                "repair_data_ids": "1",
                "result": "success",
                "accuracy_drop": drop,
            }
            for series, drop in [
                ("prepared", 0.2),
                ("lastlayer", 0.1),
                ("ours", 0.05),
            ]
        ]
    )
    paths = _write_pairwise_accuracy_drop_figures(rows, tmp_path)
    assert {path.name for path in paths} == {
        "rq2_pairwise_accuracy_drop_mnist_9x100_prepared_vs_lastlayer.png",
        "rq2_pairwise_accuracy_drop_mnist_9x100_prepared_vs_ours.png",
        "rq2_pairwise_accuracy_drop_mnist_9x100_lastlayer_vs_ours.png",
    }
    assert not (tmp_path / "rq2_pairwise_accuracy_drop_mnist_9x100.png").exists()


def test_rq5_pairwise_accuracy_figure_compares_localization_variants(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": variant,
                "inp_eps": 1,
                "buggy_points_num": 1,
                "run_index": 1,
                "repair_data_ids": "1",
                "result": "success",
                "accuracy_drop": drop,
            }
            for variant, drop in [
                ("pareto10", 0.01),
                ("pareto10_random", 0.03),
            ]
        ]
    )
    paths = _write_pairwise_accuracy_drop_figures(
        rows,
        tmp_path,
        prefix="rq5",
    )
    assert {path.name for path in paths} == {
        "rq5_pairwise_accuracy_drop_mnist_9x100_"
        "pareto10_vs_pareto10_random.png"
    }


def test_rq5_pairwise_runtime_figure_uses_matched_successes(tmp_path) -> None:
    rows = pd.DataFrame(
        [
            {
                "model_name": "mnist 9x100",
                "evaluation_series": variant,
                "inp_eps": 1,
                "buggy_points_num": num_polys,
                "run_index": run,
                "repair_data_ids": ids,
                "result": result,
                "total_repair_time_sec": runtime,
            }
            for variant, num_polys, run, ids, result, runtime in [
                ("pareto10", 1, 1, "1", "success", 10.0),
                ("pareto10_random", 1, 1, "1", "success", 20.0),
                ("pareto10", 2, 2, "2", "success", 30.0),
                ("pareto10_random", 2, 2, "2", "failed", None),
            ]
        ]
    )
    matched = _matched_runtime_pair(rows, "pareto10", "pareto10_random")
    assert len(matched) == 1
    assert matched.iloc[0]["left_runtime"] == 10.0
    assert matched.iloc[0]["right_runtime"] == 20.0

    paths = _write_pairwise_runtime_figures(
        rows,
        tmp_path,
        prefix="rq5",
    )
    assert {path.name for path in paths} == {
        "rq5_pairwise_runtime_mnist_9x100_"
        "pareto10_vs_pareto10_random.png"
    }
