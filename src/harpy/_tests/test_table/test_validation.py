import numpy as np
import pytest

from harpy.table import validate_table
from harpy.table._allocation import aggregate_points


def test_validate_table_accepts_a_registered_generic_feature_matrix(backed_class_aware_sdata):
    result = aggregate_points(
        backed_class_aware_sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )
    adata = result.tables["table"]
    adata.obsm["derived_features"] = np.ones((adata.n_obs, 2), dtype=np.float64)
    adata.uns["feature_matrices"] = {
        "derived_features": {
            "schema_version": 1,
            "source_kind": "test_features",
            "backend": "numpy",
            "dtype": "float64",
            "feature_columns": ["first", "second"],
        }
    }

    validate_table(result, "table")


def test_validate_table_checks_only_the_selected_feature_matrix_registry(backed_class_aware_sdata):
    """A registry under a custom key is validated only when validate_table is given that key."""
    result = aggregate_points(
        backed_class_aware_sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
    )
    adata = result.tables["table"]
    adata.obsm["derived_features"] = np.ones((adata.n_obs, 2), dtype=np.float64)
    # One name for two columns: an invalid record, registered under a custom key.
    adata.uns["custom_feature_matrices"] = {
        "derived_features": {
            "schema_version": 1,
            "source_kind": "test_features",
            "feature_columns": ["first"],
        }
    }

    validate_table(result, "table")
    with pytest.raises(ValueError, match="declares 1 feature columns"):
        validate_table(result, "table", feature_matrices_key="custom_feature_matrices")

    adata.uns["custom_feature_matrices"]["derived_features"]["feature_columns"] = ["first", "second"]
    validate_table(result, "table", feature_matrices_key="custom_feature_matrices")


@pytest.mark.parametrize("feature_matrices_key", ["", None])
def test_validate_table_rejects_an_invalid_feature_matrices_key(class_aware_sdata, feature_matrices_key):
    with pytest.raises(ValueError, match="Parameter 'feature_matrices_key' must be a non-empty string"):
        validate_table(class_aware_sdata, "table", feature_matrices_key=feature_matrices_key)


def test_validate_table_accepts_filtered_and_reordered_feature_axes(backed_class_aware_sdata):
    result = aggregate_points(
        backed_class_aware_sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"][:, ["GeneZero", "GeneB"]].copy()
    # A feature's class is recorded only by the matrix that holds its column,
    # so a class matrix may be subset or reordered as long as its feature_columns
    # are updated to match.
    adata.obsm["negative"] = adata.obsm["negative"][:, [2, 0]]
    adata.uns["feature_matrices"]["negative"]["feature_columns"] = ["NegativeZero", "Negative01"]
    # Nor is the expression index tied to the panel: aggregate_points names it
    # after feature_key ("gene"), but a renamed or unnamed index stays valid.
    adata.var.index.name = None
    result.tables["table"] = adata

    validate_table(result, "table")


def test_validate_table_accepts_preprocessed_matrix_representations(backed_class_aware_sdata):
    result = aggregate_points(
        backed_class_aware_sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    adata.X = adata.X.to_memory().toarray().astype(np.float32)
    adata.obsm["negative"] = adata.obsm["negative"].to_memory().toarray().astype(np.float64)

    validate_table(result, "table")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("fewer_names", "declares 1 feature columns"),
        ("fewer_columns", "has 2 columns"),
        ("registered_matrix_missing", r"missing adata.obsm\['negative'\]"),
    ],
)
def test_validate_table_rejects_non_expression_class_matrices_whose_names_do_not_match(
    mutation: str, message: str, backed_class_aware_sdata
):
    """Non-expression class matrices are validated as every registered feature matrix is."""
    result = aggregate_points(
        backed_class_aware_sdata,
        labels_name="labels_a",
        points_name="points_a",
        to_coordinate_system="sample_a",
        output_table_name="table",
        expression_class="Endogenous",
    )
    adata = result.tables["table"]
    if mutation == "fewer_names":
        adata.uns["feature_matrices"]["negative"]["feature_columns"] = ["Negative01"]
    elif mutation == "fewer_columns":
        adata.obsm["negative"] = adata.obsm["negative"][:, :-1]
    else:
        del adata.obsm["negative"]

    with pytest.raises(ValueError, match=message):
        validate_table(result, "table")
