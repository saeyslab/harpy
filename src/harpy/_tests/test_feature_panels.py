from copy import deepcopy
from dataclasses import replace

import pandas as pd
import pytest

from harpy._feature_panels import (
    _feature_membership_partition_errors,
    _feature_panel_partition_errors,
    _FeaturePanelContract,
    _make_feature_panel,
    _parse_feature_panel,
    _parse_feature_panel_registry,
    _validate_feature_panel_collision,
)


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"feature_key": ""}, "non-empty string"),
        ({"feature_class_key": ""}, "non-empty string"),
        ({"feature_class_key": "gene"}, "different feature and feature-class keys"),
        ({"classes": ()}, "non-empty tuple"),
        ({"classes": ("Endogenous", "Endogenous")}, "unique"),
        ({"classes": ("Endogenous", None)}, "non-empty string"),
        ({"features_by_class_items": (("Endogenous", ("GeneA",)),)}, "exactly the declared classes"),
        (
            {"features_by_class_items": (("Negative", ("Negative01",)), ("Endogenous", ("GeneA",)))},
            "same order",
        ),
        (
            {"features_by_class_items": (("Endogenous", ("GeneA",)), ("Endogenous", ("GeneB",)))},
            "exactly the declared classes",
        ),
        ({"features_by_class_items": (("Endogenous", ()), ("Negative", ("Negative01",)))}, "non-empty tuple"),
        (
            {"features_by_class_items": (("Endogenous", (None,)), ("Negative", ("Negative01",)))},
            "non-empty string",
        ),
        (
            {"features_by_class_items": (("Endogenous", ("GeneA", "GeneA")), ("Negative", ("Negative01",)))},
            "unique",
        ),
        ({"features_by_class_items": (("Endogenous", ("GeneA",)), ("Negative", ("GeneA",)))}, "belongs to both"),
    ],
)
def test_direct_panel_construction_checks_internal_consistency(changes, match):
    """No construction path may bypass the panel's class/feature invariants."""
    fields = {
        "feature_key": "gene",
        "feature_class_key": "kind",
        "classes": ("Endogenous", "Negative"),
        "features_by_class_items": (("Endogenous", ("GeneA",)), ("Negative", ("Negative01",))),
    }
    fields.update(changes)
    with pytest.raises(ValueError, match=match):
        _FeaturePanelContract(**fields)


@pytest.mark.parametrize(
    "changes",
    [
        {"classes": ["Endogenous"]},
        {"features_by_class_items": [("Endogenous", ("GeneA",))]},
        {"features_by_class_items": (["Endogenous", ("GeneA",)],)},
        {"features_by_class_items": (("Endogenous",),)},
        {"features_by_class_items": (("Endogenous", ["GeneA"]),)},
    ],
)
def test_direct_panel_construction_requires_immutable_tuple_structure(changes):
    fields = {
        "feature_key": "gene",
        "feature_class_key": "kind",
        "classes": ("Endogenous",),
        "features_by_class_items": (("Endogenous", ("GeneA",)),),
    }
    fields.update(changes)
    with pytest.raises(ValueError, match="tuple"):
        _FeaturePanelContract(**fields)


def test_make_feature_panel_normalizes_before_serialization():
    """New panels own sorted immutable fields before a storage record is needed."""
    supplied = {"Negative": ("Negative01",), "Endogenous": ["GeneB", "GeneA"]}
    panel = _make_feature_panel(feature_key="gene", feature_class_key="kind", features_by_class=supplied)

    assert panel.classes == ("Endogenous", "Negative")
    assert panel.features_by_class_items == (
        ("Endogenous", ("GeneA", "GeneB")),
        ("Negative", ("Negative01",)),
    )
    assert supplied["Endogenous"] == ["GeneB", "GeneA"]
    supplied["Endogenous"].append("LaterAddition")
    panel_record = panel.to_dict()
    assert panel_record == {
        "feature_key": "gene",
        "feature_class_key": "kind",
        "classes": ["Endogenous", "Negative"],
        "features_by_class": {"Endogenous": ["GeneA", "GeneB"], "Negative": ["Negative01"]},
    }
    restored_panel = _parse_feature_panel(panel_record, panel_name="stored")
    assert restored_panel == panel
    assert restored_panel.storage_key == panel.storage_key
    independent_record = panel.to_dict()
    panel_record["classes"].append("LaterClass")
    panel_record["features_by_class"]["Endogenous"].clear()
    assert panel.to_dict() == independent_record


def test_direct_panel_construction_sorts_classes_and_features_without_changing_assignments():
    supplied = (("Negative", ("Negative02", "Negative01")), ("Endogenous", ("GeneB", "GeneA")))
    panel = _FeaturePanelContract(
        feature_key="gene",
        feature_class_key="kind",
        classes=("Negative", "Endogenous"),
        features_by_class_items=supplied,
    )

    assert panel.classes == ("Endogenous", "Negative")
    assert panel.features_by_class_items == (
        ("Endogenous", ("GeneA", "GeneB")),
        ("Negative", ("Negative01", "Negative02")),
    )
    canonical = _make_feature_panel(
        feature_key="gene",
        feature_class_key="kind",
        features_by_class={"Endogenous": ["GeneA", "GeneB"], "Negative": ["Negative01", "Negative02"]},
    )
    assert panel == canonical
    assert panel.storage_key == canonical.storage_key
    assert supplied == (("Negative", ("Negative02", "Negative01")), ("Endogenous", ("GeneB", "GeneA")))


@pytest.mark.parametrize("unsorted_field", ["classes", "features"])
def test_parse_feature_panel_rejects_unsorted_records_without_mutation(unsorted_field):
    """Construction normalizes new inputs, but parsing must not repair persisted axes."""
    panel = _make_feature_panel(
        feature_key="gene",
        feature_class_key="kind",
        features_by_class={"Endogenous": ["GeneA", "GeneB"], "Negative": ["Negative01"]},
    )
    record = panel.to_dict()
    if unsorted_field == "classes":
        record["classes"].reverse()
    else:
        record["features_by_class"]["Endogenous"].reverse()
    original = deepcopy(record)

    with pytest.raises(ValueError, match=r"harpy.feature_panels.stored.*must be sorted"):
        _parse_feature_panel(record, panel_name="stored")

    assert record == original


def test_parse_panel_registry_preserves_stored_keys_and_records():
    panel = _make_feature_panel(
        feature_key="gene",
        feature_class_key="kind",
        features_by_class={"Negative": ["Negative01"], "Endogenous": ["GeneB", "GeneA"]},
    )
    record = panel.to_dict()
    record["extension"] = {"note": "keep"}
    # Preserve a stored key even when it differs from this record's content hash.
    stored_key = replace(panel, feature_key="other_feature").storage_key
    records = {stored_key: record}
    original = deepcopy(records)

    panels = _parse_feature_panel_registry(records)

    assert panels == {stored_key: panel}
    assert records == original


def test_panel_collision_compares_contracts_under_the_candidate_key():
    candidate = _make_feature_panel(
        feature_key="gene", feature_class_key="kind", features_by_class={"Endogenous": ["GeneA"]}
    )
    other = replace(candidate, feature_key="other_feature")

    _validate_feature_panel_collision(candidate, existing_panels={})
    _validate_feature_panel_collision(candidate, existing_panels={candidate.storage_key: candidate})
    _validate_feature_panel_collision(candidate, existing_panels={other.storage_key: other})

    # A matching key is not sufficient when actual contents differ.
    panels = {candidate.storage_key: other}
    with pytest.raises(ValueError, match="hash collision"):
        _validate_feature_panel_collision(candidate, existing_panels=panels)
    assert panels == {candidate.storage_key: other}


@pytest.mark.parametrize(
    "features_by_class, match",
    [
        ({"Endogenous": ["GeneA", "GeneA"]}, "unique"),
        ({"Endogenous": ["GeneA"], "Negative": ["GeneA"]}, "belongs to both"),
        ({"Endogenous": [""]}, "non-empty"),
        ({"Endogenous": ["GeneA", None]}, "string"),
    ],
)
def test_construction_and_parsing_share_panel_content_checks(features_by_class, match):
    with pytest.raises(ValueError, match=match):
        _make_feature_panel(feature_key="gene", feature_class_key="kind", features_by_class=features_by_class)
    with pytest.raises(ValueError, match=match):
        _parse_feature_panel(
            {
                "feature_key": "gene",
                "feature_class_key": "kind",
                "classes": list(features_by_class),
                "features_by_class": features_by_class,
            },
            panel_name="stored",
        )


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"classes": ["Negative"]}, "exactly the declared classes"),
        ({"classes": ["Endogenous", "Endogenous"]}, "unique"),
        ({"features_by_class": {"Endogenous": ("GeneA",)}}, "must be a list"),
    ],
)
def test_parse_feature_panel_retains_storage_structure_checks(changes, match):
    panel_record = {
        "feature_key": "gene",
        "feature_class_key": "kind",
        "classes": ["Endogenous"],
        "features_by_class": {"Endogenous": ["GeneA"]},
    }
    panel_record.update(changes)
    with pytest.raises(ValueError, match=match):
        _parse_feature_panel(panel_record, panel_name="stored")


@pytest.mark.parametrize(
    "features, error",
    [
        ([], None),
        (["GeneA", "GeneA"], None),
        (["GeneA", None], "must not be null"),
        (["GeneA", "Unknown"], "'Unknown' is absent"),
    ],
)
def test_feature_membership_checks_only_observed_features(features, error):
    """Membership needs no class column and ignores unused feature categories."""
    partition = pd.DataFrame({"target": pd.Categorical(features, categories=["GeneA", "Unknown"])})
    original = partition.copy(deep=True)

    errors = _feature_membership_partition_errors(
        partition, feature_key="target", class_by_feature={"GeneA": "Expression", "Undetected": "Control"}
    )

    if error is None:
        assert errors.empty
    else:
        assert len(errors) == 1
        assert error in errors.iloc[0]
    pd.testing.assert_frame_equal(partition, original)


@pytest.mark.parametrize(
    "features, classes, error",
    [
        ([], [], None),
        (["GeneA", "Control1"], ["Expression", "Control"], None),
        (["GeneA", None], ["Expression", "Control"], "feature values must not be null"),
        (["GeneA", "Unknown"], ["Expression", "Control"], "'Unknown' is absent"),
        (["GeneA"], [None], "feature-class values must not be null"),
        (["GeneA"], ["Control"], "expected 'Expression'"),
        (["GeneA"], ["UnknownClass"], "expected 'Expression'"),
    ],
)
def test_feature_panel_checks_membership_and_observed_classes(features, classes, error):
    """Full validation retains feature checks and also rejects invalid classes."""
    partition = pd.DataFrame({"target": pd.Categorical(features), "kind": pd.Categorical(classes)})
    original = partition.copy(deep=True)

    errors = _feature_panel_partition_errors(
        partition,
        feature_key="target",
        feature_class_key="kind",
        class_by_feature={"GeneA": "Expression", "Control1": "Control"},
    )

    if error is None:
        assert errors.empty
    else:
        assert len(errors) == 1
        assert error in errors.iloc[0]
    pd.testing.assert_frame_equal(partition, original)
