"""Preprocessing decides what the model actually learns from.

Replicate measurements are aggregated by median rather than dropped, so these
tests pin the aggregation, the de-duplication key and the handling of records
that cannot be used.
"""

import pandas as pd
import pytest

import config
from src.preprocess import preprocess

COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "pchembl_value",
    "standard_units",
    "standard_type",
]


@pytest.fixture
def run_preprocess(tmp_path, monkeypatch):
    def _run(rows):
        raw = tmp_path / "raw.csv"
        processed = tmp_path / "processed.csv"

        pd.DataFrame(rows, columns=COLUMNS).to_csv(raw, index=False)

        monkeypatch.setattr(config, "RAW_DATASET", str(raw))
        monkeypatch.setattr(config, "PROCESSED_DATASET", str(processed))

        import src.preprocess as module

        monkeypatch.setattr(module, "RAW_DATASET", str(raw))
        monkeypatch.setattr(module, "PROCESSED_DATASET", str(processed))

        preprocess()
        return pd.read_csv(processed)

    return _run


def row(chembl_id, smiles, pchembl):
    return [chembl_id, smiles, pchembl, "nM", "IC50"]


class TestAggregation:
    def test_replicates_collapse_to_their_median(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL1", "CCO", 7.0),
            row("CHEMBL1", "CCO", 6.0),
        ])

        assert len(out) == 1
        assert out.loc[0, "Activity"] == pytest.approx(6.0)

    def test_median_resists_a_single_outlier(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL1", "CCO", 5.2),
            row("CHEMBL1", "CCO", 99.0),
        ])

        assert out.loc[0, "Activity"] == pytest.approx(5.2)

    def test_distinct_molecules_are_kept_separate(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL2", "CCC", 8.0),
        ])

        assert len(out) == 2
        assert set(out["SMILES"]) == {"CCO", "CCC"}


class TestUnusableRecords:
    def test_rows_without_activity_are_dropped(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL2", "CCC", None),
        ])

        assert len(out) == 1
        assert out.loc[0, "SMILES"] == "CCO"

    def test_rows_without_smiles_are_dropped(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL2", None, 6.0),
        ])

        assert len(out) == 1

    def test_non_numeric_activity_is_dropped_not_coerced_to_zero(self, run_preprocess):
        out = run_preprocess([
            row("CHEMBL1", "CCO", "not-a-number"),
            row("CHEMBL2", "CCC", 8.0),
        ])

        assert len(out) == 1
        assert out.loc[0, "Activity"] == pytest.approx(8.0)


class TestSchema:
    def test_output_columns_are_stable(self, run_preprocess):
        out = run_preprocess([row("CHEMBL1", "CCO", 5.0)])

        assert list(out.columns) == [
            "molecule_chembl_id",
            "SMILES",
            "Activity",
            "standard_units",
            "standard_type",
        ]

    def test_is_deterministic_across_runs(self, run_preprocess):
        rows = [
            row("CHEMBL1", "CCO", 5.0),
            row("CHEMBL2", "CCC", 8.0),
            row("CHEMBL1", "CCO", 7.0),
        ]

        first = run_preprocess(rows)
        second = run_preprocess(rows)

        pd.testing.assert_frame_equal(first, second)
