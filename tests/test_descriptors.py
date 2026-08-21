"""Descriptor calculation must be deterministic and must not silently drop rows.

A QSAR model is only reproducible if the same SMILES always yields the same
feature vector, and if molecules RDKit cannot parse are excluded loudly rather
than turning into rows of NaN the model then imputes.
"""

import pandas as pd
import pytest

import config
from src.descriptors import calculate_descriptors


@pytest.fixture
def run_descriptors(tmp_path, monkeypatch):
    def _run(records):
        processed = tmp_path / "processed.csv"
        descriptors = tmp_path / "descriptors.csv"

        pd.DataFrame(records, columns=["SMILES", "Activity"]).to_csv(
            processed, index=False
        )

        import src.descriptors as module

        monkeypatch.setattr(config, "PROCESSED_DATASET", str(processed))
        monkeypatch.setattr(config, "DESCRIPTOR_DATASET", str(descriptors))
        monkeypatch.setattr(module, "PROCESSED_DATASET", str(processed))
        monkeypatch.setattr(module, "DESCRIPTOR_DATASET", str(descriptors))

        calculate_descriptors()
        return pd.read_csv(descriptors)

    return _run


VALID = [("CCO", 5.0), ("c1ccccc1", 6.0), ("CC(=O)Oc1ccccc1C(=O)O", 7.5)]


class TestDeterminism:
    def test_same_input_gives_identical_descriptors(self, run_descriptors):
        first = run_descriptors(VALID)
        second = run_descriptors(VALID)

        pd.testing.assert_frame_equal(first, second)

    def test_molecular_weight_is_chemically_correct(self, run_descriptors):
        out = run_descriptors([("CCO", 5.0)])

        # Ethanol, C2H6O = 46.07 g/mol
        assert out.loc[0, "MolWt"] == pytest.approx(46.07, abs=0.05)

    def test_benzene_ring_count(self, run_descriptors):
        out = run_descriptors([("c1ccccc1", 6.0)])

        assert out.loc[0, "RingCount"] == 1
        assert out.loc[0, "NumAromaticRings"] == 1


class TestInvalidInput:
    def test_unparseable_smiles_is_excluded(self, run_descriptors):
        out = run_descriptors([("CCO", 5.0), ("this-is-not-smiles", 6.0)])

        assert len(out) == 1
        assert out.loc[0, "SMILES"] == "CCO"

    def test_activity_column_survives(self, run_descriptors):
        out = run_descriptors(VALID)

        assert "Activity" in out.columns
        assert out["Activity"].tolist() == [5.0, 6.0, 7.5]


class TestShape:
    def test_every_valid_molecule_produces_a_row(self, run_descriptors):
        out = run_descriptors(VALID)

        assert len(out) == len(VALID)

    def test_descriptor_columns_are_wide_enough_to_be_useful(self, run_descriptors):
        out = run_descriptors(VALID)

        # SMILES + Activity + RDKit's descriptor block (200+ in modern RDKit)
        assert len(out.columns) > 100

    def test_no_all_nan_descriptor_columns(self, run_descriptors):
        out = run_descriptors(VALID)

        numeric = out.drop(columns=["SMILES"])
        all_nan = [c for c in numeric.columns if numeric[c].isna().all()]

        assert not all_nan, f"descriptors that never compute: {all_nan[:10]}"
