"""Tests for the target identity guard.

The regression these lock down is real: config once held TARGET = "PI3K"
alongside TARGET_CHEMBL_ID = "CHEMBL5145", which is human BRAF. Nothing
detected it, so the pipeline would have produced a dataset labelled PI3K
containing BRAF activities.
"""

import pytest

from src.target_resolver import (
    ResolvedTarget,
    TargetMismatchError,
    TargetResolutionError,
    choose_best_target,
    gene_symbols_of,
    target_matches,
    verify_configured_target,
)

BRAF_RECORD = {
    "target_chembl_id": "CHEMBL5145",
    "pref_name": "Serine/threonine-protein kinase B-raf",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "target_components": [
        {"gene_symbol": "BRAF", "accession": "P15056"},
    ],
}

PI3K_RECORD = {
    "target_chembl_id": "CHEMBL3145",
    "pref_name": "Phosphatidylinositol 4,5-bisphosphate 3-kinase catalytic subunit alpha",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "target_components": [{"gene_symbol": "PIK3CA"}],
}

BRAF_DUPLICATE = {
    "target_chembl_id": "CHEMBL2331061",
    "pref_name": "Serine/threonine-protein kinase B-raf",
    "target_type": "SINGLE PROTEIN",
    "organism": "Homo sapiens",
    "target_components": [{"gene_symbol": "BRAF"}],
}


class TestTargetMatches:
    def test_matches_ignoring_case_and_punctuation(self):
        assert target_matches("BRAF", "Serine/threonine-protein kinase B-raf")

    def test_matches_via_gene_symbol(self):
        assert target_matches("BRAF", "Some opaque name", ["BRAF"])

    def test_matches_pi3k_through_alias(self):
        assert target_matches("PI3K", PI3K_RECORD["pref_name"])

    def test_rejects_unrelated_protein(self):
        assert not target_matches("PI3K", "Serine/threonine-protein kinase B-raf", ["BRAF"])

    def test_rejects_empty_pref_name(self):
        assert not target_matches("BRAF", "")


class TestVerifyConfiguredTarget:
    def test_accepts_matching_record(self):
        resolved = verify_configured_target("BRAF", "CHEMBL5145", BRAF_RECORD)

        assert isinstance(resolved, ResolvedTarget)
        assert resolved.chembl_id == "CHEMBL5145"
        assert resolved.gene_symbols == ("BRAF",)
        assert resolved.source == "configured"

    def test_rejects_the_pi3k_braf_mismatch(self):
        with pytest.raises(TargetMismatchError) as excinfo:
            verify_configured_target("PI3K", "CHEMBL5145", BRAF_RECORD)

        message = str(excinfo.value)
        assert "CHEMBL5145" in message
        assert "B-raf" in message
        assert "PI3K" in message

    def test_error_names_both_sides_so_the_fix_is_obvious(self):
        with pytest.raises(TargetMismatchError) as excinfo:
            verify_configured_target("EGFR", "CHEMBL5145", BRAF_RECORD)

        assert "TARGET_CHEMBL_ID" in str(excinfo.value)
        assert "TARGET" in str(excinfo.value)


class TestChooseBestTarget:
    def test_picks_the_entry_with_most_activities(self):
        counts = {"CHEMBL5145": 4321, "CHEMBL2331061": 77}

        resolved = choose_best_target(
            "BRAF", [BRAF_DUPLICATE, BRAF_RECORD], counts.__getitem__
        )

        assert resolved.chembl_id == "CHEMBL5145"
        assert resolved.source == "search"

    def test_prefers_single_protein_entries(self):
        family = {
            "target_chembl_id": "CHEMBL_FAMILY",
            "pref_name": "BRAF family",
            "target_type": "PROTEIN FAMILY",
            "target_components": [],
        }
        counts = {"CHEMBL_FAMILY": 99999, "CHEMBL5145": 10}

        resolved = choose_best_target(
            "BRAF", [family, BRAF_RECORD], counts.__getitem__
        )

        assert resolved.chembl_id == "CHEMBL5145"

    def test_rejects_results_that_do_not_match_the_label(self):
        with pytest.raises(TargetMismatchError):
            choose_best_target("PI3K", [BRAF_RECORD], lambda _: 1)

    def test_empty_result_set_is_an_error(self):
        with pytest.raises(TargetResolutionError):
            choose_best_target("BRAF", [], lambda _: 1)


class TestGeneSymbols:
    def test_reads_gene_symbol_field(self):
        assert gene_symbols_of(BRAF_RECORD) == ("BRAF",)

    def test_falls_back_to_hgnc_xref(self):
        record = {
            "target_components": [
                {"target_component_xrefs": [
                    {"xref_src_db": "HGNC", "xref_name": "BRAF"},
                ]}
            ]
        }
        assert gene_symbols_of(record) == ("BRAF",)

    def test_tolerates_missing_components(self):
        assert gene_symbols_of({}) == ()
