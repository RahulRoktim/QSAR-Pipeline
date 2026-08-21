"""Target resolution with a mandatory identity guard.

Rationale
---------
``config.TARGET`` is a human label ("BRAF", "PI3K") that is used to name every
downstream artefact; ``config.TARGET_CHEMBL_ID`` is the identifier that decides
which bioactivities are actually downloaded. Nothing previously forced the two
to agree, so editing one without the other produced a dataset labelled with one
protein and populated with another - an error that is invisible in every plot,
metric and report the pipeline emits.

This module makes that state unrepresentable: a ChEMBL id is only accepted once
its ``pref_name`` or a gene symbol on one of its components has been checked
against the requested target label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


class TargetResolutionError(RuntimeError):
    """Raised when a target cannot be resolved from ChEMBL."""


class TargetMismatchError(TargetResolutionError):
    """Raised when a configured ChEMBL id does not correspond to the label."""


@dataclass(frozen=True)
class ResolvedTarget:
    """A ChEMBL target that has been checked against the requested label."""

    label: str
    chembl_id: str
    pref_name: str
    organism: str | None = None
    gene_symbols: tuple[str, ...] = field(default_factory=tuple)
    source: str = "configured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "chembl_id": self.chembl_id,
            "pref_name": self.pref_name,
            "organism": self.organism,
            "gene_symbols": list(self.gene_symbols),
            "source": self.source,
        }


# Curated aliases for labels whose colloquial name never appears verbatim in a
# ChEMBL pref_name. Keep this small and evidence-backed; it is not a substitute
# for checking the resolved record.
TARGET_ALIASES: dict[str, tuple[str, ...]] = {
    "PI3K": (
        "phosphatidylinositol",
        "pi3-kinase",
        "pik3ca",
        "pik3cb",
        "pik3cd",
        "pik3cg",
    ),
    "BRAF": ("b-raf", "braf"),
    "AKT1": ("akt1", "rac-alpha"),
    "EGFR": ("egfr", "epidermal growth factor receptor"),
    "BCL2": ("bcl-2", "bcl2", "apoptosis regulator bcl-2"),
}


# Shorter fragments are too generic to be evidence of identity.
_MIN_TERM_LENGTH = 3


def _normalise(text: str) -> str:
    """Lowercase and strip punctuation so 'B-Raf' matches 'BRAF'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _candidate_terms(label: str) -> tuple[str, ...]:
    terms = [label, *TARGET_ALIASES.get(label.upper(), ())]
    return tuple(_normalise(term) for term in terms if term)


def target_matches(
    label: str,
    pref_name: str,
    gene_symbols: Sequence[str] = (),
) -> bool:
    """Return True when a ChEMBL record plausibly corresponds to ``label``.

    Pure and network-free so the guard itself is unit-testable. Matching is
    deliberately permissive on formatting (case, hyphens) and strict on
    substance: something has to actually contain the label or a curated alias.
    """
    haystacks = [
        text
        for text in (_normalise(pref_name), *(_normalise(g) for g in gene_symbols))
        if text
    ]

    if not haystacks:
        return False

    for term in _candidate_terms(label):
        # Guard against degenerate inputs: an empty or near-empty string is a
        # substring of everything, so a blank pref_name must never match.
        if len(term) < _MIN_TERM_LENGTH:
            continue

        for haystack in haystacks:
            if term in haystack:
                return True

            # Reverse containment covers labels that are more specific than the
            # record ("BRAF kinase" vs pref_name "BRAF"), but only when the
            # record's own name is substantial enough to be meaningful.
            if len(haystack) >= _MIN_TERM_LENGTH and haystack in term:
                return True

    return False


def gene_symbols_of(record: dict[str, Any]) -> tuple[str, ...]:
    """Extract gene symbols from a ChEMBL target record, tolerating absences."""
    symbols: list[str] = []

    for component in record.get("target_components") or record.get("components") or []:
        symbol = component.get("gene_symbol")
        if symbol:
            symbols.append(symbol)
            continue
        for xref in component.get("target_component_xrefs") or []:
            if xref.get("xref_src_db") == "HGNC" and xref.get("xref_name"):
                symbols.append(xref["xref_name"])

    return tuple(dict.fromkeys(symbols))


def verify_configured_target(
    label: str,
    chembl_id: str,
    record: dict[str, Any],
) -> ResolvedTarget:
    """Accept an explicitly configured ChEMBL id only if it matches ``label``.

    Raises
    ------
    TargetMismatchError
        If the record's preferred name and gene symbols bear no relation to the
        requested label. This is the guard that would have caught a config
        holding ``TARGET = "PI3K"`` alongside a BRAF ChEMBL id.
    """
    pref_name = record.get("pref_name") or ""
    genes = gene_symbols_of(record)

    if not target_matches(label, pref_name, genes):
        raise TargetMismatchError(
            f"Configured TARGET_CHEMBL_ID={chembl_id!r} resolves to "
            f"{pref_name!r} (genes: {', '.join(genes) or 'unknown'}), which does "
            f"not correspond to TARGET={label!r}.\n"
            "Refusing to download: this would produce a dataset labelled "
            f"{label!r} containing {pref_name!r} activities.\n"
            "Fix config.TARGET or config.TARGET_CHEMBL_ID so the two agree, or "
            "set TARGET_CHEMBL_ID = None to resolve the id from TARGET."
        )

    return ResolvedTarget(
        label=label,
        chembl_id=chembl_id,
        pref_name=pref_name,
        organism=record.get("organism"),
        gene_symbols=genes,
        source="configured",
    )


def choose_best_target(
    label: str,
    records: Iterable[dict[str, Any]],
    activity_counter,
) -> ResolvedTarget:
    """Pick the single-protein target with the most activities for ``label``.

    ChEMBL routinely holds several entries for one protein and the minor
    duplicates carry almost no data, so record count - not search rank - is the
    correct tie-breaker. Candidates are filtered through the same identity guard
    used for configured ids.
    """
    records = list(records)

    if not records:
        raise TargetResolutionError(f"ChEMBL returned no targets for {label!r}.")

    single = [r for r in records if r.get("target_type") == "SINGLE PROTEIN"]
    candidates = single or records

    matching = [
        r
        for r in candidates
        if target_matches(label, r.get("pref_name") or "", gene_symbols_of(r))
    ]

    if not matching:
        names = ", ".join(sorted({r.get("pref_name") or "?" for r in candidates})[:5])
        raise TargetMismatchError(
            f"No ChEMBL target matching {label!r} among the search results "
            f"({names}). Set TARGET_CHEMBL_ID explicitly if the label is a "
            "colloquial name, or add it to TARGET_ALIASES."
        )

    best = max(matching, key=lambda r: activity_counter(r["target_chembl_id"]))

    return ResolvedTarget(
        label=label,
        chembl_id=best["target_chembl_id"],
        pref_name=best.get("pref_name") or "",
        organism=best.get("organism"),
        gene_symbols=gene_symbols_of(best),
        source="search",
    )
