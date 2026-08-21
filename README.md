# AutoQSAR Pipeline

End-to-end quantitative structure–activity relationship modelling: verified
ChEMBL target resolution → bioactivity retrieval → RDKit descriptors →
cross-validated feature selection → multi-model comparison → y-randomisation
validation.

[![tests](https://github.com/RahulRoktim/QSAR-Pipeline/actions/workflows/tests.yml/badge.svg)](https://github.com/RahulRoktim/QSAR-Pipeline/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## The problem this solves carefully

ChEMBL holds several target entries for most proteins, and the minor duplicates
carry almost no data — the wrong BRAF entry has roughly 77 activities where the
canonical one has thousands. Worse, a target *label* and a target *identifier*
are separate settings, and nothing normally forces them to agree.

This pipeline treats that as a correctness problem, not a convenience one. The
configured ChEMBL id is fetched and checked against the requested target label
before a single activity is downloaded. A mismatch raises `TargetMismatchError`
and stops the run:

```
TargetMismatchError: Configured TARGET_CHEMBL_ID='CHEMBL5145' resolves to
'Serine/threonine-protein kinase B-raf' (genes: BRAF), which does not
correspond to TARGET='PI3K'.
Refusing to download: this would produce a dataset labelled 'PI3K' containing
'Serine/threonine-protein kinase B-raf' activities.
```

Without that guard the run completes normally and every plot, metric and report
is mislabelled — an error invisible at every downstream stage.

---

## Method

| Stage | What happens |
|---|---|
| **Target resolution** | Verified id, or search + pick the single-protein entry with the most activities for the endpoint |
| **Download** | All activities carrying a pChEMBL value, paged with retry — not a capped first 1000 |
| **Preprocessing** | Replicates aggregated by **median** pChEMBL per unique SMILES; unusable records dropped, never coerced |
| **Descriptors** | Full RDKit descriptor block; unparseable SMILES excluded rather than imputed |
| **Feature selection** | Cross-validated selection inside the training fold |
| **Modelling** | Random Forest, XGBoost, LightGBM, CatBoost compared under one seed |
| **Validation** | Hold-out test set plus **y-randomisation** (default 10 scrambles) |
| **Provenance** | `outputs/run_manifest.json` records settings, resolved target, dataset SHA-256s, library versions and git commit |

Median aggregation matters: a molecule with ten IC50 readings should contribute
one consolidated, noise-reduced activity rather than whichever row happened to
be first.

---

## Installation

```bash
conda create -n qsar python=3.11 && conda activate qsar
pip install -r requirements.txt
pytest -q
```

To reproduce a published result, install the verified set instead:

```bash
pip install -r requirements-lock.txt
```

---

## Usage

Every setting that affects a result is a flag, so a run is reproducible from its
manifest alone.

```bash
# Resolve and validate the target, print the plan, download nothing
python main.py --target BRAF --dry-run

# Full run with defaults
python main.py --target BRAF --activity-type IC50

# Explicit id (verified against the label before use)
python main.py --target PI3K --target-chembl-id CHEMBL3145

# Re-model an existing download without re-querying ChEMBL
python main.py --skip-download --cv-folds 10 --random-state 7
```

`python main.py --help` lists every flag with its default.

### Defaults

Defaults live in `config.py`. `KNOWN_TARGET_IDS` holds hand-verified ChEMBL ids
per target, so the curated knowledge survives a change of `TARGET` instead of
being stranded in one global that then applies to the wrong protein.

---

## Validation status

| Layer | Status |
|---|---|
| Target identity guard | 15 tests, including the PI3K/BRAF mismatch regression |
| CLI override and validation | 14 tests |
| Preprocessing | 8 tests — median aggregation, de-duplication, unusable records |
| Descriptors | 8 tests — determinism, chemical correctness, invalid SMILES |
| **Total** | **45 passing** |
| End-to-end ChEMBL run | Requires network; not covered by CI |

---

## Limitations

- **Correlation, not causation.** A QSAR model describes the chemical space it was trained on. Predictions for scaffolds outside that space are extrapolation.
- **Applicability domain is not yet enforced at prediction time.** Treat predictions for dissimilar compounds as unreliable until this lands.
- **pChEMBL mixes assays.** Median aggregation reduces replicate noise but cannot correct for systematically different assay protocols contributing to the same target.
- **Descriptor-based only.** No 3D conformer or fingerprint-based representation yet.
- **Single train/test split** by default. Report cross-validated metrics for anything published, and always report the y-randomisation result alongside them.

---

## Citation

If this software contributes to published work, please cite it via
[`CITATION.cff`](CITATION.cff), and cite ChEMBL and RDKit independently.

---

## Licence

MIT — see [LICENSE](LICENSE).
