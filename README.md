# Dynamic Neural-Symbolic Reasoning — lightweight GitHub edition

Original v18 research code, packaged without datasets, trained weights, raw predictions, historical result archives or the article PDF. The original full research archive remains separate. The method, protocols, seeds and completed numerical results are unchanged.

## Upload

Extract the outer ZIP once and upload this directory's contents. Keep supporting_code.zip intact during web upload; it contains only additional code, tests and small configurations. There are fewer than 100 upload files. No evidence.part files or embedded full-release archive remain.

## Dataset download links

- **Curated FOLIO validation (used in the main formal-input study):** https://huggingface.co/datasets/DSAVlab-UNIUD/FOLIO_validation-curated/tree/952c6bbfd3ae2451030b7986e23284c631fc25b5
- **Original FOLIO:** https://github.com/Yale-LILY/FOLIO
- **ProofWriter:** https://allenai.org/data/proofwriter
- **Exact evaluated 600-question ProofWriter subset:** https://github.com/teacherpeterpan/Logic-LLM/tree/d29f1e5e7203e8826cb9c41ba3003e43dc0b9d34/data/ProofWriter
- **RuleTaker / ParaRules:** https://github.com/allenai/ruletaker
- **Original CLUTRR:** https://github.com/facebookresearch/clutrr
- **Pinned evaluated CLUTRR mirror:** https://github.com/kliang5/CLUTRR_huggingface_dataset/tree/e5b496941e91abb7c319d2618a3ce96752bc4ab7

Primary CLUTRR configuration: gen_train23_test2to10. Untouched robustness source: rob_train_clean_23_test_all_23/test.csv. The source-provenance files retain original hashes and notices. Dataset rights remain separate from the software MIT license; in particular CLUTRR upstream declares CC BY-NC 4.0.

## Local preparation

```bash
python prepare_code.py
# List exact download URLs without downloading:
python download_data.py
# Explicitly download the selected pinned inputs, checking SHA-256:
python download_data.py --dataset FOLIO_curated
python download_data.py --dataset LogicLM_ProofWriter
python download_data.py --dataset CLUTRR_external
```

The downloader retrieves only the named primary files. Other splits, RuleTaker archives and robustness inputs follow the original provenance/experiment records; they are not silently substituted or automatically downloaded.

Use a separate Python 3.12 environment and the appropriate requirements files. For the formal implementation checks (after preparation):

```bash
python -m pip install -r requirements-fol.txt
PYTHONPATH=src python -m unittest discover -s tests -p test_fol_interval.py -v
```

The original source modules are restored without changing their paths or logic. Data-dependent scripts must receive their required inputs first. Full replay of the frozen article tables additionally requires the separately preserved raw decisions and checkpoints from the full research archive; public datasets alone do not recreate those stored decisions. This lightweight package does not claim to include them. Do not rerun completed GPU generations just to verify an upload.

## Main entry points

- src/fol_interval_certifier.py: scoped must/may bounds and exact fallback.
- src/benchmark_fol_interval.py: formal-input comparison.
- src/evaluate_text_certificate.py: controlled-English interface.
- src/consistent_kinship.py: repaired relational adapter.
- src/run_unseen_robustness.py: frozen CPU robustness reproduction, requires saved models.
- src/analyze_published_baselines.py: analysis of the separately archived complete saved predictions.
- summary/: small published metric/interval tables only, not raw benchmark examples.

Code URL and software DOI remain to be supplied after deposition. Author/citation metadata is in CITATION.md. Do not replace missing results or DOI metadata with invented values.
