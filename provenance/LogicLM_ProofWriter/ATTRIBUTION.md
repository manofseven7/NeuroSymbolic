# Provenance and attribution

These are the public ProofWriter dev/test subsets distributed in the author repository for Logic-LM (Pan, Albalak, Wang and Wang, Findings EMNLP 2023). Exact repository commit, source URLs and SHA256 values are in provenance.json. The accompanying LICENSE is the original repository MIT notice, retained unchanged. It should not be confused with an independently verified license statement for every upstream ProofWriter artifact; original third-party rights are retained.

Upstream dataset: Tafjord, Dalvi and Clark, **ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language**, Findings ACL 2021, DOI 10.18653/v1/2021.findings-acl.317, https://aclanthology.org/2021.findings-acl.317/.

The author subset uses 600 examples from depth-5 OWA theories, balanced across true/false/unknown. It is not the full official train/test release and does not imply every question has five proof hops. Here dev is used as development/training data for small local baselines, and test is evaluated without tuning. No Logic-LM generated model programs, LLM outputs or published baseline results were imported.

Neither upstream authors nor dataset maintainers endorse the new pilot. The data and original notices are unmodified; diagnostic faults and derived results are kept separately.
