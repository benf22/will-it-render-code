"""will-it-render — AttnSteer: predict and steer whether a target class renders in a
generated video, from a DiT's cross-attention maps.

Subpackages:
    common      — dependency sink shared by everything (AttnSteer factory, feature
                  aggregation, prompt/config helpers, env-var path resolver).
    backends    — DiT backend adapters + attention-capture hooks + seed latents.
    inference   — steering optimization loop + the `run` entry point.
    datasets    — prompt generation, feature extraction, labeling.
    training    — AttnSteer architectures + trainer + AUROC/AP/F1 evaluation.
    evaluation  — det-all / det-ratio / VBench / runtime metrics + baselines.

Module-independence contract (plan §10c N2): `datasets`, `training`, `inference`, and
`evaluation` are standalone — each imports only `common` and `backends`, never a sibling.
"""

__version__ = "0.1.0"
