
# Cadence adjustment attenuates age-related differences in hip-to-ankle joint kinetics during walking

> Reginaldo Kisho Fukuchi, Claudiane Fukuchi, Marcos Duarte

[![Colab](https://colab.research.google.com/assets/colab-badge.svg) Effects of age and speed on step length and cadence during walking](https://colab.research.google.com/github/BMClab/h2a_age_spt/blob/main/notebooks/speed_age_spt.ipynb)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg) Associations with hip-to-ankle joint kinetics at similar speeds](https://colab.research.google.com/github/BMClab/h2a_age_spt/blob/main/notebooks/h2a_age_spt_comf_speed.ipynb)

The revised notebooks use local helpers in `notebooks/` and `scripts/`; run them
from a complete checkout. The Colab links above point to the published repository
and will not reflect local, unpublished changes.

## PeerJ revision

The working manuscript and response letter are the two DOCX files in the root.
Original inputs are preserved in `revision/originals/` and the latest author edits
in `revision/pre_opus_followup_2026-09-11/`; the first submitted
manuscript, `Redistribution_torque_gait2.docx`, is unchanged. Updated figures,
supplementary tables, PDF previews, and numerical evidence are in `revision/`.
See [REVISION_NOTES.md](REVISION_NOTES.md) for verified changes and unresolved
processing limitations. Exclusion issues and Figure 2 are deferred at the author’s request; no request for the unavailable upstream notebook is pending.

To activate the local environment, run this in a terminal from the repository root:

```sh
source .venv/bin/activate
```

Use `deactivate` to leave it. The commands below also work without activation:

```sh
.venv/bin/python scripts/execute_notebooks.py
.venv/bin/python scripts/diagnostics.py
.venv/bin/python scripts/revise_documents.py
```

For PDF previews and response-letter locations, render the manuscript first,
derive page/line references, then render the updated letter:

```sh
libreoffice --headless --convert-to pdf --outdir revision SPTparams_biomech_plasticity_PeerJ_revised.docx revision/Supplementary_tables.docx
.venv/bin/python scripts/locate_rebuttal.py
libreoffice --headless --convert-to pdf --outdir revision Rebuttal_letter.docx
.venv/bin/python scripts/validate_revision.py
```

Rebuilding the DOCX files replaces their generated working content. Preserve any
subsequent manual edits before rerunning `scripts/revise_documents.py`, or apply
those changes to its source first.

Both notebooks execute in fresh kernels. Their supplementary sections call
`notebooks/revision_analysis.py`, which can also run independently. Continuous
variables use full-sample means and sample SDs. Experiment 1 fits subject-specific
random intercepts and speed slopes in speed-containing models, with age-only
random-intercept reference fits; Experiment 2 fits OLS to one selected trial per
participant. Body-size, HC3, height, and relative-speed sensitivities are reported
in supplementary Tables S5–S6.
Sex and age-group indicators remain binary. Indirect-association bootstraps use
10,000 paired participant resamples with seed 20260910, percentile intervals as
primary and BCa as sensitivity; they do not establish causal mediation.

`revision/results/provenance.json` records input hashes and library versions;
`requirements-revision.txt` records the direct packages used. Source CSVs in
`data/` are unchanged. Fresh tables are written to `revision/results/`, not the
historical exports under `data/`.
