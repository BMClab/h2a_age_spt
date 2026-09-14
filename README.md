
# Cadence adjustment attenuates age-related differences in hip-to-ankle joint kinetics during walking

> Reginaldo Kisho Fukuchi, Claudiane Fukuchi, Marcos Duarte

## Analyses

[![Colab](https://colab.research.google.com/assets/colab-badge.svg) Effects of age and speed on step length and cadence during walking](https://colab.research.google.com/github/BMClab/h2a_age_spt/blob/main/notebooks/speed_age_spt.ipynb)

[![Colab](https://colab.research.google.com/assets/colab-badge.svg) Associations with hip-to-ankle joint kinetics at similar speeds](https://colab.research.google.com/github/BMClab/h2a_age_spt/blob/main/notebooks/h2a_age_spt_comf_speed.ipynb)

## Dataset metadata

[![Colab](https://colab.research.google.com/assets/colab-badge.svg) WBDS metadata sheet preparation](https://colab.research.google.com/github/BMClab/h2a_age_spt/blob/main/notebooks/WBDSinfo.ipynb)

This notebook builds the WBDS metadata sheet from the raw WBDS files (Fukuchi et al., 2018), which are downloaded separately from the WBDS Figshare repository and are not included here.

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/speed_age_spt.ipynb
```

Running the notebooks regenerates `figures/` and `results/` from `data/`.
