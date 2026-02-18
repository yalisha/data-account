# Data-Account Research Project

This repository tracks code, docs, and lightweight result tables for the
`15会计研究` project.

## Included
- Python scripts for data processing and empirical analysis
- Markdown docs in `docs/`
- Reproducible CSV outputs in `results/*.csv`
- Research framework `.svg` files

## Excluded
- Raw and intermediate data (`data_parquet/`, `annual_reports/`, linked third-party data)
- Binary figures and PDFs (`*.png`, `*.jpg`, `*.jpeg`, `*.pdf`)
- Cache and local environment files

## Re-run Core Analyses

```bash
/opt/miniconda3/envs/did-r/bin/python run_regression_v2.py
/opt/miniconda3/envs/did-r/bin/python run_robustness.py
/opt/miniconda3/envs/did-r/bin/python portfolio_backtest.py
/opt/miniconda3/envs/did-r/bin/python run_shap_analysis.py
```
