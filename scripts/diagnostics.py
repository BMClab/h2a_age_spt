"""Residual checks for the six principal Experiment 2 OLS models."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'notebooks'))
from revision_analysis import load_data, OUTCOMES


def make_diagnostics():
    _, data = load_data()
    paths = []
    for adjusted in [False, True]:
        fig, axes = plt.subplots(3, 2, figsize=(10, 10))
        for outcome, row in zip(OUTCOMES, axes):
            fit = smf.ols(f'{outcome} ~ AgeGroup + Cadence'+(' + Female' if adjusted else ''), data).fit()
            residual = fit.resid
            row[0].scatter(fit.fittedvalues, residual, alpha=.8)
            row[0].axhline(0, color='0.4', lw=1)
            row[0].set(xlabel='Fitted outcome (SD)', ylabel='Residual (SD)', title=outcome)
            for i in np.argsort(np.abs(residual))[-2:]:
                row[0].annotate(str(data.Subject.iloc[i]), (fit.fittedvalues.iloc[i], residual.iloc[i]))
            stats.probplot(residual, dist='norm', plot=row[1])
            row[1].set_title('Normal Q–Q')
        label = 'age_cadence_sex' if adjusted else 'age_cadence'
        fig.suptitle(f'OLS residual diagnostics: {label}; labels identify participant IDs')
        fig.tight_layout()
        out = ROOT/'figures'
        out.mkdir(parents=True, exist_ok=True)
        path = out/f'diagnostics_{label}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


if __name__ == '__main__':
    for path in make_diagnostics():
        print(path)
