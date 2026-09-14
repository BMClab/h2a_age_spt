"""Regenerate Figure 1 from all 43 participants; no historical cache inputs."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from patsy import build_design_matrices
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


def make_figure1(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / 'data/h2a_allspeeds_43subs.csv')
    assert data.Subject.nunique() == 43 and len(data) == 258
    assert data.groupby('Subject').SpeedCat.nunique().eq(6).all()
    comfortable = data.drop_duplicates('Subject').SpeedComf
    mean_speed = comfortable.mean()
    speed_ci = stats.t.interval(.95, len(comfortable)-1, loc=mean_speed,
                               scale=stats.sem(comfortable))
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 8), sharex=True)
    fits, predictions = {}, []
    colors = {'Young': '#0072B2', 'Older': '#D55E00'}
    markers = {'Young': 'v', 'Older': '^'}
    for ax, outcome, label, panel in zip(
            axes, ['StepLength', 'Cadence'],
            ['Step length [m]', 'Cadence [steps/min]'], ['A', 'B']):
        fit = smf.mixedlm(f'{outcome} ~ C(AgeGroup) + Speed + I(Speed ** 2)',
                          data, groups=data.Subject, re_formula='~Speed').fit(reml=True)
        assert fit.converged and fit.model.k_re == 2, outcome
        assert np.linalg.eigvalsh(fit.cov_re).min() > 0, outcome
        fits[outcome] = fit
        names = fit.fe_params.index
        covariance = fit.cov_params().loc[names, names].to_numpy()
        ax.axvspan(*speed_ci, color='0.2', alpha=.10, zorder=0)
        ax.axvline(mean_speed, color='0.25', lw=1, ls='--')
        for age in ['Young', 'Older']:
            group = data.loc[data.AgeGroup.eq(age)]
            grid = pd.DataFrame({'Speed': np.linspace(group.Speed.min(), group.Speed.max(), 150),
                                 'AgeGroup': age})
            design = np.asarray(build_design_matrices([fit.model.data.design_info], grid)[0])
            mu = design @ fit.fe_params.to_numpy()
            variance = np.einsum('ij,jk,ik->i', design, covariance, design)
            assert np.all(variance >= 0)
            half_width = stats.norm.ppf(.975) * np.sqrt(variance)
            assert np.allclose(mu, fit.predict(grid))
            ax.scatter(group.Speed, group[outcome], color=colors[age], marker=markers[age],
                       s=24, alpha=.65, label=age, edgecolors='none')
            ax.fill_between(grid.Speed, mu-half_width, mu+half_width,
                            color=colors[age], alpha=.18)
            ax.plot(grid.Speed, mu, color=colors[age], lw=2)
            predictions.append(grid.assign(outcome=outcome, mean=mu, ci_low=mu-half_width,
                                           ci_high=mu+half_width, fixed_mean_se=np.sqrt(variance),
                                           structure='correlated_random_intercept_speed_slope',
                                           formula=fit.model.formula, estimator='REML'))
        ax.set_ylabel(label, fontsize=12)
        ax.text(-.10, 1.02, panel, transform=ax.transAxes, fontsize=14, weight='bold')
        ax.grid(False)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(title='Age group', frameon=False)
    axes[1].set_xlabel('Speed [m/s]', fontsize=12)
    axes[1].text(.98, .05, f'Mean comfortable speed: {mean_speed:.2f} m/s',
                 transform=axes[1].transAxes, ha='right', fontsize=10,
                 bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85})
    fig.tight_layout()
    out = root / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'figure1.png', dpi=300)
    fig.savefig(out / 'figure1.svg')
    table = pd.concat(predictions, ignore_index=True)
    result_dir = root / 'results'
    result_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(result_dir / 'figure1_predictions.csv', index=False)
    pd.DataFrame([{'subjects': 43, 'mean': mean_speed, 'ci_low': speed_ci[0],
                   'ci_high': speed_ci[1], 'method': 'subject-level t, df=42'}]).to_csv(
                       result_dir / 'figure1_comfortable_speed.csv', index=False)
    return fig, fits, table


if __name__ == '__main__':
    make_figure1()
