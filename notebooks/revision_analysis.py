"""Reproducible supplementary analyses for the PeerJ revision.

Run through the two notebooks or: .venv/bin/python notebooks/revision_analysis.py
Inputs are immutable participant-level CSV exports, not raw gait trajectories.
"""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUTCOMES = ['H2A_M', 'H2A_I', 'H2A_W']
SEED = 20260910
N_BOOT = 10000


def write_table(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f'{name}.csv', index=False, float_format='%.12g')
    return frame


def load_data(all_speeds=False):
    name = 'h2a_allspeeds_43subs.csv' if all_speeds else 'h2a_comfspeed_43subs.csv'
    raw = pd.read_csv(ROOT / 'data' / name)
    assert raw.Subject.nunique() == 43
    assert not raw.isna().any().any()
    assert len(raw) == (258 if all_speeds else 43)
    if all_speeds:
        assert raw.groupby('Subject').SpeedCat.nunique().eq(6).all()
        assert not raw.duplicated(['Subject', 'SpeedCat']).any()
    data = raw.copy()
    data['AgeGroup'] = data.AgeGroup.map({'Young': 0, 'Older': 1})
    data['Female'] = data.Gender.eq('F').astype(int)
    for col in ['Speed', 'StepLength', 'Cadence', *OUTCOMES]:
        data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=1)
    data['Speed2'] = data.Speed ** 2
    return raw, data


def coefficient_rows(fit, model, outcome, fit_ml=None, sample='full', **metadata):
    fit_ml = fit if fit_ml is None else fit_ml
    names = fit.model.exog_names
    ci = fit.conf_int()
    r2 = np.corrcoef(fit.model.endog, fit.predict())[0, 1] ** 2
    return [dict(model=model, outcome=outcome, sample=sample, term=t,
                 estimate=fit.params[t], se=fit.bse[t], ci_low=ci.loc[t, 0],
                 ci_high=ci.loc[t, 1], p=fit.pvalues[t], n=int(fit.nobs),
                 r2=r2, llf=fit_ml.llf, aic=fit_ml.aic,
                 formula=fit.model.formula, **metadata) for t in names]


def indirect_effect(values, adjusted=False):
    """OLS product a*b; data columns are age, mediator, outcome, female.

    Scaling remains fixed at full-sample SD units throughout resampling.
    Whole rows are resampled together, preserving within-person associations.
    """
    age, mediator, outcome, female = values.T
    xa = np.column_stack([np.ones(len(age)), age])
    xb = np.column_stack([np.ones(len(age)), age, mediator])
    if adjusted:
        xa = np.column_stack([xa, female])
        xb = np.column_stack([xb, female])
    a = np.linalg.lstsq(xa, mediator, rcond=None)[0][1]
    b = np.linalg.lstsq(xb, outcome, rcond=None)[0][2]
    return a * b


def bootstrap_indirect(data, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(n_boot, len(data)))
    rows = []
    for adjusted in [False, True]:
        cov = ' + Female' if adjusted else ''
        for mediator in ['Cadence', 'StepLength']:
            a_fit = smf.ols(f'{mediator} ~ AgeGroup{cov}', data).fit()
            for outcome in OUTCOMES:
                b_fit = smf.ols(f'{outcome} ~ AgeGroup + {mediator}{cov}', data).fit()
                c_fit = smf.ols(f'{outcome} ~ AgeGroup{cov}', data).fit()
                values = data[['AgeGroup', mediator, outcome, 'Female']].to_numpy()
                estimate = indirect_effect(values, adjusted)
                boots = np.array([indirect_effect(values[ix], adjusted) for ix in indices])
                pct = np.quantile(boots, [.025, .975])
                jack = np.array([indirect_effect(np.delete(values, i, axis=0), adjusted)
                                 for i in range(len(values))])
                diff = jack.mean() - jack
                acceleration = np.sum(diff**3) / (6 * np.sum(diff**2)**1.5)
                rank = (np.sum(boots < estimate) + .5*np.sum(boots == estimate))/n_boot
                z0 = stats.norm.ppf(np.clip(rank, .5/n_boot, 1-.5/n_boot))
                z = stats.norm.ppf([.025, .975])
                probs = stats.norm.cdf(z0 + (z0+z)/(1-acceleration*(z0+z)))
                bca = np.quantile(boots, probs)
                a, b = a_fit.params['AgeGroup'], b_fit.params[mediator]
                total, direct = c_fit.params['AgeGroup'], b_fit.params['AgeGroup']
                assert np.isclose(estimate, a*b)
                assert np.isclose(total-direct, estimate)
                rows.append(dict(mediator=mediator, outcome=outcome,
                                 sex_adjusted=adjusted, a=a, b=b, total=total,
                                 direct=direct, indirect=estimate,
                                 percentile_low=pct[0], percentile_high=pct[1],
                                 bca_low=bca[0], bca_high=bca[1], z0=z0,
                                 acceleration=acceleration,
                                 attenuation_pct=100*estimate/total,
                                 n_boot=n_boot, seed=seed))
    # No bootstrap tail-count p values: these are not calibrated null tests.
    return write_table(pd.DataFrame(rows), 'indirect_associations')


def experiment2():
    raw, data = load_data()
    base = ['AgeGroup', 'StepLength', 'Cadence', 'AgeGroup + StepLength',
            'AgeGroup + Cadence', 'AgeGroup * StepLength', 'AgeGroup * Cadence',
            'AgeGroup * StepLength + AgeGroup * Cadence']
    primary, extra, interactions, vif_rows = [], [], [], []
    for k, outcome in enumerate(OUTCOMES):
        for i, rhs in enumerate(base):
            fit = smf.ols(f'{outcome} ~ {rhs}', data).fit()
            primary += coefficient_rows(fit, k*8+i+1, outcome)
            for term in [t for t in fit.params.index if ':' in t]:
                se = fit.bse[term]
                critical = stats.t.ppf(.975, fit.df_resid)
                # Approximate fixed-design sensitivity, conditional on observed SE;
                # not a prospective sample-size calculation or evidence for a null.
                mde = se * (critical + stats.norm.ppf(.8))
                interactions.append(dict(model=k*8+i+1, outcome=outcome, term=term,
                                         estimate=fit.params[term], se=se,
                                         ci_low=fit.conf_int().loc[term, 0],
                                         ci_high=fit.conf_int().loc[term, 1],
                                         p=fit.pvalues[term], approximate_mde80=mde))
        for name, rhs in [('age_sex', 'AgeGroup + Female'),
                          ('cadence_sex', 'AgeGroup + Cadence + Female'),
                          ('step_length_sex', 'AgeGroup + StepLength + Female'),
                          ('both_parameters', 'AgeGroup + StepLength + Cadence')]:
            fit = smf.ols(f'{outcome} ~ {rhs}', data).fit()
            extra += coefficient_rows(fit, name, outcome)
            if name == 'both_parameters':
                for j, term in enumerate(fit.model.exog_names[1:], 1):
                    vif_rows.append(dict(outcome=outcome, term=term,
                                         vif=variance_inflation_factor(fit.model.exog, j)))
    tables = dict(primary=write_table(pd.DataFrame(primary), 'experiment2_primary'),
                  sensitivity=write_table(pd.DataFrame(extra), 'experiment2_sensitivity'),
                  interactions=write_table(pd.DataFrame(interactions), 'experiment2_interactions'),
                  vif=write_table(pd.DataFrame(vif_rows), 'experiment2_vif'))
    tables['indirect'] = bootstrap_indirect(data)
    counts = pd.crosstab(raw.AgeGroup, raw.Gender).reindex(['Young', 'Older'])
    write_table(counts.reset_index(), 'sex_counts')
    write_table(pd.DataFrame([dict(fisher_p=stats.fisher_exact(counts).pvalue,
                                  cadence_step_length_r=data.Cadence.corr(data.StepLength))]), 'sample_statistics')
    tables.update(experiment2_followup())
    return tables


def mixed_diagnostic(fit, model, outcome, sample, structure, scaling, role):
    """Covariance diagnostics are in the actual fitted outcome/predictor units."""
    covariance = np.asarray(fit.cov_re)
    eigenvalues = np.linalg.eigvalsh(covariance)
    slope_variance = covariance[1, 1] if len(covariance) == 2 else np.nan
    correlation = (covariance[0, 1]/np.sqrt(covariance[0, 0]*slope_variance)
                   if len(covariance) == 2 and covariance[0, 0]*slope_variance > 0
                   else np.nan)
    return dict(model=model, outcome=outcome, sample=sample, role=role,
                formula=fit.model.formula, structure=structure, scaling=scaling,
                estimator='REML' if fit.reml else 'ML', converged=bool(fit.converged),
                llf=fit.llf, aic=fit.aic, residual_variance=fit.scale,
                intercept_variance=covariance[0, 0], slope_variance=slope_variance,
                intercept_slope_covariance=covariance[0, 1] if len(covariance) == 2 else np.nan,
                intercept_slope_correlation=correlation, min_eigenvalue=eigenvalues.min(),
                max_eigenvalue=eigenvalues.max(), covariance_positive_definite=bool(eigenvalues.min() > 0),
                eigenvalue_ratio=eigenvalues.min()/eigenvalues.max(),
                n=int(fit.nobs), subjects=fit.model.n_groups)


def fit_mixed_pair(formula, data, re_formula='1'):
    # Separate model instances: fit() mutates the model REML setting.
    ml = smf.mixedlm(formula, data, groups=data.Subject, re_formula=re_formula).fit(reml=False)
    re = smf.mixedlm(formula, data, groups=data.Subject, re_formula=re_formula).fit(reml=True)
    assert ml.converged and re.converged, (formula, re_formula)
    assert np.isfinite(ml.llf) and np.isfinite(re.llf)
    assert np.isfinite(re.fe_params).all() and np.isfinite(re.bse_fe).all()
    return ml, re


def fixed_effect_comparisons(fits, predictors, outcome, sample, structure, scaling):
    rows = []
    for reduced, full, term in [(2, 5, 'AgeGroup:Speed (linear)'),
                                (4, 6, 'AgeGroup:Speed (quadratic)'),
                                (2, 4, 'Speed2'), (3, 4, 'AgeGroup')]:
        # Each comparison has the same random structure and observations.
        assert fits[reduced].model.k_re == fits[full].model.k_re
        lr = 2*(fits[full].llf-fits[reduced].llf)
        assert lr >= -1e-6
        rows.append(dict(outcome=outcome, sample=sample, term=term,
                         reduced=predictors[reduced], full=predictors[full],
                         lr=lr, df=1, p=stats.chi2.sf(max(0, lr), 1),
                         subjects=fits[full].model.n_groups, n=int(fits[full].nobs),
                         structure=structure, scaling=scaling,
                         inference='asymptotic ML fixed-effect LRT; identical random structure'))
    return rows


def experiment1():
    """Random slopes primary for fixed-speed models; RI models as sensitivity.

    The age-only model remains RI: it is a descriptive reference, not nested
    in the speed-containing RS fits for a fixed-effect likelihood-ratio test.
    """
    raw, data = load_data(True)
    predictors = ['AgeGroup', 'Speed', 'AgeGroup + Speed', 'Speed + Speed2',
                  'AgeGroup + Speed + Speed2', 'AgeGroup * Speed',
                  'AgeGroup * Speed + Speed2']
    scaling = 'Speed and outcomes full-sample z scores, ddof=1; Speed2=squared z Speed; AgeGroup=0/1'
    coefficients, comparisons, ri_coefficients, ri_comparisons = [], [], [], []
    diagnostics, structure_rows = [], []
    for sample, subset in [('full', data), ('omit_7_31', data[~data.Subject.isin([7, 31])])]:
        assert subset.Subject.nunique() == (43 if sample == 'full' else 41)
        assert len(subset) == (258 if sample == 'full' else 246)
        for k, outcome in enumerate(['StepLength', 'Cadence']):
            primary_ml, ri_ml = [], []
            for i, rhs in enumerate(predictors):
                model = k*7+i+1
                formula = f'{outcome} ~ {rhs}'
                ml_ri, re_ri = fit_mixed_pair(formula, subset)
                ri_ml.append(ml_ri)
                ri_coefficients += coefficient_rows(re_ri, model, outcome, ml_ri, sample,
                                                     structure='random_intercept', scaling=scaling)
                for fit in [ml_ri, re_ri]:
                    diagnostics.append(mixed_diagnostic(fit, model, outcome, sample,
                                                        'random_intercept', scaling, 'RI sensitivity'))
                if i == 0:
                    ml, re = ml_ri, re_ri
                    structure = 'random_intercept_age_only_reference'
                else:
                    ml, re = fit_mixed_pair(formula, subset, re_formula='~Speed')
                    structure = 'correlated_random_intercept_speed_slope'
                    for fit in [ml, re]:
                        diagnostics.append(mixed_diagnostic(fit, model, outcome, sample,
                                                            structure, scaling, 'primary'))
                    lr = 2*(ml.llf-ml_ri.llf)
                    assert lr >= -1e-6
                    structure_rows.append(dict(model=model, outcome=outcome, sample=sample,
                                               formula=formula, ri_llf=ml_ri.llf, rs_llf=ml.llf,
                                               ri_aic=ml_ri.aic, rs_aic=ml.aic, descriptive_lr=lr,
                                               parameter_count_difference=2, p=np.nan,
                                               inference='descriptive LR only; null slope variance is on a boundary; no calibrated chi-square p',
                                               scaling=scaling, n=len(subset), subjects=subset.Subject.nunique()))
                primary_ml.append(ml)
                coefficients += coefficient_rows(re, model, outcome, ml, sample,
                                                  structure=structure, scaling=scaling)
            comparisons += fixed_effect_comparisons(primary_ml, predictors, outcome, sample,
                                                     'correlated_random_intercept_speed_slope', scaling)
            ri_comparisons += fixed_effect_comparisons(ri_ml, predictors, outcome, sample,
                                                        'random_intercept', scaling)
    tables = dict(coefficients=write_table(pd.DataFrame(coefficients), 'experiment1_coefficients'),
                  lrt=write_table(pd.DataFrame(comparisons), 'experiment1_lrt'),
                  ri_coefficients=write_table(pd.DataFrame(ri_coefficients), 'experiment1_ri_coefficients'),
                  ri_lrt=write_table(pd.DataFrame(ri_comparisons), 'experiment1_ri_lrt'),
                  random_structure=write_table(pd.DataFrame(structure_rows), 'experiment1_random_structure'),
                  diagnostics=write_table(pd.DataFrame(diagnostics), 'experiment1_fit_diagnostics'))
    tables.update(experiment1_body_size())
    return tables


def dimensionless_data(raw):
    """Hof-type scaling: L in metres, speed m/s, cadence steps/min; g=9.81."""
    result = raw.copy()
    result['AgeGroup'] = raw.AgeGroup.map({'Young': 0, 'Older': 1})
    result['Female'] = raw.Gender.eq('F').astype(int)
    result['StepLengthHof'] = raw.StepLength/raw.LegLength
    result['CadenceHof'] = raw.Cadence/60*np.sqrt(raw.LegLength/9.81)
    result['SpeedHof'] = raw.Speed/np.sqrt(9.81*raw.LegLength)
    result['RelativeSpeed'] = raw.Speed/raw.SpeedComf
    return result


def experiment1_body_size():
    """Separate physical-speed and dimensionless-speed estimands explicitly."""
    raw, _ = load_data(True)
    data = dimensionless_data(raw)
    rows, diagnostics, comparisons = [], [], []
    settings = [
        ('hof_physical_speed', 'Speed', ['StepLengthHof', 'CadenceHof'], '',
         'Hof outcomes in dimensionless units; Speed m/s; matched physical speed'),
        ('hof_dimensionless_speed', 'SpeedHof', ['StepLengthHof', 'CadenceHof'], '',
         'Hof outcomes and Speed/sqrt(g*LegLength) in dimensionless units; matched dimensionless speed'),
        ('leg_length_covariate', 'Speed', ['StepLength', 'Cadence'], ' + LegLength',
         'raw outcomes (m, steps/min), Speed m/s, LegLength m; matched physical speed and leg length')]
    for setting, speed, outcomes, covariate, scaling in settings:
        for outcome in outcomes:
            fits = {}
            for name, rhs in [
                ('no_age', f'{speed} + I({speed} ** 2){covariate}'),
                ('additive', f'AgeGroup + {speed} + I({speed} ** 2){covariate}'),
                ('interaction', f'AgeGroup * {speed} + I({speed} ** 2){covariate}')]:
                model = f'{setting}_{name}'
                ml, re = fit_mixed_pair(f'{outcome} ~ {rhs}', data, re_formula=f'~{speed}')
                fits[name] = ml
                structure = f'correlated_random_intercept_{speed}_slope'
                rows += coefficient_rows(re, model, outcome, ml, scaling=scaling,
                                         structure=structure, setting=setting, speed_variable=speed)
                for fit in [ml, re]:
                    diagnostics.append(mixed_diagnostic(fit, model, outcome, 'full', structure, scaling, 'body-size sensitivity'))
            for reduced, full, term in [('no_age', 'additive', 'AgeGroup'),
                                       ('additive', 'interaction', f'AgeGroup:{speed}')]:
                lr = 2*(fits[full].llf-fits[reduced].llf)
                assert lr >= -1e-6
                comparisons.append(dict(setting=setting, outcome=outcome, term=term,
                                        reduced=fits[reduced].model.formula, full=fits[full].model.formula,
                                        lr=lr, df=1, p=stats.chi2.sf(max(0, lr), 1), n=258, subjects=43,
                                        structure=structure, scaling=scaling))
    return dict(body_size=write_table(pd.DataFrame(rows), 'experiment1_body_size'),
                body_size_lrt=write_table(pd.DataFrame(comparisons), 'experiment1_body_size_lrt'),
                body_size_diagnostics=write_table(pd.DataFrame(diagnostics), 'experiment1_body_size_diagnostics'))


def experiment2_followup():
    """HC3 t inference and independent size/relative-speed covariate checks.

    Existing primary, sex-adjusted, interaction and bootstrap tables are untouched.
    Hof substitutions are standardized after physical normalization, using the
    full sample and ddof=1, so attenuation is on the original outcome SD scale.
    """
    raw, data = load_data()
    dim = dimensionless_data(raw)
    for variable in ['Height', 'RelativeSpeed', 'StepLengthHof', 'CadenceHof']:
        source = dim[variable]
        data[variable] = (source-source.mean())/source.std(ddof=1)
    scaling = 'continuous variables full-sample z scores ddof=1; AgeGroup and Female=0/1'
    extra, attenuation, model_checks = [], [], []
    settings = [
        ('cadence_sex_classical', 'AgeGroup + Cadence + Female', 'nonrobust'),
        ('cadence_sex_hc3', 'AgeGroup + Cadence + Female', 'HC3'),
        ('cadence_sex_height', 'AgeGroup + Cadence + Female + Height', 'nonrobust'),
        ('cadence_sex_relative_speed', 'AgeGroup + Cadence + Female + RelativeSpeed', 'nonrobust')]
    for outcome in OUTCOMES:
        for name, rhs, cov_type in settings:
            fit = smf.ols(f'{outcome} ~ {rhs}', data).fit(cov_type=cov_type, use_t=True)
            assert fit.use_t
            extra += coefficient_rows(fit, name, outcome, covariance=cov_type,
                                       inference='Student t, residual degrees of freedom', df_resid=fit.df_resid,
                                       scaling=scaling, structure='OLS independent subjects')
            influence = fit.get_influence()
            model_checks.append(dict(model=name, outcome=outcome, formula=fit.model.formula,
                                     covariance=cov_type, use_t=fit.use_t, df_resid=fit.df_resid,
                                     n=43, r2=fit.rsquared, design_rank=np.linalg.matrix_rank(fit.model.exog),
                                     condition_number=fit.condition_number,
                                     max_leverage=influence.hat_matrix_diag.max(),
                                     max_cooks_distance=influence.cooks_distance[0].max()))
        for sex_adjusted in [False, True]:
            covariate = ' + Female' if sex_adjusted else ''
            age_only = smf.ols(f'{outcome} ~ AgeGroup{covariate}', data).fit()
            for predictor in ['CadenceHof', 'StepLengthHof']:
                fit = smf.ols(f'{outcome} ~ AgeGroup + {predictor}{covariate}', data).fit()
                model = f'{predictor}' + ('_sex' if sex_adjusted else '')
                extra += coefficient_rows(fit, model, outcome, covariance='nonrobust',
                                           inference='Student t, residual degrees of freedom', df_resid=fit.df_resid,
                                           scaling='Hof parameter normalized first, then '+scaling,
                                           structure='OLS independent subjects')
                total, adjusted = age_only.params.AgeGroup, fit.params.AgeGroup
                attenuation.append(dict(outcome=outcome, predictor=predictor, sex_adjusted=sex_adjusted,
                                        age_reference=total, age_adjusted=adjusted,
                                        attenuation=total-adjusted, attenuation_pct=100*(total-adjusted)/total,
                                        age_adjusted_ci_low=fit.conf_int().loc['AgeGroup', 0],
                                        age_adjusted_ci_high=fit.conf_int().loc['AgeGroup', 1],
                                        age_adjusted_p=fit.pvalues.AgeGroup, r2=fit.rsquared,
                                        formula=fit.model.formula, reference_formula=age_only.model.formula,
                                        scaling=scaling, inference='associational coefficient attenuation; not causal mediation'))
    descriptions = []
    # One selected trial per subject: Welch tests use 43 independent units.
    for variable in ['StepLengthHof', 'CadenceHof', 'SpeedHof', 'RelativeSpeed', 'LegLength', 'Height']:
        older, young = [dim.loc[dim.AgeGroup.eq(group), variable] for group in [1, 0]]
        test = stats.ttest_ind(older, young, equal_var=False)
        ci = test.confidence_interval()
        pooled = np.sqrt(((len(older)-1)*older.var(ddof=1)+(len(young)-1)*young.var(ddof=1))/(len(dim)-2))
        descriptions.append(dict(variable=variable, young_n=len(young), older_n=len(older),
                                 young_mean=young.mean(), young_sd=young.std(ddof=1),
                                 older_mean=older.mean(), older_sd=older.std(ddof=1),
                                 difference=older.mean()-young.mean(), percent_difference=100*(older.mean()-young.mean())/young.mean(),
                                 ci_low=ci.low, ci_high=ci.high, p=test.pvalue, df=test.df,
                                 cohen_d=(older.mean()-young.mean())/pooled,
                                 inference='Welch t; Older minus Young; independent selected trials',
                                 scaling='Hof dimensionless variables before z scoring; relative speed=Speed/SpeedComf; length and height m'))
    return dict(followup=write_table(pd.DataFrame(extra), 'experiment2_followup'),
                hof_attenuation=write_table(pd.DataFrame(attenuation), 'experiment2_hof_attenuation'),
                followup_diagnostics=write_table(pd.DataFrame(model_checks), 'experiment2_followup_diagnostics'),
                normalized_descriptive=write_table(pd.DataFrame(descriptions), 'normalized_descriptive'))


def descriptive():
    raw, _ = load_data()
    rows = []
    for col in ['Age', 'Height', 'Mass', 'BMI', 'LegLength', 'Speed', 'StepLength', 'Cadence', *OUTCOMES]:
        older, young = [raw.loc[raw.AgeGroup.eq(g), col] for g in ['Older', 'Young']]
        test = stats.ttest_ind(older, young, equal_var=False)
        ci = test.confidence_interval()
        pooled = np.sqrt(((len(older)-1)*older.var()+(len(young)-1)*young.var())/(len(raw)-2))
        rows.append(dict(variable=col, young_mean=young.mean(), young_sd=young.std(),
                         older_mean=older.mean(), older_sd=older.std(),
                         difference=older.mean()-young.mean(),
                         percent_difference=100*(older.mean()-young.mean())/young.mean(),
                         ci_low=ci.low, ci_high=ci.high, p=test.pvalue,
                         cohen_d=(older.mean()-young.mean())/pooled))
    return write_table(pd.DataFrame(rows), 'descriptive')


def provenance():
    OUT.mkdir(parents=True, exist_ok=True)
    files = list((ROOT/'data').glob('*43subs.csv'))
    record = dict(python=platform.python_version(),
                  packages={x: importlib.metadata.version(x) for x in
                            ['numpy', 'scipy', 'statsmodels', 'pandas', 'nbclient', 'python-docx', 'pingouin']},
                  inputs={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  bootstrap_seed=SEED, bootstrap_samples=N_BOOT,
                  scaling='Continuous variables: full-sample mean and sample SD (ddof=1); age and sex: 0/1',
                  bootstrap='Paired subject rows, fixed scaling; percentile primary, BCa sensitivity; no causal interpretation')
    (OUT/'provenance.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    provenance()
    descriptive()
    print('Experiment 1', flush=True)
    experiment1()
    print('Experiment 2, including 10,000 bootstrap resamples per path', flush=True)
    experiment2()
    print(f'Completed: {OUT}', flush=True)
