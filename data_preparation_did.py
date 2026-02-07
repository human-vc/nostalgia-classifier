import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

np.random.seed(42)


def aggregate_nostalgia_by_dma(ads_df, dma_col='dma', label_col='nostalgic'):
    dma_agg = ads_df.groupby(dma_col).agg(
        nostalgic_count=(label_col, 'sum'),
        total_count=(label_col, 'count')
    ).reset_index()
    dma_agg['nostalgia_pct'] = (dma_agg['nostalgic_count'] / dma_agg['total_count']) * 100
    return dma_agg


def assign_nostalgia_to_counties(dma_nostalgia, dma_county_map, dma_col='dma'):
    merged = dma_county_map.merge(
        dma_nostalgia[[dma_col, 'nostalgia_pct']], on=dma_col, how='left'
    )
    return merged


def compute_turnout(votes_df, votes_col='total_votes', pop_col='population'):
    votes_df = votes_df.copy()
    votes_df['turnout_pct'] = (votes_df[votes_col] / votes_df[pop_col]) * 100
    return votes_df


def prepare_did_dataset(ads_2020, ads_2024, turnout_2020, turnout_2024,
                        demographics, dma_county_map):
    nost_2020 = aggregate_nostalgia_by_dma(ads_2020)
    nost_2024 = aggregate_nostalgia_by_dma(ads_2024)

    county_nost_2020 = assign_nostalgia_to_counties(nost_2020, dma_county_map)
    county_nost_2024 = assign_nostalgia_to_counties(nost_2024, dma_county_map)

    turnout_2020 = compute_turnout(turnout_2020)
    turnout_2024 = compute_turnout(turnout_2024)

    df = county_nost_2020[['county_fips', 'state', 'county_name', 'dma']].copy()

    df = df.merge(
        county_nost_2020[['county_fips', 'nostalgia_pct']],
        on='county_fips', how='left'
    )
    df = df.rename(columns={'nostalgia_pct': 'nostalgia_2020'})

    df = df.merge(
        county_nost_2024[['county_fips', 'nostalgia_pct']],
        on='county_fips', how='left'
    )
    df = df.rename(columns={'nostalgia_pct': 'nostalgia_2024'})

    df = df.merge(
        turnout_2020[['county_fips', 'turnout_pct']],
        on='county_fips', how='left'
    )
    df = df.rename(columns={'turnout_pct': 'turnout_2020'})

    df = df.merge(
        turnout_2024[['county_fips', 'turnout_pct']],
        on='county_fips', how='left'
    )
    df = df.rename(columns={'turnout_pct': 'turnout_2024'})

    df = df.merge(
        demographics[['county_fips', 'pct_white', 'pct_black', 'pct_college', 'median_income']],
        on='county_fips', how='left'
    )

    df['delta_nostalgia'] = df['nostalgia_2024'] - df['nostalgia_2020']
    df['delta_turnout'] = df['turnout_2024'] - df['turnout_2020']

    df = df.dropna(subset=['delta_nostalgia', 'delta_turnout'])
    df = df.drop_duplicates(subset='county_fips')

    return df


def calculate_descriptive_statistics(df):
    return df.groupby('state').agg({
        'nostalgia_2020': ['mean', 'std'],
        'nostalgia_2024': ['mean', 'std'],
        'delta_nostalgia': ['mean', 'std'],
        'turnout_2020': ['mean', 'std'],
        'turnout_2024': ['mean', 'std'],
        'delta_turnout': ['mean', 'std']
    }).round(2)


def run_ols_regression(df, control_variables=None, cluster_var=None, standardize=False):
    if control_variables is None:
        X = df[['delta_nostalgia']].copy()
    else:
        X = df[['delta_nostalgia'] + list(control_variables)].copy()

    if standardize:
        X = (X - X.mean()) / X.std()

    X = sm.add_constant(X)
    y = df['delta_turnout'].copy()

    if cluster_var is not None and cluster_var in df.columns:
        model = sm.OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': df[cluster_var]}
        )
    else:
        model = sm.OLS(y, X).fit(cov_type='HC3')

    return model


def run_state_regression(df, state, control_variables=None, cluster_var=None):
    df_state = df[df['state'] == state].copy()
    return run_ols_regression(df_state, control_variables, cluster_var)


def run_subgroup_regression(df, group_var, threshold, control_variables=None,
                            cluster_var=None):
    df_above = df[df[group_var] > threshold].copy()
    df_below = df[df[group_var] <= threshold].copy()

    model_above = run_ols_regression(df_above, control_variables, cluster_var)
    model_below = run_ols_regression(df_below, control_variables, cluster_var)

    return model_above, model_below


def calculate_spearman(x, y):
    rho, p = spearmanr(x, y)
    return rho, p


def bootstrap_ci(x, y, n_boot=5000, alpha=0.05):
    np.random.seed(42)
    n = len(x)
    rhos = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        rho, _ = spearmanr(x.iloc[idx], y.iloc[idx])
        rhos.append(rho)
    lower = np.percentile(rhos, 100 * (alpha / 2))
    upper = np.percentile(rhos, 100 * (1 - alpha / 2))
    return lower, upper


def permutation_test(x, y, n_perm=5000):
    np.random.seed(42)
    observed, _ = spearmanr(x, y)
    count = 0
    for i in range(n_perm):
        y_perm = y.sample(frac=1, random_state=42 + i).reset_index(drop=True)
        rho_perm, _ = spearmanr(x, y_perm)
        if abs(rho_perm) >= abs(observed):
            count += 1
    return count / n_perm


def fisher_z_test(r1, n1, r2, n2):
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def calculate_vif(df, variables):
    X = df[variables].copy()
    X = (X - X.mean()) / X.std()
    X = sm.add_constant(X)
    vif = pd.DataFrame({
        'Variable': variables,
        'VIF': [variance_inflation_factor(X.values, i + 1) for i in range(len(variables))]
    })
    return vif


def extract_results(model):
    return {
        'coefficients': model.params,
        'std_errors': model.bse,
        'p_values': model.pvalues,
        't_statistics': model.tvalues,
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'n_observations': int(model.nobs),
        'aic': model.aic,
        'bic': model.bic
    }
