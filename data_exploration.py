#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from merge_data import merge_geodata_omi_redditi_votes_2021


OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
BOOTSTRAP_RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 5000

sns.set_theme(style='white', context='notebook')

ANALYSIS_COLUMNS = {
    'Voti si': 'yes_votes',
    'Voti validi': 'valid_votes',
    'Affluenza referendum 2026': 'turnout_referendum_2026',
    'Affluenza politiche 2022': 'turnout_politiche_2022',
    'Affluenza politiche 2022 male': 'turnout_politiche_2022_male',
    'Affluenza politiche 2022 female': 'turnout_politiche_2022_female',
    'Perc voti si': 'referendum_yes',
    'Perc voti no': 'referendum_no',
    'Perc voti coalizione - CENTRODESTRA': 'center_right',
    'Perc voti coalizione - CENTROSINISTRA': 'center_left',
    'Perc voti coalizione - M5S': 'm5s',
    'Perc voti coalizione - TERZO_POLO': 'third_pole',
    'Perc voti coalizione - ALTRI': 'others',
    'avg_income': 'avg_income',
    'price_m2': 'house_price_m2',
    'rent_m2': 'house_rent_m2',
    'population_2021': 'population_2021',
    'distance_to_milano_km': 'distance_to_milano_km',
}

PLOT_COLUMNS = [
    'referendum_yes',
    'referendum_no',
    'turnout_referendum_2026',
    'turnout_politiche_2022',
    'turnout_politiche_2022_male',
    'turnout_politiche_2022_female',
    'center_right',
    'center_left',
    'm5s',
    'third_pole',
    'others',
    'avg_income',
    'house_price_m2',
    'house_rent_m2',
    'distance_to_milano_km',
]

DISPLAY_LABELS = {
    'referendum_yes': 'yes_ref',
    'referendum_no': 'no_ref',
    'turnout_referendum_2026': 'turnout_ref26',
    'turnout_politiche_2022': 'turnout_pol22',
    'turnout_politiche_2022_male': 'turnout_male22',
    'turnout_politiche_2022_female': 'turnout_female22',
    'center_right': 'center_right',
    'center_left': 'center_left',
    'm5s': 'm5s',
    'third_pole': 'third_pole',
    'others': 'others',
    'avg_income': 'avg_income',
    'house_price_m2': 'price_m2',
    'house_rent_m2': 'rent_m2',
    'distance_to_milano_km': 'dist_milan_km',
}


def build_analysis_df(region: str = 'Lombardia') -> pd.DataFrame:
    df = merge_geodata_omi_redditi_votes_2021(region=region).copy()
    df = df.rename(columns=ANALYSIS_COLUMNS)

    keep_columns = ['Comune'] + list(ANALYSIS_COLUMNS.values())
    analysis_df = df[keep_columns].copy()
    analysis_df = analysis_df.dropna(subset=PLOT_COLUMNS)
    analysis_df = analysis_df[analysis_df['valid_votes'] > 0].copy()
    return analysis_df


def write_excel_with_fallback(sheets: dict[str, pd.DataFrame], output_path: Path) -> Path:
    try:
        with pd.ExcelWriter(output_path) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return output_path
    except PermissionError:
        fallback_path = output_path.with_stem(output_path.stem + '_updated')
        with pd.ExcelWriter(fallback_path) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return fallback_path


def gini_coefficient(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return np.nan
    if np.any(array < 0):
        raise ValueError('Gini requer valores nao negativos.')
    if np.allclose(array.sum(), 0):
        return 0.0

    sorted_array = np.sort(array)
    n = sorted_array.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * sorted_array) / (n * sorted_array.sum())) - (n + 1) / n)


def bootstrap_gini(
    values: pd.Series | np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    random_seed: int = BOOTSTRAP_RANDOM_SEED,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    rng = np.random.default_rng(random_seed)
    sample_size = array.size
    bootstrap_estimates = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = rng.choice(array, size=sample_size, replace=True)
        bootstrap_estimates[i] = gini_coefficient(sample)

    return bootstrap_estimates


def plot_correlation_triangle(
    analysis_df: pd.DataFrame,
    columns: list[str],
    output_path: Path,
    method: str = 'pearson',
    title: str = 'Triangular Correlation Matrix',
    cbar_label: str = 'Correlation',
) -> pd.DataFrame:
    corr = analysis_df[columns].corr(method=method)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    corr_plot = corr.rename(index=DISPLAY_LABELS, columns=DISPLAY_LABELS)

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_plot,
        mask=mask,
        cmap='RdBu_r',
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': cbar_label},
        ax=ax,
    )
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return corr


def plot_regression_triangle(
    analysis_df: pd.DataFrame,
    columns: list[str],
    output_path: Path,
) -> None:
    plot_df = analysis_df[columns].rename(columns=DISPLAY_LABELS)
    grid = sns.PairGrid(plot_df, corner=True, height=1.6, diag_sharey=False)
    grid.map_lower(
        sns.regplot,
        scatter_kws={'s': 14, 'alpha': 0.45, 'color': '#1f4e79'},
        line_kws={'color': '#c0392b', 'linewidth': 1.2},
        ci=None,
    )
    grid.map_diag(sns.histplot, bins=20, color='#4c956c', edgecolor='white')
    grid.figure.suptitle('Regression Triangle', y=1.02)
    grid.figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(grid.figure)


def plot_gini_bootstrap(
    analysis_df: pd.DataFrame,
    output_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    bootstrap_estimates = bootstrap_gini(analysis_df['avg_income'])
    quartiles = np.quantile(bootstrap_estimates, [0.25, 0.5, 0.75])
    observed_gini = gini_coefficient(analysis_df['avg_income'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(bootstrap_estimates, bins=40, color='#4c956c', edgecolor='white', ax=ax)
    ax.axvline(observed_gini, color='#c0392b', linewidth=2, label=f'Observed Gini = {observed_gini:.3f}')

    for quartile, label, color in zip(
        quartiles,
        ['Q1', 'Q2 (Median)', 'Q3'],
        ['#1f4e79', '#7f8c8d', '#f39c12'],
    ):
        ax.axvline(quartile, color=color, linestyle='--', linewidth=1.5, label=f'{label} = {quartile:.3f}')

    ax.set_title('Bootstrap Distribution of Gini for Average Income')
    ax.set_xlabel('Gini coefficient')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=True)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig) 

    summary = pd.DataFrame(
        {
            'statistic': ['observed_gini', 'q1', 'median', 'q3'],
            'value': [observed_gini, *quartiles],
        }
    )
    summary.to_csv(summary_path, index=False)
    return summary


def summarize_correlations(corr: pd.DataFrame, top_n: int = 10, value_name: str = 'correlation') -> pd.DataFrame:
    pairs = (
        corr.where(~np.eye(len(corr), dtype=bool))
        .stack()
        .reset_index()
        .rename(columns={'level_0': 'var_1', 'level_1': 'var_2', 0: value_name})
    )
    pairs['pair_key'] = pairs.apply(lambda row: tuple(sorted((row['var_1'], row['var_2']))), axis=1)
    pairs = pairs.drop_duplicates('pair_key').drop(columns='pair_key')
    pairs['abs_r'] = pairs[value_name].abs()
    return pairs.sort_values('abs_r', ascending=False).head(top_n)


def fit_binomial_model(
    analysis_df: pd.DataFrame,
    predictors: list[str],
    model_name: str,
) -> tuple[sm.GLM, pd.DataFrame]:
    model_df = analysis_df[['yes_votes', 'valid_votes', 'referendum_yes'] + predictors].dropna().copy()
    for column in ['yes_votes', 'valid_votes', 'referendum_yes'] + predictors:
        model_df[column] = pd.to_numeric(model_df[column], errors='coerce')
    model_df = model_df.dropna()
    X = sm.add_constant(model_df[predictors], has_constant='add').astype(float)
    y = model_df['referendum_yes'].astype(float)
    weights = model_df['valid_votes'].astype(float)

    result = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=weights,
    ).fit()

    coef_table = result.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
    coef_table.insert(0, 'model', model_name)
    coef_table['n_municipalities'] = len(model_df)
    coef_table['valid_votes_sum'] = model_df['valid_votes'].sum()
    return result, coef_table


def fit_binomial_models(analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_specs = {
        'binomial_income': ['avg_income'],
        'binomial_house_price': ['house_price_m2'],
        'binomial_house_rent': ['house_rent_m2'],
        'binomial_distance_milano': ['distance_to_milano_km'],
        'binomial_center_right': ['center_right'],
        'binomial_center_left': ['center_left'],
        'binomial_m5s': ['m5s'],
        'binomial_third_pole': ['third_pole'],
        'binomial_socioeconomic': ['avg_income', 'house_price_m2', 'house_rent_m2', 'distance_to_milano_km'],
        'binomial_political_economic': [
            'center_right',
            'center_left',
            'm5s',
            'third_pole',
            'avg_income',
            'house_price_m2',
            'house_rent_m2',
            'distance_to_milano_km',
        ],
    }

    coef_tables = []
    fit_summaries = []

    for model_name, predictors in model_specs.items():
        result, coef_table = fit_binomial_model(analysis_df, predictors, model_name)
        coef_tables.append(coef_table)
        fit_summaries.append(
            {
                'model': model_name,
                'predictors': ', '.join(predictors),
                'n_parameters': len(result.params),
                'log_likelihood': result.llf,
                'aic': result.aic,
                'bic_llf': result.bic_llf,
                'deviance': result.deviance,
                'pearson_chi2': result.pearson_chi2,
                'pseudo_r2_mcfadden': result.pseudo_rsquared(kind='mcf'),
            }
        )

    return pd.concat(coef_tables, ignore_index=True), pd.DataFrame(fit_summaries)


def run_analysis(region: str = 'Lombardia') -> pd.DataFrame:
    analysis_df = build_analysis_df(region=region)
    pearson_corr = plot_correlation_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=OUTPUT_DIR / 'correlation_triangle.png',
        method='pearson',
        title='Triangular Correlation Matrix - Pearson',
        cbar_label='Pearson r',
    )
    spearman_corr = plot_correlation_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=OUTPUT_DIR / 'spearman_correlation_triangle.png',
        method='spearman',
        title='Triangular Correlation Matrix - Spearman',
        cbar_label='Spearman rho',
    )
    plot_regression_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=OUTPUT_DIR / 'regression_triangle.png',
    )

    analysis_df.to_excel(OUTPUT_DIR / 'analysis_dataset.xlsx', index=False)
    top_pearson = summarize_correlations(pearson_corr, value_name='pearson_r')
    top_pearson.to_csv(OUTPUT_DIR / 'top_correlations.csv', index=False)
    pearson_corr.to_csv(OUTPUT_DIR / 'pearson_correlation_matrix.csv')
    top_spearman = summarize_correlations(spearman_corr, value_name='spearman_rho')
    top_spearman.to_csv(OUTPUT_DIR / 'top_spearman_correlations.csv', index=False)
    spearman_corr.to_csv(OUTPUT_DIR / 'spearman_correlation_matrix.csv')
    binomial_coefficients, binomial_model_fit = fit_binomial_models(analysis_df)
    binomial_coefficients.to_csv(OUTPUT_DIR / 'binomial_model_coefficients.csv', index=False)
    binomial_model_fit.to_csv(OUTPUT_DIR / 'binomial_model_fit_summary.csv', index=False)
    excel_output_path = write_excel_with_fallback(
        {
            'coefficients': binomial_coefficients,
            'fit_summary': binomial_model_fit,
        },
        OUTPUT_DIR / 'binomial_model_results.xlsx',
    )
    plot_gini_bootstrap(
        analysis_df=analysis_df,
        output_path=OUTPUT_DIR / 'gini_bootstrap_histogram.png',
        summary_path=OUTPUT_DIR / 'gini_bootstrap_summary.csv',
    )
    print(f'Binomial Excel export: {excel_output_path}')
    return analysis_df


#%%
if __name__ == '__main__':
    analysis_df = run_analysis()
    print(analysis_df[PLOT_COLUMNS].describe().round(3))

#%%
