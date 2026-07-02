#%%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor, plot_tree
from statsmodels.iolib.summary2 import summary_col
from statsmodels.stats.outliers_influence import variance_inflation_factor

from merge_data import merge_geodata_omi_redditi_votes_2021


OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok=True)
CORR_DIR = OUTPUT_DIR / 'correlation_and_simple_regressions'
MULTIVAR_DIR = OUTPUT_DIR / 'multivariate_regressions'
TREE_DIR = OUTPUT_DIR / 'decision_trees'
for directory in [CORR_DIR, MULTIVAR_DIR, TREE_DIR]:
    directory.mkdir(exist_ok=True)
BOOTSTRAP_RANDOM_SEED = 42
BOOTSTRAP_SAMPLES = 5000

sns.set_theme(style='white', context='notebook')

ANALYSIS_COLUMNS = {
    'Voti si': 'yes_votes',
    'Voti validi': 'valid_votes',
    'Voti coalizione - CENTRODESTRA': 'center_right_votes',
    'Voti lista totali politiche 2022': 'valid_votes_politiche_2022',
    'gini_income': 'gini_income',
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
    'gini_income',
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
    'gini_income': 'gini_income',
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

MODEL_LABELS = {
    'income': 'Income',
    'house_price': 'House price',
    'house_rent': 'House rent',
    'distance_milano': 'Distance Milan',
    'turnout_ref26': 'Turnout ref26',
    'turnout_pol22': 'Turnout pol22',
    'turnout_male22': 'Turnout male22',
    'turnout_female22': 'Turnout female22',
    'socioeconomic': 'Socioeconomic',
    'income_rent_turnout2026': 'Income + rent + turnout2026',
    'socioeconomic_turnout': 'Socioeconomic + turnout',
    'income_rent_turnout': 'Income + rent + turnout',
}

TERM_LABELS = {
    'const': 'Intercept',
    'avg_income_z': 'Average income (z)',
    'house_price_m2_z': 'House price/m2 (z)',
    'house_rent_m2_z': 'House rent/m2 (z)',
    'distance_to_milano_km_z': 'Distance to Milan (z)',
    'center_right': 'Center right vote share',
    'center_left': 'Center left vote share',
    'm5s': 'M5S vote share',
    'third_pole': 'Third pole vote share',
    'turnout_referendum_2026_z': 'Turnout ref26 (z)',
    'turnout_politiche_2022_z': 'Turnout pol22 (z)',
    'turnout_politiche_2022_male_z': 'Turnout male22 (z)',
    'turnout_politiche_2022_female_z': 'Turnout female22 (z)',
}

TREE_FEATURES = [
    'avg_income',
    'house_rent_m2',
    'turnout_politiche_2022',
    'distance_to_milano_km',
]

TREE_FEATURE_LABELS = {
    'avg_income': 'Average income',
    'house_rent_m2': 'House rent/m2',
    'turnout_politiche_2022': 'Turnout politiche 2022',
    'distance_to_milano_km': 'Distance to Milan (km)',
}


def build_analysis_df(region: str = 'Lombardia') -> pd.DataFrame:
    df = merge_geodata_omi_redditi_votes_2021(region=region).copy()
    return build_analysis_df_from_merged(df)


def build_analysis_df_from_merged(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=ANALYSIS_COLUMNS)

    keep_columns = ['Comune'] + list(ANALYSIS_COLUMNS.values())
    analysis_df = df[keep_columns].copy()
    analysis_df = analysis_df.dropna(subset=PLOT_COLUMNS)
    analysis_df = analysis_df[
        (analysis_df['valid_votes'] > 0)
        & (analysis_df['valid_votes_politiche_2022'] > 0)
    ].copy()
    return analysis_df


def add_standardized_model_columns(analysis_df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = analysis_df.copy()
    columns_to_standardize = [
        'referendum_yes',
        'gini_income',
        'avg_income',
        'house_price_m2',
        'house_rent_m2',
        'distance_to_milano_km',
        'turnout_referendum_2026',
        'turnout_politiche_2022',
        'turnout_politiche_2022_male',
        'turnout_politiche_2022_female',
    ]

    for column in columns_to_standardize:
        series = pd.to_numeric(standardized_df[column], errors='coerce')
        std = series.std()
        if pd.isna(std) or std == 0:
            standardized_df[f'{column}_z'] = np.nan
        else:
            standardized_df[f'{column}_z'] = (series - series.mean()) / std

    return standardized_df


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
    success_column: str,
    proportion_column: str,
    weight_column: str,
    predictors: list[str],
    model_name: str,
) -> tuple[sm.GLM, pd.DataFrame]:
    model_df = analysis_df[[success_column, weight_column, proportion_column] + predictors].dropna().copy()
    for column in [success_column, weight_column, proportion_column] + predictors:
        model_df[column] = pd.to_numeric(model_df[column], errors='coerce')
    model_df = model_df.dropna()
    X = sm.add_constant(model_df[predictors], has_constant='add').astype(float)
    y = model_df[proportion_column].astype(float)
    weights = model_df[weight_column].astype(float)

    result = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
        freq_weights=weights,
    ).fit()

    coef_table = result.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
    coef_table.insert(0, 'model', model_name)
    coef_table['n_municipalities'] = len(model_df)
    coef_table['weight_sum'] = model_df[weight_column].sum()
    return result, coef_table


def compute_vif_table(model_df: pd.DataFrame, predictors: list[str], model_name: str) -> pd.DataFrame:
    X = model_df[predictors].copy()
    X = X.apply(pd.to_numeric, errors='coerce').dropna().astype(float)
    vif_table = pd.DataFrame(
        {
            'model': model_name,
            'term': predictors,
            'vif': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        }
    )
    return vif_table


def fit_binomial_models(
    analysis_df: pd.DataFrame,
    success_column: str,
    proportion_column: str,
    weight_column: str,
    model_prefix: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, sm.GLM], pd.DataFrame]:
    model_specs = {
        'income': ['avg_income_z'],
        'house_price': ['house_price_m2_z'],
        'house_rent': ['house_rent_m2_z'],
        'distance_milano': ['distance_to_milano_km_z'],
        'turnout_ref26': ['turnout_referendum_2026_z'],
        'turnout_pol22': ['turnout_politiche_2022_z'],
        'turnout_male22': ['turnout_politiche_2022_male_z'],
        'turnout_female22': ['turnout_politiche_2022_female_z'],
        'socioeconomic': ['avg_income_z', 'house_price_m2_z', 'house_rent_m2_z', 'distance_to_milano_km_z'],
        'income_rent_turnout2026': [
            'avg_income_z',
            'house_rent_m2_z',
            'turnout_referendum_2026_z',
        ],
        'income_rent_turnout': [
            'avg_income_z',
            'house_rent_m2_z',
            'turnout_referendum_2026_z',
            'turnout_politiche_2022_z',
        ],
        'socioeconomic_turnout': [
            'avg_income_z',
            'house_price_m2_z',
            'house_rent_m2_z',
            'distance_to_milano_km_z',
            'turnout_referendum_2026_z',
            'turnout_politiche_2022_z',
            'turnout_politiche_2022_male_z',
            'turnout_politiche_2022_female_z',
        ],
    }

    coef_tables = []
    fit_summaries = []
    fitted_models = {}
    vif_tables = []

    for model_name, predictors in model_specs.items():
        full_model_name = f'{model_prefix}_{model_name}'
        model_df = analysis_df[[success_column, weight_column, proportion_column] + predictors].dropna().copy()
        result, coef_table = fit_binomial_model(
            analysis_df,
            success_column=success_column,
            proportion_column=proportion_column,
            weight_column=weight_column,
            predictors=predictors,
            model_name=full_model_name,
        )
        fitted_models[full_model_name] = result
        coef_tables.append(coef_table)
        fit_summaries.append(
            {
                'model': full_model_name,
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
        if len(predictors) > 1:
            vif_tables.append(compute_vif_table(model_df, predictors, full_model_name))

    vif_df = pd.concat(vif_tables, ignore_index=True) if vif_tables else pd.DataFrame(columns=['model', 'term', 'vif'])
    return pd.concat(coef_tables, ignore_index=True), pd.DataFrame(fit_summaries), fitted_models, vif_df


def _summary_to_dataframe(summary) -> pd.DataFrame:
    table = summary.tables[0]
    if isinstance(table, pd.DataFrame):
        return table.reset_index().rename(columns={'index': 'term'})

    rows = table.data
    header = rows[0]
    body = rows[1:]
    return pd.DataFrame(body, columns=header)


def export_multimodel_summary(
    fitted_models: dict[str, sm.GLM],
    model_names: list[str],
    prefix: str,
    dependent_variable_label: str,
    weight_label: str,
    vif_table: pd.DataFrame | None = None,
) -> None:
    selected_models = [fitted_models[name] for name in model_names]
    selected_labels = []
    for name in model_names:
        short_name = name
        for candidate_prefix in ['referendum_yes_', 'center_right_']:
            if short_name.startswith(candidate_prefix):
                short_name = short_name.removeprefix(candidate_prefix)
                break
        selected_labels.append(MODEL_LABELS[short_name])

    summary = summary_col(
        selected_models,
        stars=True,
        float_format='%0.4f',
        model_names=selected_labels,
        info_dict={
            'N': lambda x: f"{int(x.nobs)}",
            'AIC': lambda x: f"{x.aic:.1f}",
            'Pseudo R2': lambda x: f"{x.pseudo_rsquared(kind='mcf'):.4f}",
        },
    )

    summary_df = _summary_to_dataframe(summary)
    first_col = summary_df.columns[0]
    summary_df[first_col] = summary_df[first_col].replace(TERM_LABELS)

    header_lines = [
        f'Dependent variable: {dependent_variable_label}',
        'Model family: Binomial GLM with logit link',
        f'Weights: {weight_label}',
        'Coefficients are on the log-odds scale.',
        '',
    ]
    header_text = '\n'.join(header_lines)

    html_path = MULTIVAR_DIR / f'{prefix}.html'
    png_path = MULTIVAR_DIR / f'{prefix}.png'
    csv_path = MULTIVAR_DIR / f'{prefix}.csv'
    txt_path = MULTIVAR_DIR / f'{prefix}.txt'

    vif_text = ''
    vif_html = ''
    if vif_table is not None and not vif_table.empty:
        vif_subset = vif_table[vif_table['model'].isin(model_names)].copy()
        vif_subset['term'] = vif_subset['term'].replace(TERM_LABELS)
        if not vif_subset.empty:
            vif_text = '\n\nVIF by model\n' + vif_subset.to_string(index=False)
            vif_html = '<h3>VIF by model</h3>' + vif_subset.to_html(index=False)

    txt_path.write_text(header_text + summary.as_text() + vif_text, encoding='utf-8')
    html_path.write_text(
        (
            '<html><head><style>'
            'body { font-family: Arial, sans-serif; margin: 24px; }'
            'table { border-collapse: separate; border-spacing: 12px 4px; }'
            'th, td { padding: 6px 14px; min-width: 120px; text-align: center; }'
            'th:first-child, td:first-child { min-width: 220px; text-align: left; }'
            '</style></head><body>'
            f'<p><strong>Dependent variable:</strong> {dependent_variable_label}<br>'
            '<strong>Model family:</strong> Binomial GLM with logit link<br>'
            f'<strong>Weights:</strong> {weight_label}<br>'
            '<strong>Scale:</strong> coefficients in log-odds</p>'
            f'{summary.as_html()}'
            f'{vif_html}'
            '</body></html>'
        ),
        encoding='utf-8',
    )
    summary_df.to_csv(csv_path, index=False)

    fig_height = max(4, 0.42 * (len(summary_df) + 2))
    fig_width = max(10, 2.3 * len(summary_df.columns))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title(
        prefix.replace('_', ' ').title()
        + f'\nDependent variable: {dependent_variable_label} | Binomial GLM (logit) | Weights: {weight_label}',
        pad=18,
    )
    plt.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def fit_decision_tree(
    analysis_df: pd.DataFrame,
    target_column: str,
    target_label: str,
    prefix: str,
    max_depth: int = 3,
    min_samples_leaf: int = 40,
) -> pd.DataFrame:
    tree_df = analysis_df[TREE_FEATURES + [target_column]].copy()
    tree_df[TREE_FEATURES + [target_column]] = tree_df[TREE_FEATURES + [target_column]].apply(
        pd.to_numeric, errors='coerce'
    )
    tree_df = tree_df.dropna()

    X = tree_df[TREE_FEATURES]
    y = tree_df[target_column]

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    model.fit(X, y)
    tree_r2 = model.score(X, y)

    importance_df = pd.DataFrame(
        {
            'target': target_column,
            'feature': TREE_FEATURES,
            'feature_label': [TREE_FEATURE_LABELS[col] for col in TREE_FEATURES],
            'importance': model.feature_importances_,
            'tree_r2': tree_r2,
        }
    ).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        model,
        feature_names=[TREE_FEATURE_LABELS[col] for col in TREE_FEATURES],
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title(f'Decision Tree for {target_label}')
    plt.tight_layout()
    fig.savefig(TREE_DIR / f'{prefix}_decision_tree.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    importance_df.to_csv(TREE_DIR / f'{prefix}_decision_tree_importance.csv', index=False)
    (TREE_DIR / f'{prefix}_decision_tree_importance.txt').write_text(
        f'Target: {target_label}\n'
        f'Features: {", ".join(TREE_FEATURES)}\n\n'
        f'R2: {tree_r2:.4f}\n\n'
        + importance_df[['feature_label', 'importance']].to_string(index=False),
        encoding='utf-8',
    )
    return importance_df


def fit_gini_regression(analysis_df: pd.DataFrame) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    predictors = ['referendum_yes_z', 'avg_income_z', 'house_rent_m2_z', 'turnout_politiche_2022_z']
    model_df = analysis_df[['gini_income'] + predictors].copy()
    model_df = model_df.apply(pd.to_numeric, errors='coerce').dropna()

    X = sm.add_constant(model_df[predictors], has_constant='add').astype(float)
    y = model_df['gini_income'].astype(float)
    result = sm.OLS(y, X).fit()

    coef_table = result.summary2().tables[1].reset_index().rename(columns={'index': 'term'})
    coef_table['term'] = coef_table['term'].replace(
        {
            'const': 'Intercept',
            'referendum_yes_z': 'Referendum yes (z)',
            'avg_income_z': 'Average income (z)',
            'house_rent_m2_z': 'House rent/m2 (z)',
            'turnout_politiche_2022_z': 'Turnout politiche 2022 (z)',
        }
    )
    return result, coef_table


def export_gini_regression_summary(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    coef_table: pd.DataFrame,
) -> None:
    summary_text = (
        'Dependent variable: gini_income (municipal Gini from income brackets)\n'
        'Model family: OLS\n'
        'Predictors: referendum_yes_z, avg_income_z, house_rent_m2_z, turnout_politiche_2022_z\n\n'
        + result.summary().as_text()
    )
    (MULTIVAR_DIR / 'gini_regression_summary.txt').write_text(summary_text, encoding='utf-8')
    coef_table.to_csv(MULTIVAR_DIR / 'gini_regression_coefficients.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 3 + 0.45 * len(coef_table)))
    ax.axis('off')
    table = ax.table(
        cellText=coef_table.round(4).values,
        colLabels=coef_table.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)
    ax.set_title(
        f'Gini regression | R2 = {result.rsquared:.4f} | Adj. R2 = {result.rsquared_adj:.4f}',
        pad=18,
    )
    plt.tight_layout()
    fig.savefig(MULTIVAR_DIR / 'gini_regression_summary.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_gini_vs_yes_map(merged_gdf: pd.DataFrame) -> None:
    plot_gdf = merged_gdf.rename(columns=ANALYSIS_COLUMNS).copy()
    plot_gdf = plot_gdf.to_crs(epsg=4326)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_gdf.plot(
        column='gini_income',
        cmap='magma',
        legend=True,
        linewidth=0.15,
        edgecolor='white',
        missing_kwds={'color': '#f2f2f2'},
        ax=axes[0],
    )
    axes[0].set_title('Municipal Gini coefficient')
    axes[0].set_axis_off()

    plot_gdf.plot(
        column='referendum_yes',
        cmap='viridis',
        legend=True,
        linewidth=0.15,
        edgecolor='white',
        missing_kwds={'color': '#f2f2f2'},
        ax=axes[1],
    )
    axes[1].set_title('Referendum yes vote share')
    axes[1].set_axis_off()

    plt.tight_layout()
    fig.savefig(CORR_DIR / 'gini_vs_referendum_yes_map.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_analysis(region: str = 'Lombardia') -> pd.DataFrame:
    merged_gdf = merge_geodata_omi_redditi_votes_2021(region=region).copy()
    merged_gdf.drop(columns=['geometry']).to_excel('data/dados.xlsx', index=False)
    analysis_df = build_analysis_df_from_merged(merged_gdf)
    model_df = add_standardized_model_columns(analysis_df)
    pearson_corr = plot_correlation_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=CORR_DIR / 'correlation_triangle.png',
        method='pearson',
        title='Triangular Correlation Matrix - Pearson',
        cbar_label='Pearson r',
    )
    spearman_corr = plot_correlation_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=CORR_DIR / 'spearman_correlation_triangle.png',
        method='spearman',
        title='Triangular Correlation Matrix - Spearman',
        cbar_label='Spearman rho',
    )
    plot_regression_triangle(
        analysis_df=analysis_df,
        columns=PLOT_COLUMNS,
        output_path=CORR_DIR / 'regression_triangle.png',
    )

    analysis_df.to_excel(OUTPUT_DIR / 'analysis_dataset.xlsx', index=False)
    top_pearson = summarize_correlations(pearson_corr, value_name='pearson_r')
    top_pearson.to_csv(CORR_DIR / 'top_correlations.csv', index=False)
    pearson_corr.to_csv(CORR_DIR / 'pearson_correlation_matrix.csv')
    top_spearman = summarize_correlations(spearman_corr, value_name='spearman_rho')
    top_spearman.to_csv(CORR_DIR / 'top_spearman_correlations.csv', index=False)
    spearman_corr.to_csv(CORR_DIR / 'spearman_correlation_matrix.csv')
    model_df['center_right'] = pd.to_numeric(model_df['center_right'], errors='coerce')
    model_df['center_right_votes'] = pd.to_numeric(model_df['center_right_votes'], errors='coerce')
    model_df['valid_votes_politiche_2022'] = pd.to_numeric(model_df['valid_votes_politiche_2022'], errors='coerce')

    referendum_coefficients, referendum_model_fit, referendum_models, referendum_vif = fit_binomial_models(
        model_df,
        success_column='yes_votes',
        proportion_column='referendum_yes',
        weight_column='valid_votes',
        model_prefix='referendum_yes',
    )
    center_right_coefficients, center_right_model_fit, center_right_models, center_right_vif = fit_binomial_models(
        model_df,
        success_column='center_right_votes',
        proportion_column='center_right',
        weight_column='valid_votes_politiche_2022',
        model_prefix='center_right',
    )

    binomial_coefficients = pd.concat([referendum_coefficients, center_right_coefficients], ignore_index=True)
    binomial_model_fit = pd.concat([referendum_model_fit, center_right_model_fit], ignore_index=True)
    fitted_models = {**referendum_models, **center_right_models}
    binomial_vif = pd.concat([referendum_vif, center_right_vif], ignore_index=True)

    binomial_coefficients.to_csv(MULTIVAR_DIR / 'binomial_model_coefficients.csv', index=False)
    binomial_model_fit.to_csv(MULTIVAR_DIR / 'binomial_model_fit_summary.csv', index=False)
    binomial_vif.to_csv(MULTIVAR_DIR / 'binomial_model_vif.csv', index=False)
    excel_output_path = write_excel_with_fallback(
        {
            'coefficients': binomial_coefficients,
            'fit_summary': binomial_model_fit,
            'vif': binomial_vif,
        },
        MULTIVAR_DIR / 'binomial_model_results.xlsx',
    )
    export_multimodel_summary(
        fitted_models=fitted_models,
        model_names=[
            'center_right_income_rent_turnout2026',
            'center_right_income_rent_turnout',
            'center_right_socioeconomic',
            'center_right_socioeconomic_turnout',
        ],
        prefix='center_right_models_multivariate_summary',
        dependent_variable_label='center_right (share of center-right list votes in politiche 2022)',
        weight_label='valid_votes_politiche_2022 (total list votes in politiche 2022)',
        vif_table=binomial_vif,
    )
    export_multimodel_summary(
        fitted_models=fitted_models,
        model_names=[
            'referendum_yes_income_rent_turnout2026',
            'referendum_yes_income_rent_turnout',
            'referendum_yes_socioeconomic',
            'referendum_yes_socioeconomic_turnout',
        ],
        prefix='referendum_yes_models_multivariate_summary',
        dependent_variable_label='referendum_yes (share of Yes votes in referendum 2026)',
        weight_label='valid_votes (number of valid referendum votes)',
        vif_table=binomial_vif,
    )
    referendum_tree_importance = fit_decision_tree(
        analysis_df=analysis_df,
        target_column='referendum_yes',
        target_label='referendum_yes',
        prefix='referendum_yes',
    )
    center_right_tree_importance = fit_decision_tree(
        analysis_df=analysis_df,
        target_column='center_right',
        target_label='center_right',
        prefix='center_right',
    )
    gini_tree_importance = fit_decision_tree(
        analysis_df=analysis_df,
        target_column='gini_income',
        target_label='gini_income',
        prefix='gini_income',
    )
    pd.concat([referendum_tree_importance, center_right_tree_importance, gini_tree_importance], ignore_index=True).to_csv(
        TREE_DIR / 'decision_tree_variable_importance.csv',
        index=False,
    )
    gini_result, gini_coef_table = fit_gini_regression(model_df)
    export_gini_regression_summary(gini_result, gini_coef_table)
    plot_gini_vs_yes_map(merged_gdf)
    plot_gini_bootstrap(
        analysis_df=analysis_df,
        output_path=CORR_DIR / 'gini_bootstrap_histogram.png',
        summary_path=CORR_DIR / 'gini_bootstrap_summary.csv',
    )
    print(f'Binomial Excel export: {excel_output_path}')
    return analysis_df


#%%
if __name__ == '__main__':
    analysis_df = run_analysis()
    print(analysis_df[PLOT_COLUMNS].describe().round(3))

#%%
