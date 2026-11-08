from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')
    men = df[df['Gender'] == 'Male']
    women = df[df['Gender'] == 'Female']
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    men_mri_correlation = {
        feat: float(men[feat].corr(men['MRI_Count'], method='pearson'))
        for feat in features
    }
    women_mri_correlation = {
        feat: float(women[feat].corr(women['MRI_Count'], method='pearson'))
        for feat in features
    }
    all_correlations = {}
    for feat in features:
        vals = [men_mri_correlation[feat], women_mri_correlation[feat]]
        vals = [abs(v) for v in vals if not pd.isna(v)]
        all_correlations[feat] = max(vals) if vals else float("-inf")
    strongest_mri_feature = max(reversed(features), key=lambda f: all_correlations[f])
    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_mri_correlation,
        men_mri_correlation=men_mri_correlation,
        strongest_mri_feature=strongest_mri_feature
    )
