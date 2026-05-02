# Visualization Findings Report

## Scope

This report summarizes the additional analyses added to `visualization/data_visualization_tsne.ipynb`. The new findings are aligned with the original notebook goal: using dimensionality reduction to inspect whether the student dropout dataset shows visible local class structure.

## Notebook Alignment

The original notebook structure was preserved and extended rather than replaced:

- Original sections 1-2 still load packages and processed data.
- A new exploratory findings section was inserted after data loading.
- Original t-SNE feature preparation remains before any t-SNE computation.
- A PCA baseline was inserted before the main t-SNE projection.
- Original 2D t-SNE, 3D t-SNE, perplexity sensitivity, auxiliary metrics, and discussion remain in order.

The added analyses support the original interpretation rule: t-SNE should be treated as a local-neighborhood visualization, not as proof of global class separation.

## Added Analyses

### 1. Target Class Distribution

The processed dataset has 4,424 rows. The target classes are moderately imbalanced:

| Target | Count | Percent |
|---|---:|---:|
| Graduate | 2,209 | 49.93% |
| Dropout | 1,421 | 32.12% |
| Enrolled | 794 | 17.95% |

Interpretation: `Graduate` is the largest class and `Enrolled` is the smallest. In scatter plots, visual density can partly reflect this imbalance, so class frequency should be considered before interpreting cluster size.

### 2. Feature-Target Relationship

Mean academic performance by class shows a clear monotonic pattern:

| Target | 1st sem approved | 2nd sem approved | 1st sem grade | 2nd sem grade | Age at enrollment |
|---|---:|---:|---:|---:|---:|
| Dropout | 2.55 | 1.94 | 7.26 | 5.90 | 26.07 |
| Enrolled | 4.32 | 4.06 | 11.13 | 11.12 | 22.37 |
| Graduate | 6.23 | 6.18 | 12.64 | 12.70 | 21.78 |

Interpretation: academic progression variables strongly distinguish `Graduate` from `Dropout`. Dropout students have lower approved course counts and lower semester grades, while graduate students show consistently higher academic performance.

### 3. Financial and Administrative Indicators

Class-level rates for selected binary features:

| Target | Tuition fees up to date | Debtor | Scholarship holder | Gender |
|---|---:|---:|---:|---:|
| Dropout | 0.678 | 0.220 | 0.094 | 0.493 |
| Enrolled | 0.947 | 0.113 | 0.164 | 0.387 |
| Graduate | 0.987 | 0.046 | 0.378 | 0.248 |

Interpretation: students who graduate are much more likely to have tuition fees up to date and less likely to be debtors. Scholarship holder status is also more common among graduates.

### 4. Correlation-Based Screening

The top features by absolute Pearson correlation with encoded target are:

| Feature | Correlation with Target |
|---|---:|
| Curricular units 2nd sem (approved) | 0.6242 |
| Curricular units 2nd sem (grade) | 0.5668 |
| Curricular units 1st sem (approved) | 0.5291 |
| Curricular units 1st sem (grade) | 0.4852 |
| Tuition fees up to date | 0.4098 |
| Scholarship holder | 0.2976 |
| Age at enrollment | -0.2434 |
| Debtor | -0.2410 |
| Gender | -0.2293 |
| Application mode | -0.2217 |

Interpretation: the strongest target-related signals are academic performance and progression variables. However, this is only exploratory because the target is encoded as `0=Dropout`, `1=Enrolled`, and `2=Graduate`; correlation does not imply causality.

### 5. PCA Baseline

The PCA baseline was added as a linear comparison before t-SNE:

| Metric | Value |
|---|---:|
| PC1 explained variance | 0.2340 |
| PC2 explained variance | 0.0974 |
| Total explained variance | 0.3314 |
| PCA silhouette by target | 0.0032 |
| PCA Davies-Bouldin by target | 2.9621 |

Interpretation: the first two principal components explain about one third of the prepared feature-space variance, but the near-zero silhouette score suggests weak class separation in a simple linear 2D projection. This supports using t-SNE for local-neighborhood exploration while still being cautious about interpreting visible clusters as clean class boundaries.

## Overall Finding

The dataset has interpretable class-related structure, especially in semester grades, approved curricular units, tuition status, debtor status, scholarship status, and age at enrollment. PCA suggests that the three target classes are not cleanly separable in a simple linear 2D view. Therefore, the t-SNE analysis is useful as a local visualization tool, but it should be paired with the exploratory findings and quality metrics instead of being interpreted alone.
