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

### 4. Academic Performance Block Redundancy

Before compressing academic features, we first checked whether the academic performance block contains repeated information. The strongest inter-feature correlations are between matching first- and second-semester measures:

| Feature pair | Absolute correlation |
|---|---:|
| 1st/2nd sem credited | 0.9448 |
| 1st/2nd sem enrolled | 0.9426 |
| 1st/2nd sem approved | 0.9040 |
| 1st/2nd sem grade | 0.8372 |

Interpretation: these high correlations show that the semester-specific academic variables form a redundant feature block. This motivates two compression strategies: an average-based summary for readable exploratory screening, and PCA-based components for modeling-oriented compression.

### 5. Compact Average-Based Feature Screening

The first compression method is an interpretable average-based summary. Paired first- and second-semester curricular-unit variables are averaged into composite features before ranking by correlation with the encoded target:

| Feature | Correlation with Target |
|---|---:|
| Average curricular units approved | 0.5904 |
| Average curricular units grade | 0.5504 |
| Tuition fees up to date | 0.4098 |
| Scholarship holder | 0.2976 |
| Age at enrollment | -0.2434 |
| Debtor | -0.2410 |
| Gender | -0.2293 |
| Application mode | -0.2217 |
| Average curricular units enrolled | 0.1677 |
| Admission grade | 0.1209 |
| Displaced | 0.1140 |
| Previous qualification (grade) | 0.1038 |
| Average curricular units without evaluations | -0.0920 |
| Marital status | -0.0898 |
| Application order | 0.0898 |

Interpretation: academic progression remains the strongest target-related signal after combining semester pairs, especially average approved units and average semester grades. Financial and administrative features such as tuition-fee status, scholarship status, debtor status, age, and gender remain important secondary signals. However, this is only exploratory because the target is encoded as `0=Dropout`, `1=Enrolled`, and `2=Graduate`; correlation does not imply causality.

### 6. PCA-Based Academic Feature Compression

The second compression method applies PCA to the standardized academic block. This creates orthogonal components that can preserve multiple directions of academic-performance variation:

| Component set | Explained variance |
|---|---:|
| Academic_PC1 | 51.38% |
| Academic_PC1 + Academic_PC2 | 68.36% |
| Academic_PC1 + Academic_PC2 + Academic_PC3 | 80.63% |
| Academic_PC1 to Academic_PC6 | 94.31% |

The academic PCs also show target-related signal across multiple directions:

| Component | Correlation with Target |
|---|---:|
| Academic_PC1 | 0.3458 |
| Academic_PC2 | 0.4625 |
| Academic_PC3 | 0.1212 |
| Academic_PC4 | -0.3525 |

Interpretation: averaged academic features are easier to explain, while PCA is a stronger candidate for downstream modeling because it handles multicollinearity and preserves several independent directions of academic-performance variation. If PCA features are used in supervised models, PCA should be fitted on the training split and then applied to validation/test data to avoid leakage.

### 7. PCA Baseline

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
