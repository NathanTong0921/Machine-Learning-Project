# Noise Feature Diagnostics

Train rows: 3096, test rows: 1328, features: 36.

## Top heuristic SVM noise candidates

| feature                                        | kind        | group                 |   harmful_for_svm_score |   low_signal_score |   svm_noise_pressure |   mutual_info |   association_metric |   univariate_macro_f1 |   univariate_enrolled_recall |   shift_metric |   train_unique |   rare_train_mass |   unseen_test_mass |   train_top_freq |
|:-----------------------------------------------|:------------|:----------------------|------------------------:|-------------------:|---------------------:|--------------:|---------------------:|----------------------:|-----------------------------:|---------------:|---------------:|------------------:|-------------------:|-----------------:|
| Curricular units 2nd sem (credited)            | numeric     | second_semester_macro |                  0.8218 |             0.9429 |               0.6243 |        0.0000 |               0.0024 |                0.1689 |                       0.8921 |         0.0031 |             19 |            0.0000 |             0.0000 |           0.8795 |
| Curricular units 1st sem (credited)            | numeric     | first_semester        |                  0.8208 |             0.9557 |               0.6007 |        0.0000 |               0.0021 |                0.1693 |                       0.8723 |         0.0056 |             21 |            0.0000 |             0.0000 |           0.8679 |
| Unemployment rate                              | numeric     | second_semester_macro |                  0.8206 |             0.9029 |               0.6864 |        0.0000 |               0.0023 |                0.2839 |                       0.5737 |         0.0114 |             10 |            0.0000 |             0.0000 |           0.1256 |
| Curricular units 2nd sem (without evaluations) | numeric     | second_semester_macro |                  0.7791 |             0.8871 |               0.6029 |        0.0000 |               0.0071 |                0.2700 |                       0.0000 |         0.0069 |             10 |            0.0000 |             0.0000 |           0.9373 |
| International                                  | categorical | enrollment_profile    |                  0.7740 |             0.8500 |               0.6500 |        0.0001 |               0.0150 |                0.2190 |                       0.1924 |         0.0054 |              2 |            0.0000 |             0.0000 |           0.9767 |
| GDP                                            | numeric     | second_semester_macro |                  0.7507 |             0.7743 |               0.7121 |        0.0197 |               0.0018 |                0.3341 |                       0.1097 |         0.0086 |             10 |            0.0000 |             0.0000 |           0.1256 |
| Curricular units 1st sem (without evaluations) | numeric     | first_semester        |                  0.7441 |             0.8386 |               0.5900 |        0.0039 |               0.0054 |                0.2725 |                       0.0198 |         0.0079 |             11 |            0.0000 |             0.0000 |           0.9315 |
| Educational special needs                      | categorical | family_finance        |                  0.7258 |             0.7657 |               0.6607 |        0.0002 |               0.0182 |                0.2909 |                       0.0090 |         0.0014 |              2 |            0.0000 |             0.0000 |           0.9880 |
| Inflation rate                                 | numeric     | second_semester_macro |                  0.6823 |             0.7757 |               0.5300 |        0.0230 |               0.0011 |                0.3220 |                       0.2734 |         0.0262 |              9 |            0.0000 |             0.0000 |           0.2025 |
| Displaced                                      | categorical | family_finance        |                  0.6638 |             0.6157 |               0.7421 |        0.0063 |               0.1127 |                0.3365 |                       0.0000 |         0.0003 |              2 |            0.0000 |             0.0000 |           0.5484 |
| Application order                              | numeric     | enrollment_profile    |                  0.6532 |             0.6971 |               0.5814 |        0.0145 |               0.0116 |                0.3458 |                       0.1349 |         0.0133 |              8 |            0.0000 |             0.0000 |           0.6831 |
| Daytime/evening attendance                     | categorical | enrollment_profile    |                  0.6317 |             0.6586 |               0.5879 |        0.0041 |               0.0924 |                0.3275 |                       0.3543 |         0.0118 |              2 |            0.0000 |             0.0000 |           0.8944 |

## Largest train/test shifts

| feature                                | kind        |   shift_metric | shift_name      |   max_frequency_delta |   unseen_test_mass |   unseen_test_level_count |   ks_pvalue |   mean_std_diff |
|:---------------------------------------|:------------|---------------:|:----------------|----------------------:|-------------------:|--------------------------:|------------:|----------------:|
| Father's occupation                    | categorical |         0.0486 | total_variation |                0.0238 |             0.0045 |                    6.0000 |    nan      |        nan      |
| Mother's qualification                 | categorical |         0.0450 | total_variation |                0.0160 |             0.0008 |                    1.0000 |    nan      |        nan      |
| Course                                 | categorical |         0.0438 | total_variation |                0.0141 |             0.0000 |                    0.0000 |    nan      |        nan      |
| Mother's occupation                    | categorical |         0.0429 | total_variation |                0.0175 |             0.0008 |                    1.0000 |    nan      |        nan      |
| Application mode                       | categorical |         0.0343 | total_variation |                0.0244 |             0.0000 |                    0.0000 |    nan      |        nan      |
| Age at enrollment                      | numeric     |         0.0342 | ks_statistic    |              nan      |           nan      |                  nan      |      0.2215 |          0.0213 |
| Father's qualification                 | categorical |         0.0328 | total_variation |                0.0152 |             0.0015 |                    2.0000 |    nan      |        nan      |
| Previous qualification (grade)         | numeric     |         0.0316 | ks_statistic    |              nan      |           nan      |                  nan      |      0.3046 |          0.0378 |
| Inflation rate                         | numeric     |         0.0262 | ks_statistic    |              nan      |           nan      |                  nan      |      0.5373 |          0.0458 |
| Previous qualification                 | categorical |         0.0218 | total_variation |                0.0110 |             0.0000 |                    0.0000 |    nan      |        nan      |
| Curricular units 2nd sem (evaluations) | numeric     |         0.0209 | ks_statistic    |              nan      |           nan      |                  nan      |      0.7994 |          0.0404 |
| Admission grade                        | numeric     |         0.0193 | ks_statistic    |              nan      |           nan      |                  nan      |      0.8719 |          0.0144 |

## Categorical rare-category pressure

| feature                    |   rare_threshold |   rare_level_count |   rare_train_mass |   singleton_level_count |   unseen_test_level_count |   unseen_test_mass |
|:---------------------------|-----------------:|-------------------:|------------------:|------------------------:|--------------------------:|-------------------:|
| Father's qualification     |           0.0100 |                 25 |            0.0430 |                      11 |                         2 |             0.0015 |
| Previous qualification     |           0.0100 |                 12 |            0.0368 |                       3 |                         0 |             0.0000 |
| Mother's occupation        |           0.0100 |                 21 |            0.0313 |                       7 |                         1 |             0.0008 |
| Application mode           |           0.0100 |                  9 |            0.0313 |                       3 |                         0 |             0.0000 |
| Nacionality                |           0.0100 |                 16 |            0.0233 |                       6 |                         4 |             0.0038 |
| Father's occupation        |           0.0100 |                 28 |            0.0229 |                      13 |                         6 |             0.0045 |
| Mother's qualification     |           0.0100 |                 19 |            0.0210 |                       8 |                         1 |             0.0008 |
| Marital status             |           0.0100 |                  3 |            0.0055 |                       1 |                         0 |             0.0000 |
| Course                     |           0.0100 |                  1 |            0.0029 |                       0 |                         0 |             0.0000 |
| Daytime/evening attendance |           0.0100 |                  0 |            0.0000 |                       0 |                         0 |             0.0000 |
| Displaced                  |           0.0100 |                  0 |            0.0000 |                       0 |                         0 |             0.0000 |
| Educational special needs  |           0.0100 |                  0 |            0.0000 |                       0 |                         0 |             0.0000 |

## Lowest univariate CV signal

| feature                                        |   univariate_macro_f1 |   univariate_enrolled_recall |
|:-----------------------------------------------|----------------------:|-----------------------------:|
| Curricular units 2nd sem (credited)            |                0.1689 |                       0.8921 |
| Curricular units 1st sem (credited)            |                0.1693 |                       0.8723 |
| International                                  |                0.2190 |                       0.1924 |
| Nacionality                                    |                0.2212 |                       0.1888 |
| Marital status                                 |                0.2660 |                       0.7302 |
| Curricular units 2nd sem (without evaluations) |                0.2700 |                       0.0000 |
| Curricular units 1st sem (without evaluations) |                0.2725 |                       0.0198 |
| Unemployment rate                              |                0.2839 |                       0.5737 |
| Educational special needs                      |                0.2909 |                       0.0090 |
| Inflation rate                                 |                0.3220 |                       0.2734 |
| Daytime/evening attendance                     |                0.3275 |                       0.3543 |
| GDP                                            |                0.3341 |                       0.1097 |
