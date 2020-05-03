Feature Engineering
---
#### Motivation
Having collected and aggregated various datasets including on-court metrics, player measurements, positional estimates, and salary information we used those existing features to engineer new metrics that might be more predictive in our projection model. Our final model input dataframe contains six 'feature subsets,' each containing a different set of engineered features. Our intent is to use these feature subsets as a parameter in the model to gridsearch over in hopes of selecting the optimal inputs.

#### Raw Data
The majority of our data was scraped from Basketball-Reference and ESPN. Additional information on the tables we acquired along with our scraping scripts can be found in the  `/data_scraping` folder within this repo. Broadly, our data sources can be categorized into the following areas.

1. Basketball-Reference
    - Per-100 Possession
    - Advanced
    - Totals
    - Salary
    - Measurements
    - Positional Estimates
2.  ESPN
    - Advanced

#### Imputation
When joined together our data sources have a few systematic and random cases of missing data. For instance, in most shooting percentage metrics if a player has not made an attempt our data reflects a null value instead of a zero. These simple systematic nulls were imputed with zeroes for all observations. In other non-shooting metrics we observed random null values, which we handled by imputing the mean of a player's season and advance position cluster (guard, wing, big) for each metric. For instance, if a guard had a null value in OREB% during his 2018-2019 season we imputed the mean OREB% for guards in 2018-2019.

#### Three-Season Weighted Average
The bulk of the feature engineering work involved transforming all single-season metrics into a three-season weighted average equivalent. This may result in more predictive features as the three-season weighted average will account for large single-season fluctuations that may not truly represent a player's ability. For instance, if a player shoots 40% in each of his first two seasons in the league but shoots an abysmal 20% in year three, all on the same number of attempts, we may be better off feeding his three-season weighted average of 36% into the model as that may be more representative of his true shooting ability.

If a player has been in the league for less than three seasons we either calculate the two-season moving average or single-season metric.

#### Feature Subsets
Our final model input contains six 'feature subsets,' listed below, each containing a different set of engineered features. Each subset contains a mix of on-court metrics, player measurements, positional estimates, and salary information. Our intent is to use these feature subsets as a parameter in the model to gridsearch over in hopes of selecting the optimal inputs.


1. Box-Score: Single-Season Per-100 Possession and Advanced Metrics
2. Three-Season Weighted-Average Box-Score: Three-season weighted averages for Per-100 Possession and Advanced Metrics
3. League Percentiles: Single-season percentile of a player's performance in a given metric compared to the entire league
4. Three-Season Weighted-Average League Percentiles: Three-season weighted average percentile of a player's performance in a given metric compared to the entire league
5. Position Percentiles: Single-season percentile of a player's performance in a given metric compared to the player's advanced cluster position (Guard, Wing, Big)
6. Three-Season Weighted-Average Position Percentiles: Three-season weighted average percentile of a player's performance in a given metric compared to the player's advanced cluster position (Guard, Wing, Big)

The final model input dataframe can be found in `/feature_selection/featurized_inputs/complete_feature_matrix.csv'`.
