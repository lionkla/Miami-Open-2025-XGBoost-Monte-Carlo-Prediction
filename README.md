# Miami-Open-2025-XGBoost-Monte-Carlo-Prediction
This repository contains a tennis tournament outcome prediction system developed as part of a university seminar on Monte Carlo Simulation at the University of Augsburg. The project combines two core methods — an XGBoost-based feature selection model and a Monte Carlo simulation — to estimate win probabilities for ATP tennis matches.

# XGBoost Model
The XGBoost model processes historical ATP match data to learn which player-specific features are most predictive of match outcomes. It builds an ensemble of decision trees iteratively, where each new tree corrects the residual errors of the previous one, ultimately producing a trained classifier that outputs a point-win probability for a given player in a match. This probability serves as the core input parameter for the subsequent simulation. The model is trained on data from the Jeff Sackmann ATP dataset.

# Monte Carlo Simulation
The Monte Carlo simulation uses the point-win probability estimated by XGBoost to simulate full tennis matches repeatedly — typically thousands of runs — following the official ATP scoring structure (points → games → sets → match). Because the outcome of a tennis match is not analytically solvable from a simple win probability, the simulation replaces the closed-form solution with a probabilistic approximation: across a large number of independent match simulations, the fraction of runs won by each player converges to a stable win probability estimate. The results are then compared against real tournament outcomes and bookmaker odds to validate the model's predictive quality.

# Tennis Tournament Prediction with XGBoost and Monte Carlo Simulation

This repository contains a workflow for tennis tournament prediction that combines a data-driven XGBoost model with a Monte Carlo simulation. The seminar material describes the project pipeline as **data preparation -> XGBoost model -> simulation -> result evaluation** and frames the simulation as a way to approximate outcomes for complex systems through repeated random sampling.[cite:1] The supporting study also explains that XGBoost is used to identify and learn influential features from match data, while Monte Carlo methods simulate stochastic developments when a direct analytical solution is not practical.[cite:6]

## Project Purpose

The goal of the repository is to estimate match and tournament outcomes from historical tennis data. In this setup, the machine learning component learns predictive structure from previous matches, and the simulation component propagates these learned probabilities through a tournament tree over many repeated runs.[cite:1][cite:6]

## Workflow Overview

The repository is organized around four conceptual steps:[cite:1]

1. **Data preparation**: historical match data are cleaned, transformed, and converted into a feature table suitable for model training.[cite:1]
2. **Model training**: XGBoost is trained on player and match features to estimate outcome-related probabilities or scores.[cite:6]
3. **Tournament simulation**: the trained model output is used inside a Monte Carlo simulation that repeatedly samples match outcomes across the tournament bracket.[cite:1]
4. **Result generation**: the simulation aggregates repeated runs into stable estimates such as win probabilities, likely finalists, and expected tournament paths.[cite:1]

## Training on New Data

To train the model on a new dataset, the new data should follow the same logical structure as the historical ATP source used in the seminar example.[cite:1] Each row should represent one completed match, and the dataset should contain enough information to derive player-level, match-level, and context-level features that can be used consistently during both training and inference.[cite:1][cite:6]

A practical training process can follow these steps:

1. Replace or extend the historical dataset with a new file in CSV format.
2. Keep one row per match.
3. Ensure that player identifiers are consistent across seasons and tournaments.
4. Recompute all engineered features before model fitting.
5. Split the data into training and validation subsets.
6. Train XGBoost on the training subset.
7. Store the fitted model so that the tournament simulation can reuse it for future predictions.

## Expected Data Format

The original seminar material references ATP match data as the input basis for the custom implementation.[cite:1] The research paper further states that feature-based modeling in tennis relies on match indicators such as serve-related and scoring-related variables, and that XGBoost can rank the relative importance of such features.[cite:6]

For practical repository use, the training file should be a rectangular table such as CSV with the following structure:

| Column group | Purpose |
|---|---|
| `tournament_id`, `tournament_name`, `surface`, `round`, `date` | Tournament context and match grouping. |
| `player_1_id`, `player_1_name`, `player_2_id`, `player_2_name` | Stable player identity information. |
| `winner_id`, `winner_name` | Supervised learning target at match level. |
| `best_of`, `set_score` | Match format and final result structure. |
| `p1_serve_points_won`, `p2_serve_points_won` | Serve performance indicators. |
| `p1_break_points_won`, `p2_break_points_won` | Pressure and return indicators. |
| `p1_aces`, `p2_aces`, `p1_double_faults`, `p2_double_faults` | Point-level or match-level event indicators. |
| `ranking_p1`, `ranking_p2`, `elo_p1`, `elo_p2` | Pre-match strength indicators. |
| `head_to_head_p1`, `head_to_head_p2` | Historical interaction features. |
| `target` | Target variable, for example match win or point-win probability proxy. |

The exact feature set may vary, but the feature definition must remain identical between training data and future prediction data. XGBoost depends on a fixed feature space, and the cited study highlights that model quality depends on the selected indicators and their interactions rather than on isolated linear correlations alone.[cite:6]

## Data Structure Rules

The following structural rules are recommended so that retraining remains stable and reproducible:

- Use one row per completed match.
- Use explicit column names with fixed spelling.
- Store numeric variables as numeric values, not as formatted strings.
- Encode categorical variables consistently, for example `surface` as `Hard`, `Clay`, or `Grass`.
- Do not mix training targets with unavailable future information.
- Ensure that all features used at prediction time are already known before the simulated match starts.

This last rule is important because the simulation must consume only pre-match information when it predicts new brackets. Otherwise, information leakage can bias the model and inflate apparent accuracy.[cite:6]

## Modifying Model Parameters

The repository can be adapted in two major places: the XGBoost model and the Monte Carlo simulation. The research paper explains that XGBoost iteratively builds trees from previous residuals until the error no longer improves or a predefined limit is reached.[cite:6] This means the most relevant training parameters are the number of trees, tree depth, learning rate, and regularization strength.[cite:6]

Typical XGBoost parameters that can be modified are:

- `nrounds` or `n_estimators`: number of boosting iterations.
- `max_depth`: maximum depth of each tree.
- `eta` or `learning_rate`: update step size.
- `subsample`: fraction of rows used per tree.
- `colsample_bytree`: fraction of features used per tree.
- `min_child_weight`: lower bound for leaf splitting.
- `objective`: prediction target, for example binary classification.

When a new dataset differs in scale or feature richness, these parameters usually require retuning. A smaller dataset often benefits from a simpler model, while a larger dataset can support deeper trees and more rounds if overfitting is controlled.[cite:6]

## Modifying Matches and Brackets

The tournament simulation can also be changed to represent different competition settings. The seminar presentation describes the simulation stage as the final step after model training and shows that real outcomes, simulated outcomes, and bookmaker comparisons can all be produced from the same pipeline.[cite:1]

Three components are usually configurable:

### Match format

The match logic defines how many sets are required to win a match. This can be changed by a parameter such as:

```text
best_of = 3
```

or

```text
best_of = 5
```

This parameter changes the path from point-level or game-level probabilities to the final match winner. If the repository uses only match-level probabilities, the parameter still matters because it determines how those probabilities are interpreted in the tournament context.

### Tournament tree

The bracket structure defines who plays whom and in which round. A simple bracket file can be represented as a table such as:

| round | match_id | player_1 | player_2 | next_match |
|---|---|---|---|---|
| R32 | 1 | Player A | Player B | 17 |
| R32 | 2 | Player C | Player D | 17 |
| R16 | 17 | winner(1) | winner(2) | 25 |

This structure allows the repository to model standard knockout tournaments. To simulate another event, only the player list and the bracket mapping need to be changed.

### Number of simulation runs

Monte Carlo simulation quality depends on repeated sampling. The seminar material states that Monte Carlo methods estimate outcomes by repeated random sampling over many runs.[cite:1] A parameter such as the following usually controls this behavior:

```text
n_simulations = 10000
```

A larger value produces more stable probability estimates but also increases runtime.[cite:1]

## Typical Output

The repository output is an aggregated summary of repeated tournament simulations rather than a single deterministic prediction. The seminar presentation explicitly contrasts simulation results with real outcomes and bookmaker expectations, which implies that the output is intended for comparative evaluation as well as prediction.[cite:1]

Typical outputs are:

- Match win probabilities for each scheduled pairing.
- Round-by-round advancement probabilities.
- Probability of reaching quarterfinals, semifinals, and finals.
- Tournament title probability for each player.
- Most frequent final pairings.
- Optional comparison tables between simulation, real outcomes, and external odds.[cite:1]

Depending on the implementation, the output can be written as CSV files, R objects, plots, or console summaries. The key result is always a probability distribution over possible tournament outcomes rather than one fixed bracket.[cite:1]

## Practical Extension

When extending the repository to another tournament or season, the most important requirement is consistency. The new dataset must use the same feature definitions, the same preprocessing rules, and a compatible bracket description. If these conditions are met, the trained XGBoost model can estimate match probabilities on the new input, and the Monte Carlo engine can transform them into tournament-level forecasts.[cite:1][cite:6]
