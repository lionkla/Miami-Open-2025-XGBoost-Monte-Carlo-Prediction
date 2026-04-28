############################################################
## 1. Pakete laden
############################################################

library(data.table)
library(xgboost)
library(Matrix)
library(dplyr)
library(Metrics)

############################################################
## 2. Daten einlesen
############################################################

train_path <- "output/atp_preprocessed_for_simulation_training.csv"
valid_path <- "output/atp_preprocessed_for_simulation_valid.csv"

if (!file.exists(train_path)) stop(paste0("Trainingsdatei nicht gefunden: ", train_path))
if (!file.exists(valid_path)) stop(paste0("2024-Datei nicht gefunden: ", valid_path))

dt_train <- fread(train_path)
dt_2024  <- fread(valid_path)

if (!"p1_wins" %in% names(dt_train) || !"p1_wins" %in% names(dt_2024)) {
  stop("Spalte 'p1_wins' fehlt in einer der CSV-Dateien.")
}

dt_train <- dt_train[!is.na(p1_wins)]
dt_2024  <- dt_2024[!is.na(p1_wins)]

if ("tourney_year" %in% names(dt_train)) print(table(dt_train$tourney_year))
if ("tourney_year" %in% names(dt_2024))  print(table(dt_2024$tourney_year))

############################################################
## 3. Eine Zeile pro Match — dt_2024 und dt_train
############################################################

balance_one_row_per_match <- function(dt, seed_val) {
  set.seed(seed_val)
  
  dt[, match_key := paste(
    pmin(p1_name, p2_name),
    pmax(p1_name, p2_name),
    tourney_date, sep = "|"
  )]
  
  dt_one   <- dt[p1_wins == 1]
  idx_flip <- sample.int(nrow(dt_one), size = floor(0.5 * nrow(dt_one)))
  dt_flip  <- dt_one[idx_flip]
  
  p1_cols   <- grep("^p1_", names(dt_flip), value = TRUE)
  p2_cols   <- grep("^p2_", names(dt_flip), value = TRUE)
  swap_cols <- intersect(sub("^p1_", "", p1_cols),
                         sub("^p2_", "", p2_cols))
  
  for (col in swap_cols) {
    tmp                            <- dt_flip[[paste0("p1_", col)]]
    dt_flip[[paste0("p1_", col)]] <- dt_flip[[paste0("p2_", col)]]
    dt_flip[[paste0("p2_", col)]] <- tmp
  }
  dt_flip$p1_wins <- 0L
  
  if (all(c("p1_rank", "p2_rank") %in% names(dt_flip)))
    dt_flip[, rank_diff        := p1_rank - p2_rank]
  if (all(c("p1_rank_points", "p2_rank_points") %in% names(dt_flip)))
    dt_flip[, rank_points_diff := p1_rank_points - p2_rank_points]
  if (all(c("p1_ht", "p2_ht") %in% names(dt_flip)))
    dt_flip[, height_diff      := p1_ht - p2_ht]
  if (all(c("p1_seed", "p2_seed") %in% names(dt_flip)))
    dt_flip[, seed_diff        := p1_seed - p2_seed]
  if (all(c("p1_age", "p2_age") %in% names(dt_flip)))
    dt_flip[, age_diff         := p1_age - p2_age]
  
  rbind(dt_one[-idx_flip], dt_flip)
}

dt_train_balanced <- balance_one_row_per_match(dt_train, seed_val = 42)
cat("Train-Label nach Balancierung:", table(dt_train_balanced$p1_wins), "\n")
cat("Train-Zeilen:", nrow(dt_train_balanced), "\n")

dt_2024_balanced <- balance_one_row_per_match(dt_2024, seed_val = 123)
cat("2024-Label nach Balancierung:", table(dt_2024_balanced$p1_wins), "\n")

set.seed(123)
n_bal     <- nrow(dt_2024_balanced)
idx_valid <- sample.int(n_bal, size = floor(0.5 * n_bal))
dt_valid  <- dt_2024_balanced[ idx_valid]
dt_test   <- dt_2024_balanced[-idx_valid]

cat("Valid:", nrow(dt_valid), "| Test:", nrow(dt_test), "\n")
cat("Test-Label:", table(dt_test$p1_wins), "\n")

############################################################
## 4. Feature-Set definieren
############################################################

cols_exclude <- intersect(
  c("p1_wins", "tourney_id", "tourney_name", "tourney_date",
    "match_num", "score", "round", "p1_name", "p2_name",
    "source_file", "match_key"),
  names(dt_train)
)

feature_candidates <- setdiff(names(dt_train), cols_exclude)
id_like            <- grep("(_id$|_player$|_playerid$)",
                           feature_candidates, value = TRUE)
feature_cols_raw   <- setdiff(feature_candidates, id_like)

## In-Match-Statistiken entfernen (Target Leakage)
match_stat_patterns <- c(
  "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
  "SvGms", "bpSaved", "bpFaced", "minutes"
)
pattern_regex <- paste0(
  "^(p1_|p2_)(",
  paste(match_stat_patterns, collapse = "|"),
  ")$"
)
leaky_cols <- grep(pattern_regex, feature_cols_raw,
                   value = TRUE, perl = TRUE)

## Redundante Geburtsdatum-Spalten entfernen
## (age und age_diff bilden dieselbe Information ab)
redundant_cols <- c("p1_birth_date", "p2_birth_date")

feature_cols <- setdiff(feature_cols_raw,
                        c(leaky_cols, redundant_cols))

cat("Entfernte Leakage-Features    (", length(leaky_cols),    "):\n")
print(leaky_cols)
cat("Entfernte redundante Features (", length(redundant_cols), "):\n")
print(redundant_cols)
cat("Verbleibende Features:", length(feature_cols), "\n")
print(feature_cols)

############################################################
## 5. Hilfsfunktionen: Imputation, Matrix, Alignment
############################################################

compute_imputation_values <- function(dt_subset, feature_cols) {
  dat <- as.data.frame(dt_subset[, feature_cols, with = FALSE])
  impute_vals <- list()
  for (col in feature_cols) {
    if (is.numeric(dat[[col]])) {
      impute_vals[[col]] <- median(dat[[col]], na.rm = TRUE)
    } else {
      tbl <- table(dat[[col]])
      impute_vals[[col]] <- names(tbl)[which.max(tbl)]
    }
  }
  return(impute_vals)
}

build_matrix <- function(dt_subset, feature_cols, impute_vals) {
  dat <- as.data.frame(dt_subset[, c("p1_wins", feature_cols), with = FALSE])
  
  for (col in feature_cols) {
    if (is.character(dat[[col]]) || is.factor(dat[[col]]))
      dat[[col]] <- as.factor(dat[[col]])
  }
  for (col in feature_cols) {
    if (anyNA(dat[[col]]))
      dat[[col]][is.na(dat[[col]])] <- impute_vals[[col]]
  }
  
  rows_na <- sum(!complete.cases(dat))
  cat("Verbleibende NA-Zeilen:", rows_na, "von", nrow(dat), "\n")
  
  y           <- as.integer(dat$p1_wins == 1)
  dat$p1_wins <- y
  cat("Klassenverteilung: 0 =", sum(y == 0),
      "| 1 =", sum(y == 1),
      "| Anteil 1:", round(mean(y), 3), "\n")
  
  X <- sparse.model.matrix(p1_wins ~ . - 1, data = dat)
  stopifnot(nrow(X) == length(y))
  list(X = X, y = y)
}

align_matrix <- function(X_new, train_cols) {
  missing_cols <- setdiff(train_cols, colnames(X_new))
  if (length(missing_cols) > 0) {
    zero_mat <- Matrix::Matrix(0,
                               nrow     = nrow(X_new),
                               ncol     = length(missing_cols),
                               sparse   = TRUE,
                               dimnames = list(NULL, missing_cols))
    X_new <- cbind(X_new, zero_mat)
  }
  X_new[, train_cols, drop = FALSE]
}

############################################################
## 6. Matrizen bauen
############################################################

impute_vals      <- compute_imputation_values(dt_train_balanced, feature_cols)
train_xy         <- build_matrix(dt_train_balanced, feature_cols, impute_vals)
train_cols_fixed <- colnames(train_xy$X)

valid_raw <- build_matrix(dt_valid, feature_cols, impute_vals)
test_raw  <- build_matrix(dt_test,  feature_cols, impute_vals)

valid_xy <- list(X = align_matrix(valid_raw$X, train_cols_fixed),
                 y = valid_raw$y)
test_xy  <- list(X = align_matrix(test_raw$X,  train_cols_fixed),
                 y = test_raw$y)

cat("Spalten — Train:", ncol(train_xy$X),
    "| Valid:", ncol(valid_xy$X),
    "| Test:",  ncol(test_xy$X), "\n")
cat("Spalten identisch:",
    identical(colnames(train_xy$X), colnames(test_xy$X)), "\n")
cat("Train-Label:", table(train_xy$y), "\n")
cat("Test-Label: ", table(test_xy$y),  "\n")

dtrain    <- xgb.DMatrix(data = train_xy$X, label = train_xy$y)
dvalid    <- xgb.DMatrix(data = valid_xy$X, label = valid_xy$y)
dtest     <- xgb.DMatrix(data = test_xy$X,  label = test_xy$y)
watchlist <- list(train = dtrain, valid = dvalid)

############################################################
## 7a. Cross-Validation
############################################################

set.seed(123)
cv_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 5,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1
)
cv_model <- xgb.cv(
  params                = cv_params,
  data                  = dtrain,
  nrounds               = 500,
  nfold                 = 5,
  stratified            = TRUE,
  early_stopping_rounds = 50,
  maximize              = FALSE,
  print_every_n         = 10
)
best_rounds <- cv_model$best_iteration
if (is.null(best_rounds) || is.na(best_rounds) || best_rounds < 1)
  best_rounds <- 100
cat("Beste Runde:", best_rounds, "\n")

############################################################
## 7b. Initiales Modell
############################################################

final_model <- xgb.train(
  params        = cv_params,
  data          = dtrain,
  nrounds       = best_rounds,
  evals         = watchlist,
  print_every_n = 10
)

############################################################
## 8. Grid Search
############################################################

grid <- expand.grid(
  eta              = c(0.05, 0.1, 0.2),
  max_depth        = c(4, 6, 8),
  min_child_weight = c(3, 5, 10),
  stringsAsFactors = FALSE
)
res_list <- list()
for (i in seq_len(nrow(grid))) {
  pars    <- unlist(grid[i, ])
  param_i <- list(
    objective        = "binary:logistic",
    eval_metric      = "logloss",
    eta              = pars["eta"],
    max_depth        = pars["max_depth"],
    min_child_weight = pars["min_child_weight"],
    subsample        = 0.8,
    colsample_bytree = 0.8,
    lambda           = 1
  )
  cat("Grid", i, "/", nrow(grid), "\n")
  cv_r <- xgb.cv(
    params                = param_i,
    data                  = dtrain,
    nrounds               = 300,
    nfold                 = 5,
    stratified            = TRUE,
    early_stopping_rounds = 30,
    maximize              = FALSE,
    print_every_n         = 100
  )
  best_i <- cv_r$best_iteration
  if (is.null(best_i) || is.na(best_i) || best_i < 1)
    best_i <- nrow(cv_r$evaluation_log)
  
  cv_names        <- names(cv_r$evaluation_log)
  logloss_col     <- grep("test_logloss_mean", cv_names, value = TRUE)[1]
  logloss_std_col <- grep("test_logloss_std",  cv_names, value = TRUE)[1]
  
  res_list[[i]] <- data.frame(
    iteration        = i,
    eta              = pars["eta"],
    max_depth        = pars["max_depth"],
    min_child_weight = pars["min_child_weight"],
    logloss_mean     = cv_r$evaluation_log[[logloss_col]][best_i],
    logloss_std      = if (length(logloss_std_col) == 1 &&
                           !is.na(logloss_std_col))
      cv_r$evaluation_log[[logloss_std_col]][best_i]
    else NA,
    nrounds          = best_i,
    stringsAsFactors = FALSE
  )
}
res_df   <- do.call(rbind, res_list)
best_row <- res_df[which.min(res_df$logloss_mean), ]
cat("\nBester Parameter-Satz:\n")
print(best_row)

############################################################
## 9. Finales Modell
############################################################

final_params_grid <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = best_row$eta,
  max_depth        = best_row$max_depth,
  min_child_weight = best_row$min_child_weight,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1
)
model_grid <- xgb.train(
  params        = final_params_grid,
  data          = dtrain,
  nrounds       = best_row$nrounds,
  evals         = watchlist,
  print_every_n = 10
)
saveRDS(model_grid,        "output/model_grid.rds")
saveRDS(feature_cols,      "output/feature_cols.rds")
saveRDS(impute_vals,       "output/impute_vals.rds")
saveRDS(train_cols_fixed,  "output/train_cols_fixed.rds")
cat("Modell gespeichert.\n")

############################################################
## 10. Evaluation
############################################################

pred_test_prob  <- predict(model_grid, newdata = dtest)
logloss_test    <- logLoss(test_xy$y, pred_test_prob)
auc_test        <- Metrics::auc(test_xy$y, pred_test_prob)
pred_test_class <- as.integer(pred_test_prob >= 0.5)
accuracy_test   <- mean(pred_test_class == test_xy$y)

pred_train_prob <- predict(model_grid, newdata = dtrain)
auc_train       <- Metrics::auc(train_xy$y, pred_train_prob)
logloss_train   <- logLoss(train_xy$y, pred_train_prob)

cat("\n========================================\n")
cat("Train-Logloss:        ", round(logloss_train,  4), "\n")
cat("Train-AUC:            ", round(auc_train,      4), "\n")
cat("----------------------------------------\n")
cat("Test-Label-Verteilung:", table(test_xy$y),          "\n")
cat("Test-Logloss:         ", round(logloss_test,   4),  "\n")
cat("Test-AUC:             ", round(auc_test,       4),  "\n")
cat("Test-Accuracy (>=0.5):", round(accuracy_test,  4),  "\n")
cat("========================================\n")

############################################################
## 11. Feature Importance
############################################################

importance <- xgb.importance(model = model_grid)
cat("\nTop-20 Features:\n")
print(head(importance, 20))
xgb.plot.importance(importance[1:20])

############################################################
## 12. Kalibrierung
############################################################

cal_df <- data.frame(
  prob  = pred_test_prob,
  label = test_xy$y
)
cal_df$bucket <- cut(cal_df$prob,
                     breaks         = seq(0, 1, by = 0.1),
                     include.lowest = TRUE)

cal_summary <- cal_df |>
  dplyr::group_by(bucket) |>
  dplyr::summarise(
    mean_pred  = round(mean(prob),  3),
    actual_wr  = round(mean(label), 3),
    n          = dplyr::n(),
    .groups    = "drop"
  )

cat("\nKalibrierung (vorhergesagt vs. tatsächlich):\n")
print(cal_summary)

## Kalibrierungsplot
plot(cal_summary$mean_pred,
     cal_summary$actual_wr,
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Vorhergesagte Wahrscheinlichkeit",
     ylab = "Tatsächliche Gewinnrate",
     main = "Kalibrierungskurve",
     pch  = 19, col = "steelblue", cex = 1.4)
abline(0, 1, lty = 2, col = "gray50")
text(cal_summary$mean_pred,
     cal_summary$actual_wr,
     labels = paste0("n=", cal_summary$n),
     pos = 3, cex = 0.75)

############################################################
## 13. Hilfsfunktion für neue Matches
############################################################

predict_match_win_prob <- function(new_matches, feature_cols,
                                   xgb_model, ref_data) {
  missing <- setdiff(feature_cols, names(new_matches))
  if (length(missing) > 0)
    stop(paste("Fehlende Feature-Spalten:", paste(missing, collapse = ", ")))
  
  dat <- as.data.frame(new_matches[, feature_cols, drop = FALSE])
  
  for (col in feature_cols) {
    if (is.character(dat[[col]]) || is.factor(dat[[col]])) {
      ref_col <- ref_data[[col]]
      if (!is.null(ref_col) &&
          (is.character(ref_col) || is.factor(ref_col))) {
        dat[[col]] <- factor(dat[[col]], levels = unique(ref_col))
      } else {
        dat[[col]] <- as.factor(dat[[col]])
      }
    }
  }
  
  dat$p1_wins <- 0L
  X_new <- sparse.model.matrix(p1_wins ~ . - 1, data = dat)
  dnew  <- xgb.DMatrix(data = X_new)
  predict(xgb_model, newdata = dnew)
}