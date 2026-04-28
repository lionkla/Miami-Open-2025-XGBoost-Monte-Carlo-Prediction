############################################################
## Miami Open 2025 — Monte Carlo Bracket Simulation
## Optimiert: Match-Probs einmalig vorberechnen
############################################################

library(xgboost)
library(Matrix)
library(data.table)
library(ggplot2)
library(scales)
library(tidyr)
library(reshape2)

############################################################
## 1. Laden
############################################################

model_grid   <- readRDS("output/model_grid.rds")
feature_cols <- readRDS("output/feature_cols.rds")
dt_train     <- fread("output/atp_preprocessed_for_simulation_training.csv")

############################################################
## 2. Spieler definieren
############################################################

player_names <- c(
  "Taylor Fritz",
  "Matteo Berrettini",
  "Novak Djokovic",
  "Sebastian Korda",
  "Jakub Mensik",
  "Arthur Fils",
  "Grigor Dimitrov",
  "Francisco Cerundolo"
)

############################################################
## 3. Letzten Feature-Vektor pro Spieler holen (als p1)
############################################################

get_player_features <- function(name, dt, feature_cols) {
  rows <- dt[p1_name == name]
  if (nrow(rows) == 0) {
    # Fallback: als p2 suchen und Spalten spiegeln
    rows <- dt[p2_name == name]
    if (nrow(rows) == 0) { warning(paste("Nicht gefunden:", name)); return(NULL) }
    row  <- rows[.N]
    feat <- as.data.frame(row[, feature_cols, with = FALSE])
    old  <- names(feat)
    new  <- ifelse(startsWith(old,"p1_"), paste0("p2_",substring(old,4)),
                   ifelse(startsWith(old,"p2_"), paste0("p1_",substring(old,4)), old))
    names(feat) <- new
    for (fc in feature_cols) if (!fc %in% names(feat)) feat[[fc]] <- NA
    return(feat[, feature_cols, drop = FALSE])
  }
  row  <- rows[.N]
  as.data.frame(row[, feature_cols, with = FALSE])
}

player_feats <- lapply(player_names, get_player_features,
                       dt = dt_train, feature_cols = feature_cols)
names(player_feats) <- player_names

############################################################
## FIX: Auch dt_train auf eine Zeile pro Match bringen
############################################################

# match_key muss auch in dt_train existieren
dt_train[, match_key := paste(
  pmin(p1_name, p2_name),
  pmax(p1_name, p2_name),
  tourney_date, sep = "|"
)]

# Nur Gewinnerperspektive behalten
dt_train_one <- dt_train[p1_wins == 1]

# 50% zufällig spiegeln → Balance 50/50
set.seed(42)
idx_flip_tr <- sample.int(nrow(dt_train_one),
                          size = floor(0.5 * nrow(dt_train_one)))
dt_flip_tr  <- dt_train_one[idx_flip_tr]

p1_cols_tr  <- grep("^p1_", names(dt_flip_tr), value = TRUE)
p2_cols_tr  <- grep("^p2_", names(dt_flip_tr), value = TRUE)
swap_cols_tr <- intersect(
  sub("^p1_", "", p1_cols_tr),
  sub("^p2_", "", p2_cols_tr)
)
for (col in swap_cols_tr) {
  tmp                            <- dt_flip_tr[[paste0("p1_", col)]]
  dt_flip_tr[[paste0("p1_", col)]] <- dt_flip_tr[[paste0("p2_", col)]]
  dt_flip_tr[[paste0("p2_", col)]] <- tmp
}
dt_flip_tr$p1_wins <- 0L

# Differenz-Features neu berechnen
if (all(c("p1_rank","p2_rank") %in% names(dt_flip_tr))) {
  dt_flip_tr[, rank_diff        := p1_rank - p2_rank]
  dt_flip_tr[, rank_points_diff := p1_rank_points - p2_rank_points]
}
if (all(c("p1_ht","p2_ht") %in% names(dt_flip_tr))) {
  dt_flip_tr[, height_diff := p1_ht - p2_ht]
}
if (all(c("p1_seed","p2_seed") %in% names(dt_flip_tr))) {
  dt_flip_tr[, seed_diff := p1_seed - p2_seed]
}

dt_train_balanced <- rbind(dt_train_one[-idx_flip_tr], dt_flip_tr)

cat("Train-Label nach Fix:", table(dt_train_balanced$p1_wins), "\n")
cat("Train-Zeilen:", nrow(dt_train_balanced), "\n")

############################################################
## 4. Match-Wahrscheinlichkeiten EINMALIG vorberechnen
##    prob_matrix[i,j] = P(Spieler i schlägt Spieler j)
############################################################

cat("Berechne Match-Wahrscheinlichkeitsmatrix...\n")

n_players    <- length(player_names)
prob_matrix  <- matrix(0.5, nrow = n_players, ncol = n_players,
                       dimnames = list(player_names, player_names))

build_match_row <- function(f1, f2, feature_cols) {
  row <- as.data.frame(matrix(NA, nrow = 1, ncol = length(feature_cols)))
  names(row) <- feature_cols
  
  # p1-Features eintragen
  for (col in feature_cols) {
    if (col %in% names(f1)) row[[col]] <- f1[[col]]
  }
  
  # p2-Features überschreiben (p2_* aus f2's p1_*)
  for (col in feature_cols) {
    if (startsWith(col, "p2_")) {
      src <- paste0("p1_", substring(col, 4))
      if (src %in% names(f2)) row[[col]] <- f2[[src]]
    }
  }
  
  # Differenz-Features neu berechnen
  if (all(c("p1_rank","p2_rank") %in% names(row))) {
    r1 <- suppressWarnings(as.numeric(row[["p1_rank"]]))
    r2 <- suppressWarnings(as.numeric(row[["p2_rank"]]))
    if (!is.na(r1) && !is.na(r2) && "rank_diff" %in% feature_cols)
      row[["rank_diff"]] <- r1 - r2
  }
  
  # Kategorische Spalten faktorisieren
  for (col in feature_cols) {
    v <- row[[col]]
    if (is.character(v) || is.factor(v)) row[[col]] <- as.factor(v)
  }
  
  row$p1_wins <- 0L
  row
}

for (i in 1:n_players) {
  for (j in 1:n_players) {
    if (i == j) next
    f1 <- player_feats[[player_names[i]]]
    f2 <- player_feats[[player_names[j]]]
    if (is.null(f1) || is.null(f2)) next
    
    match_row <- build_match_row(f1, f2, feature_cols)
    
    X <- tryCatch(
      sparse.model.matrix(p1_wins ~ . - 1, data = match_row),
      error = function(e) NULL
    )
    if (is.null(X)) next
    
    prob_matrix[i, j] <- predict(model_grid,
                                 newdata = xgb.DMatrix(data = X))[1]
  }
}

cat("Wahrscheinlichkeitsmatrix:\n")
print(round(prob_matrix, 3))

############################################################
## 5. Bracket-Definition
############################################################

train_xy <- build_matrix(dt_train_balanced, feature_cols, impute_vals)

# QF: je list(p1 = oberer, p2 = unterer)
qf_bracket <- list(
  list(p1 = "Taylor Fritz",        p2 = "Matteo Berrettini"),
  list(p1 = "Sebastian Korda",     p2 = "Novak Djokovic"),
  list(p1 = "Arthur Fils",         p2 = "Jakub Mensik"),
  list(p1 = "Francisco Cerundolo", p2 = "Grigor Dimitrov")
)

# Reale Sieger (für Vergleich am Ende)
real <- list(
  qf = c("Taylor Fritz","Novak Djokovic","Jakub Mensik","Grigor Dimitrov"),
  sf = c("Jakub Mensik","Novak Djokovic"),
  champion = "Jakub Mensik"
)

############################################################
## 6. Schnelle Monte Carlo Simulation (prob_matrix lookup)
############################################################

sim_match <- function(p1, p2) {
  p <- prob_matrix[p1, p2]
  if (runif(1) < p) p1 else p2
}

n_sim     <- 10000
set.seed(42)

reach_sf   <- setNames(integer(n_players), player_names)
reach_f    <- setNames(integer(n_players), player_names)
reach_win  <- setNames(integer(n_players), player_names)
win_log    <- matrix(0L, nrow = n_sim, ncol = n_players,
                     dimnames = list(NULL, player_names))

# Reale Bracket-Verläufe zählen
real_path_count <- 0L

cat("Starte Simulation (n =", n_sim, ")...\n")
pb <- txtProgressBar(min = 0, max = n_sim, style = 3)

for (s in 1:n_sim) {
  
  # QF
  qf_w <- vapply(qf_bracket, function(m) sim_match(m$p1, m$p2),
                 character(1))
  
  # SF
  sf_w <- c(
    sim_match(qf_w[1], qf_w[2]),
    sim_match(qf_w[3], qf_w[4])
  )
  
  # Final
  champion <- sim_match(sf_w[1], sf_w[2])
  
  # Tracking
  reach_sf[qf_w]  <- reach_sf[qf_w]  + 1L
  reach_f[sf_w]   <- reach_f[sf_w]   + 1L
  reach_win[champion] <- reach_win[champion] + 1L
  win_log[s, champion] <- 1L
  
  # Prüfen ob realer Pfad reproduziert
  if (identical(sort(qf_w), sort(real$qf)) &&
      identical(sort(sf_w), sort(real$sf)) &&
      champion == real$champion) {
    real_path_count <- real_path_count + 1L
  }
  
  setTxtProgressBar(pb, s)
}
close(pb)

############################################################
## 7. Ergebnis-Tabelle
############################################################

results_df <- data.frame(
  Spieler       = player_names,
  HF_Einzug     = round(reach_sf[player_names]  / n_sim * 100, 1),
  F_Einzug      = round(reach_f[player_names]   / n_sim * 100, 1),
  Titel         = round(reach_win[player_names] / n_sim * 100, 1),
  stringsAsFactors = FALSE
) |> dplyr::arrange(desc(Titel))

cat("\n========================================\n")
cat(sprintf("  Miami Open 2025 — %d Simulationen\n", n_sim))
cat("========================================\n")
print(results_df, row.names = FALSE)
cat(sprintf("\nRealer Turnierpfad reproduziert in: %.2f%% der Simulationen\n",
            real_path_count / n_sim * 100))

############################################################
## 8. Plots
############################################################

dir.create("output", showWarnings = FALSE)

## Plot 1 — Balkendiagramm pro Runde
plot_long <- tidyr::pivot_longer(results_df,
                                 cols = c("HF_Einzug","F_Einzug","Titel"),
                                 names_to = "Runde", values_to = "Pct")

plot_long$Runde   <- factor(plot_long$Runde,
                            levels = c("HF_Einzug","F_Einzug","Titel"),
                            labels = c("Halbfinale","Finale","Titel"))
plot_long$Spieler <- factor(plot_long$Spieler,
                            levels = results_df$Spieler)

p1 <- ggplot(plot_long, aes(x = Spieler, y = Pct, fill = Runde)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.7) +
  geom_text(aes(label = paste0(Pct,"%")),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3.2) +
  scale_fill_manual(values = c("#4e9af1","#f1a74e","#e05c5c")) +
  scale_y_continuous(limits = c(0,105),
                     labels = function(x) paste0(x,"%")) +
  labs(
    title    = "Miami Open 2025 — Monte Carlo Simulation",
    subtitle = sprintf("n = %d Simulationen  |  Oberfläche: Hard", n_sim),
    x = NULL, y = "Wahrscheinlichkeit (%)", fill = "Runde",
    caption  = "Modell: XGBoost (Grid-Search + 5-fold CV) | Training: ATP 2021–2023"
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 28, hjust = 1),
        legend.position = "top",
        plot.title = element_text(face = "bold"),
        panel.grid.major.x = element_blank())

ggsave("output/miami_mc_balken.png", p1, width = 12, height = 7, dpi = 300)
cat("Plot 1 gespeichert.\n")

## Plot 2 — Konvergenzkurven
cum_df <- as.data.frame(
  apply(win_log, 2, function(x) cumsum(x) / seq_along(x)) * 100
)
cum_df$sim <- 1:n_sim

cum_long <- tidyr::pivot_longer(cum_df, cols = -sim,
                                names_to = "Spieler", values_to = "Titelprob")

top_players <- results_df$Spieler[results_df$Titel > 1]

p2 <- ggplot(cum_long[cum_long$Spieler %in% top_players,],
             aes(x = sim, y = Titelprob, color = Spieler)) +
  geom_line(linewidth = 0.75, alpha = 0.85) +
  scale_x_continuous(labels = scales::comma) +
  scale_y_continuous(labels = function(x) paste0(x,"%")) +
  labs(
    title    = "Konvergenz der Titelwahrscheinlichkeit",
    subtitle = "Kumulativer Verlauf über alle Simulationen",
    x = "Anzahl Simulationen", y = "Kumul. Titelwahrsch. (%)",
    color = "Spieler",
    caption = "Miami Open 2025 | XGBoost"
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "right")

ggsave("output/miami_mc_konvergenz.png", p2, width = 12, height = 6, dpi = 300)
cat("Plot 2 gespeichert.\n")

## Plot 3 — Heatmap
heat_long <- reshape2::melt(
  as.matrix(results_df[, c("HF_Einzug","F_Einzug","Titel")]),
  varnames = c("Spieler","Runde"), value.name = "Pct"
)
heat_long$Spieler <- results_df$Spieler[heat_long$Spieler]
heat_long$Runde   <- factor(heat_long$Runde,
                            levels = c("HF_Einzug","F_Einzug","Titel"),
                            labels = c("Halbfinale","Finale","Titel"))

p3 <- ggplot(heat_long, aes(x = Runde, y = Spieler, fill = Pct)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = paste0(Pct,"%")),
            fontface = "bold", size = 4.5,
            color = ifelse(heat_long$Pct > 40, "white", "white")) +
  scale_fill_gradient(low = "#1a3a5c", high = "#e05c5c",
                      name = "Wahrsch. (%)") +
  scale_y_discrete(limits = rev(results_df$Spieler)) +
  labs(
    title   = "Heatmap — Rundenwahrscheinlichkeiten Miami Open 2025",
    x = NULL, y = NULL,
    caption = sprintf("XGBoost | n = %d Simulationen", n_sim)
  ) +
  theme_minimal(base_size = 13) +
  theme(plot.title = element_text(face = "bold"),
        axis.text  = element_text(size = 11))

ggsave("output/miami_mc_heatmap.png", p3, width = 9, height = 6, dpi = 300)
cat("Plot 3 gespeichert.\n")

cat("\nFertig. Alle Outputs in output/\n")