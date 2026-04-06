# =========================================================
# R script: Extract metrics from JSON experiment files
# and create multiple insightful metric-focused graphs
# =========================================================

# Install if needed:
# install.packages(c("jsonlite", "dplyr", "ggplot2", "readr", "tidyr", "forcats", "stringr"))

library(jsonlite)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(forcats)
library(stringr)

# -----------------------------
# 1) Set your folder path here
# -----------------------------
# Change this to the folder that contains your JSON files
data_dir <- "results\graphs\experement1_json"

# Output files/folders
metrics_csv <- file.path(data_dir, "model_language_metrics.csv")
graphs_dir  <- file.path(data_dir, "generated_graphs")

if (!dir.exists(graphs_dir)) {
  dir.create(graphs_dir, recursive = TRUE)
}

# -----------------------------
# 2) Helper functions
# -----------------------------
safe_div <- function(a, b) {
  ifelse(is.na(b) | b == 0, NA_real_, a / b)
}

extract_model_name <- function(x) {
  x <- tolower(x)
  case_when(
    str_detect(x, "claude") ~ "Claude",
    str_detect(x, "gemini") ~ "Gemini",
    str_detect(x, "openai") ~ "OpenAI",
    str_detect(x, "llama|meta-llama") ~ "Llama",
    str_detect(x, "deepseek") ~ "DeepSeek",
    TRUE ~ x
  )
}

extract_language_name <- function(x) {
  x <- tolower(x)
  case_when(
    str_detect(x, "arabic") ~ "Arabic",
    str_detect(x, "chinese") ~ "Chinese",
    str_detect(x, "urdu") ~ "Urdu",
    TRUE ~ x
  )
}

compute_metrics_from_results <- function(results_df) {
  if (is.null(results_df) || nrow(results_df) == 0) {
    return(NULL)
  }

  classified <- results_df %>%
    filter(!is.na(prediction), !is.na(ground_truth))

  if (nrow(classified) == 0) {
    return(NULL)
  }

  tp <- sum(classified$ground_truth == "depressed" & classified$prediction == "depressed", na.rm = TRUE)
  fp <- sum(classified$ground_truth == "not depressed" & classified$prediction == "depressed", na.rm = TRUE)
  tn <- sum(classified$ground_truth == "not depressed" & classified$prediction == "not depressed", na.rm = TRUE)
  fn <- sum(classified$ground_truth == "depressed" & classified$prediction == "not depressed", na.rm = TRUE)

  total_samples <- nrow(results_df)
  total_classified <- tp + fp + tn + fn
  error_responses <- sum(!is.na(results_df$error) & results_df$error != "", na.rm = TRUE)
  unclear_responses <- total_samples - total_classified - error_responses

  precision <- safe_div(tp, tp + fp)
  recall <- safe_div(tp, tp + fn)
  f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0,
               NA_real_, 2 * precision * recall / (precision + recall))
  accuracy <- safe_div(tp + tn, total_samples)

  tibble(
    true_positives = tp,
    false_positives = fp,
    true_negatives = tn,
    false_negatives = fn,
    precision = precision,
    recall = recall,
    f1_score = f1,
    accuracy = accuracy,
    total_samples = total_samples,
    total_classified = total_classified,
    unclear_responses = unclear_responses,
    error_responses = error_responses
  )
}

read_one_json_metrics <- function(file_path) {
  cat("Reading:", basename(file_path), "\n")

  obj <- tryCatch(
    fromJSON(file_path, simplifyDataFrame = TRUE),
    error = function(e) NULL
  )

  if (is.null(obj)) {
    warning(paste("Skipping unreadable file:", file_path))
    return(NULL)
  }

  meta <- obj$metadata
  if (is.null(meta)) {
    meta <- list()
  }

  filename <- basename(file_path)

  model <- if (!is.null(meta$model)) meta$model else filename
  language <- if (!is.null(meta$language)) meta$language else filename
  experiment <- if (!is.null(meta$experiment)) meta$experiment else NA
  timestamp <- if (!is.null(meta$timestamp)) meta$timestamp else if (!is.null(meta$last_saved)) meta$last_saved else NA
  sample_size <- if (!is.null(meta$sample_size)) meta$sample_size else if (!is.null(meta$total)) meta$total else NA
  status <- if (!is.null(meta$status)) meta$status else "complete"
  completed <- if (!is.null(meta$completed)) meta$completed else NA

  if (!is.null(obj$metrics)) {
    m <- obj$metrics

    out <- tibble(
      file_name = filename,
      model_raw = as.character(model),
      model = extract_model_name(as.character(model)),
      language_raw = as.character(language),
      language = extract_language_name(as.character(language)),
      experiment = suppressWarnings(as.numeric(experiment)),
      timestamp = as.character(timestamp),
      status = as.character(status),
      completed = suppressWarnings(as.numeric(completed)),
      sample_size = suppressWarnings(as.numeric(sample_size)),
      true_positives = suppressWarnings(as.numeric(m$confusion_matrix$true_positives)),
      false_positives = suppressWarnings(as.numeric(m$confusion_matrix$false_positives)),
      true_negatives = suppressWarnings(as.numeric(m$confusion_matrix$true_negatives)),
      false_negatives = suppressWarnings(as.numeric(m$confusion_matrix$false_negatives)),
      precision = suppressWarnings(as.numeric(m$precision)),
      recall = suppressWarnings(as.numeric(m$recall)),
      f1_score = suppressWarnings(as.numeric(m$f1_score)),
      accuracy = suppressWarnings(as.numeric(m$accuracy)),
      total_samples = suppressWarnings(as.numeric(m$total_samples)),
      total_classified = suppressWarnings(as.numeric(m$total_classified)),
      unclear_responses = suppressWarnings(as.numeric(m$unclear_responses)),
      error_responses = suppressWarnings(as.numeric(m$error_responses))
    )

    return(out)
  }

  if (!is.null(obj$results)) {
    derived <- compute_metrics_from_results(as_tibble(obj$results))
    if (!is.null(derived)) {
      out <- tibble(
        file_name = filename,
        model_raw = as.character(model),
        model = extract_model_name(as.character(model)),
        language_raw = as.character(language),
        language = extract_language_name(as.character(language)),
        experiment = suppressWarnings(as.numeric(experiment)),
        timestamp = as.character(timestamp),
        status = as.character(status),
        completed = suppressWarnings(as.numeric(completed)),
        sample_size = suppressWarnings(as.numeric(sample_size))
      ) %>%
        bind_cols(derived)

      return(out)
    }
  }

  warning(paste("No usable metrics/results found in:", file_path))
  NULL
}

save_plot <- function(plot_obj, filename, width = 10, height = 6, dpi = 300) {
  ggsave(
    filename = file.path(graphs_dir, filename),
    plot = plot_obj,
    width = width,
    height = height,
    dpi = dpi
  )
}

# -----------------------------
# 3) Read all JSON files
# -----------------------------
json_files <- list.files(data_dir, pattern = "\\.json$", full.names = TRUE)

if (length(json_files) == 0) {
  stop("No JSON files found in the folder.")
}

metrics_df <- lapply(json_files, read_one_json_metrics) %>%
  bind_rows()

if (nrow(metrics_df) == 0) {
  stop("No usable metrics were extracted from the JSON files.")
}

# -----------------------------
# 4) Add useful derived columns
# -----------------------------
metrics_df <- metrics_df %>%
  mutate(
    coverage_rate = safe_div(total_classified, total_samples),
    error_rate = safe_div(error_responses, total_samples),
    unclear_rate = safe_div(unclear_responses, total_samples),
    fp_rate_of_total = safe_div(false_positives, total_samples),
    fn_rate_of_total = safe_div(false_negatives, total_samples),
    model_language = paste(model, language, sep = " - "),
    is_partial = tolower(status) == "partial"
  ) %>%
  arrange(language, desc(f1_score))

# Save consolidated CSV
write_csv(metrics_df, metrics_csv)
cat("\nSaved metrics CSV to:", metrics_csv, "\n")

# Optional: keep only full runs for main comparison plots
full_df <- metrics_df %>%
  filter(!is_partial | is.na(is_partial))

if (nrow(full_df) == 0) {
  full_df <- metrics_df
}

# -----------------------------
# 5) Graph 1: F1 heatmap
# -----------------------------
heatmap_df <- full_df %>%
  select(model, language, f1_score)

p1 <- ggplot(heatmap_df, aes(x = language, y = model, fill = f1_score)) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.3f", f1_score)), size = 4) +
  scale_fill_gradient(low = "lightyellow", high = "steelblue", na.value = "grey90") +
  labs(
    title = "F1 Score Heatmap by Model and Language",
    x = "Language",
    y = "Model",
    fill = "F1 Score"
  ) +
  theme_minimal(base_size = 12)

save_plot(p1, "01_f1_heatmap.png", width = 8, height = 5)

# -----------------------------
# 6) Graph 2: Precision vs Recall
# -----------------------------
p2 <- ggplot(full_df, aes(x = recall, y = precision, label = model_language, shape = language)) +
  geom_point(size = 4) +
  geom_text(nudge_y = 0.015, size = 3.5, check_overlap = TRUE) +
  labs(
    title = "Precision vs Recall by Model-Language Run",
    x = "Recall",
    y = "Precision",
    shape = "Language"
  ) +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_minimal(base_size = 12)

save_plot(p2, "02_precision_vs_recall.png", width = 10, height = 6)

# -----------------------------
# 7) Graph 3: Grouped F1 bars
# -----------------------------
p3 <- ggplot(full_df, aes(x = language, y = f1_score, fill = model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(
    aes(label = sprintf("%.3f", f1_score)),
    position = position_dodge(width = 0.8),
    vjust = -0.35,
    size = 3.3
  ) +
  labs(
    title = "F1 Score by Model Within Each Language",
    x = "Language",
    y = "F1 Score",
    fill = "Model"
  ) +
  ylim(0, min(1.05, max(full_df$f1_score, na.rm = TRUE) + 0.1)) +
  theme_minimal(base_size = 12)

save_plot(p3, "03_grouped_f1_bars.png", width = 10, height = 6)

# -----------------------------
# 8) Graph 4: Accuracy vs Error Rate
# -----------------------------
p4 <- ggplot(metrics_df, aes(x = error_rate, y = accuracy, label = model_language, shape = language)) +
  geom_point(size = 4) +
  geom_text(nudge_y = 0.015, size = 3.5, check_overlap = TRUE) +
  labs(
    title = "Accuracy vs Error Rate",
    x = "Error Rate",
    y = "Accuracy",
    shape = "Language"
  ) +
  xlim(0, max(metrics_df$error_rate, na.rm = TRUE) * 1.1 + 0.01) +
  ylim(0, 1) +
  theme_minimal(base_size = 12)

save_plot(p4, "04_accuracy_vs_error_rate.png", width = 10, height = 6)

# -----------------------------
# 9) Graph 5: False Positives vs False Negatives
# -----------------------------
p5 <- ggplot(full_df, aes(x = false_positives, y = false_negatives, label = model_language, shape = language)) +
  geom_point(size = 4) +
  geom_text(nudge_y = 20, size = 3.5, check_overlap = TRUE) +
  labs(
    title = "False Positives vs False Negatives",
    x = "False Positives",
    y = "False Negatives",
    shape = "Language"
  ) +
  theme_minimal(base_size = 12)

save_plot(p5, "05_fp_vs_fn.png", width = 10, height = 6)

# -----------------------------
# 10) Graph 6: Average metric by model
# -----------------------------
avg_model_df <- full_df %>%
  group_by(model) %>%
  summarise(
    precision = mean(precision, na.rm = TRUE),
    recall = mean(recall, na.rm = TRUE),
    f1_score = mean(f1_score, na.rm = TRUE),
    accuracy = mean(accuracy, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_longer(
    cols = c(precision, recall, f1_score, accuracy),
    names_to = "metric",
    values_to = "value"
  )

p6 <- ggplot(avg_model_df, aes(x = model, y = value, fill = metric)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(
    aes(label = sprintf("%.3f", value)),
    position = position_dodge(width = 0.8),
    vjust = -0.35,
    size = 3.1
  ) +
  labs(
    title = "Average Metrics by Model",
    x = "Model",
    y = "Average Score",
    fill = "Metric"
  ) +
  ylim(0, 1.05) +
  theme_minimal(base_size = 12)

save_plot(p6, "06_average_metrics_by_model.png", width = 11, height = 6)

# -----------------------------
# 11) Bonus Graph 7: Coverage / unclear / error composition
# -----------------------------
comp_df <- metrics_df %>%
  transmute(
    model_language,
    language,
    Classified = coverage_rate,
    Errors = error_rate,
    Unclear = unclear_rate
  ) %>%
  pivot_longer(
    cols = c(Classified, Errors, Unclear),
    names_to = "bucket",
    values_to = "rate"
  )

p7 <- ggplot(comp_df, aes(x = fct_reorder(model_language, rate, .fun = sum), y = rate, fill = bucket)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Classification Coverage Composition",
    x = "Run",
    y = "Rate",
    fill = "Bucket"
  ) +
  theme_minimal(base_size = 12)

save_plot(p7, "07_coverage_composition.png", width = 11, height = 8)

cat("\nDone.\n")
cat("CSV:", metrics_csv, "\n")
cat("Graphs folder:", graphs_dir, "\n")
