#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(illuminaio))
suppressPackageStartupMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop(
    paste(
      "Usage: Rscript 02_build_expression.R",
      "<CEDAR_selected_samples.csv> <idat_dir> <processed_dir>",
      "<p_list_comma_separated>"
    )
  )
}

sample_file <- normalizePath(args[[1]], mustWork = TRUE)
idat_dir <- normalizePath(args[[2]], mustWork = TRUE)
processed_dir <- args[[3]]
p_list <- as.integer(strsplit(args[[4]], ",", fixed = TRUE)[[1]])
if (anyNA(p_list) || any(p_list <= 0)) {
  stop("p_list must contain positive integers, e.g. 3000,5000")
}
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

samples <- fread(sample_file)
required <- c("donor", "cell_type", "idat")
if (!all(required %in% names(samples))) {
  stop("Selected sample table must contain donor, cell_type and idat columns.")
}
if (anyDuplicated(samples$idat)) {
  stop("CEDAR_selected_samples.csv contains a duplicated IDAT filename.")
}

files <- file.path(idat_dir, samples$idat)
missing_files <- files[!file.exists(files)]
if (length(missing_files) > 0) {
  stop("Missing IDAT files; first missing file: ", missing_files[[1]])
}

cat("Reading", length(files), "selected IDAT arrays...\n")
first <- readIDAT(files[[1]])
probe_id <- as.character(first$Quants$CodesBinData)
expr <- matrix(
  NA_real_,
  nrow = length(probe_id),
  ncol = length(files),
  dimnames = list(probe_id, samples$idat)
)

for (i in seq_along(files)) {
  dat <- readIDAT(files[[i]])
  codes <- as.character(dat$Quants$CodesBinData)
  values <- as.numeric(dat$Quants$MeanBinData)

  if (!identical(codes, probe_id)) {
    ord <- match(probe_id, codes)
    if (anyNA(ord)) {
      stop("Probe codes differ irreconcilably in array: ", files[[i]])
    }
    values <- values[ord]
  }
  if (length(values) != length(probe_id) || any(!is.finite(values))) {
    stop("Invalid intensity vector in array: ", files[[i]])
  }
  expr[, i] <- values

  if (i %% 25 == 0 || i == length(files)) {
    cat("  completed", i, "/", length(files), "\n")
  }
}

cat("Applying log2(x+1) and quantile normalization...\n")
expr_log <- log2(pmax(expr, 0) + 1)
expr_norm <- normalizeBetweenArrays(expr_log, method = "quantile")
rownames(expr_norm) <- probe_id
colnames(expr_norm) <- samples$idat

probe_var <- apply(expr_norm, 1, var)
valid <- is.finite(probe_var) & probe_var > 0
expr_norm <- expr_norm[valid, , drop = FALSE]
probe_var <- probe_var[valid]
ranked <- order(probe_var, decreasing = TRUE)

if (max(p_list) > nrow(expr_norm)) {
  stop(
    "Requested p exceeds the number of finite nonconstant probes: ",
    nrow(expr_norm)
  )
}

saveRDS(
  list(expression = expr_norm, metadata = samples, probe_variance = probe_var),
  file.path(processed_dir, "CEDAR_normalized_expression.rds")
)

for (p in p_list) {
  idx <- ranked[seq_len(p)]
  expr_p <- expr_norm[idx, , drop = FALSE]

  expr_file <- file.path(processed_dir, paste0("CEDAR_expression_p", p, ".csv.gz"))
  meta_file <- file.path(processed_dir, paste0("CEDAR_metadata_p", p, ".csv"))
  probe_file <- file.path(processed_dir, paste0("CEDAR_selected_probes_p", p, ".csv"))

  fwrite(as.data.table(expr_p, keep.rownames = "probe_id"), expr_file)
  fwrite(samples, meta_file)
  fwrite(
    data.table(
      probe_id = rownames(expr_p),
      normalized_variance = probe_var[idx],
      variance_rank = seq_len(p)
    ),
    probe_file
  )

  cat("Saved p =", p, "expression:", expr_file, "\n")
  cat("Saved p =", p, "metadata:", meta_file, "\n")
}
