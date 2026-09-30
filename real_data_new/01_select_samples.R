#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop(
    "Usage: Rscript 01_select_samples.R <E-MTAB-6667.sdrf.txt> <processed_dir>"
  )
}

sdrf_file <- normalizePath(args[[1]], mustWork = TRUE)
processed_dir <- args[[2]]
dir.create(processed_dir, recursive = TRUE, showWarnings = FALSE)

selected_file <- file.path(processed_dir, "CEDAR_selected_samples.csv")
audit_file <- file.path(processed_dir, "CEDAR_cell_type_audit.csv")
context_file <- file.path(processed_dir, "CEDAR_cell_label_context.csv")

target_celltypes <- c(
  "B cell",
  "CD14-positive monocyte",
  "CD15 positive leukocyte",
  "platelet",
  "T cell",
  "thymocyte"
)

sdrf <- fread(
  sdrf_file,
  sep = "\t",
  header = TRUE,
  data.table = TRUE,
  check.names = FALSE
)

required <- c(
  "Characteristics[individual]",
  "Characteristics[cell type]",
  "Array Data File"
)
missing_cols <- setdiff(required, names(sdrf))
if (length(missing_cols) > 0) {
  stop("Missing SDRF columns: ", paste(missing_cols, collapse = ", "))
}

donor_col <- "Characteristics[individual]"
cell_col <- "Characteristics[cell type]"
idat_col <- "Array Data File"
organism_col <- "Characteristics[organism part]"

audit_by <- c(cell_col)
if (organism_col %in% names(sdrf)) {
  audit_by <- c(audit_by, organism_col)
}

audit <- sdrf[, .(
  n_arrays = .N,
  n_donors = uniqueN(get(donor_col))
), by = audit_by]
setnames(audit, cell_col, "raw_cell_type")
if (organism_col %in% names(audit)) {
  setnames(audit, organism_col, "organism_part")
}
setorder(audit, raw_cell_type)
fwrite(audit, audit_file)

context_cols <- names(sdrf)[grepl(
  "cell|marker|phenotype|organism part|description",
  names(sdrf),
  ignore.case = TRUE
)]
context_cols <- unique(c(cell_col, context_cols))
cell_context <- unique(
  sdrf[get(cell_col) %chin% target_celltypes, ..context_cols]
)
setnames(cell_context, cell_col, "raw_cell_type")
setorder(cell_context, raw_cell_type)
fwrite(cell_context, context_file)

sub <- sdrf[get(cell_col) %chin% target_celltypes]
if (nrow(sub) == 0) {
  stop("No target immune-cell samples were found in the SDRF.")
}

selected <- data.table(
  donor = sub[[donor_col]],
  cell_type = sub[[cell_col]],
  idat = basename(sub[[idat_col]])
)
if (organism_col %in% names(sub)) {
  selected[, organism_part := sub[[organism_col]]]
} else {
  selected[, organism_part := NA_character_]
}

# Remove only exact duplicate SDRF records.  Distinct technical-replicate IDATs
# are retained and will be averaged after joint normalization.
setorder(selected, donor, cell_type, idat)
n_before <- nrow(selected)
selected <- unique(selected, by = c("donor", "cell_type", "idat"))
n_exact_duplicates_removed <- n_before - nrow(selected)

presence <- dcast(
  selected,
  donor ~ cell_type,
  fun.aggregate = length,
  value.var = "idat"
)
missing_target_cols <- setdiff(target_celltypes, names(presence))
if (length(missing_target_cols) > 0) {
  stop(
    "The following target cell types are absent: ",
    paste(missing_target_cols, collapse = ", ")
  )
}

complete_mask <- apply(
  presence[, ..target_celltypes],
  1,
  function(x) all(x >= 1)
)
complete_donors <- presence$donor[complete_mask]
selected <- selected[donor %chin% complete_donors]
selected[, cell_index := match(cell_type, target_celltypes)]
selected[, pair_replicates := .N, by = .(donor, cell_type)]
selected[, technical_replicate := seq_len(.N), by = .(donor, cell_type)]
setorder(selected, donor, cell_index, technical_replicate, idat)

unique_pairs <- unique(selected[, .(donor, cell_type)])
counts <- selected[, .(
  n_arrays = .N,
  n_donors = uniqueN(donor),
  donor_cell_pairs_with_replicates = uniqueN(donor[pair_replicates > 1]),
  max_replicates_per_pair = max(pair_replicates)
), by = cell_type]
if (
  nrow(counts) != length(target_celltypes) ||
  nrow(unique_pairs) != length(complete_donors) * length(target_celltypes) ||
  any(counts$n_donors != length(complete_donors))
) {
  stop("The selected sample table is not a complete donor-by-cell grid.")
}

fwrite(selected, selected_file)

cat("SDRF:", sdrf_file, "\n")
cat("Selected table:", selected_file, "\n")
cat("Cell-type audit:", audit_file, "\n")
cat("Cell-label context:", context_file, "\n")
cat("Complete donors:", length(complete_donors), "\n")
cat("Selected arrays:", nrow(selected), "\n")
cat("Donor-cell pairs:", nrow(unique_pairs), "\n")
cat("Extra technical-replicate arrays retained:", nrow(selected) - nrow(unique_pairs), "\n")
cat("Exact duplicate SDRF rows removed:", n_exact_duplicates_removed, "\n")
print(counts)
cat(
  "\nIMPORTANT: inspect CEDAR_cell_type_audit.csv and verify how the raw labels\n",
  "'T cell' and 'thymocyte' map to CD4+ and CD8+ T-cell samples before\n",
  "writing the biological labels in the manuscript.\n",
  sep = ""
)
