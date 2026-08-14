MIN_CELLTYPES <- 2
MIN_TRANSCRIPTS <- 2

# ------------------------------------------------------------------------------
# JS divergence for one transcript across cell types
# Higher value means stronger cell-type specificity
# ------------------------------------------------------------------------------
calculate_js <- function(x, min_valid = 2) {
  x <- x[!is.na(x)]

  if (length(x) < min_valid) return(NA_real_)
  if (all(x == 0)) return(0)

  s <- sum(x)
  if (s == 0) return(0)

  p <- x / s
  n <- length(p)
  q <- rep(1 / n, n)
  m <- 0.5 * (p + q)

  kl <- function(a, b) {
    idx <- a > 0 & b > 0
    if (!any(idx)) return(0)
    sum(a[idx] * log2(a[idx] / b[idx]))
  }

  js <- 0.5 * kl(p, m) + 0.5 * kl(q, m)
  max(js, 0)
}

# ------------------------------------------------------------------------------
# Detect dominant transcript switching within one gene
# ------------------------------------------------------------------------------
detect_switch_js <- function(gene_psi, gene_expr, tol = 1e-9) {
  n_total_celltypes <- ncol(gene_psi)
  tx_names <- rownames(gene_psi)

  valid_ct <- colSums(gene_expr, na.rm = TRUE) > 0
  n_valid_celltypes <- sum(valid_ct)
  n_filtered_celltypes <- n_total_celltypes - n_valid_celltypes

  empty_res <- list(
    n_total_celltypes = n_total_celltypes,
    n_valid_celltypes = n_valid_celltypes,
    n_filtered_celltypes = n_filtered_celltypes,
    n_unique_dominant = NA_integer_,
    n_valid_celltypes_switching = NA_integer_,
    mean_dominant_psi = NA_real_,
    dominant_transcripts = NA_character_,
    dominant_distribution = NA_character_,
    mean_js = NA_real_,
    sum_js = NA_real_,
    score_js = NA_real_
  )

  if (n_valid_celltypes < 2) return(empty_res)

  gene_psi <- gene_psi[, valid_ct, drop = FALSE]
  n_ct <- ncol(gene_psi)
  ct_names <- colnames(gene_psi)
  n_tx <- nrow(gene_psi)

  dominant_share <- matrix(
    0,
    nrow = n_tx,
    ncol = n_ct,
    dimnames = list(tx_names, ct_names)
  )

  dominant_psi <- numeric(n_ct)
  valid_flags <- logical(n_ct)

  for (j in seq_len(n_ct)) {
    v <- gene_psi[, j]

    if (all(is.na(v)) || all(v == 0, na.rm = TRUE)) {
      dominant_psi[j] <- NA_real_
      valid_flags[j] <- FALSE
      next
    }

    valid_flags[j] <- TRUE
    mx <- max(v, na.rm = TRUE)
    dominant_psi[j] <- mx

    idx <- which(!is.na(v) & abs(v - mx) < tol)
    dominant_share[idx, j] <- 1 / length(idx)
  }

  dominant_counts <- rowSums(dominant_share)
  n_valid_switch <- sum(valid_flags)

  if (n_valid_switch < 2) {
    empty_res$n_valid_celltypes_switching <- n_valid_switch
    return(empty_res)
  }

  qualified_tx <- names(dominant_counts)[dominant_counts >= 1]
  n_unique <- length(qualified_tx)
  mean_dominant_psi <- mean(dominant_psi[valid_flags], na.rm = TRUE)

  nonzero_tx <- names(dominant_counts)[dominant_counts > 0]
  dominant_dist <- paste(
    paste0(nonzero_tx, ":", round(dominant_counts[nonzero_tx], 2)),
    collapse = ";"
  )

  if (n_unique <= 1) {
    return(list(
      n_total_celltypes = n_total_celltypes,
      n_valid_celltypes = n_valid_celltypes,
      n_filtered_celltypes = n_filtered_celltypes,
      n_unique_dominant = n_unique,
      n_valid_celltypes_switching = n_valid_switch,
      mean_dominant_psi = mean_dominant_psi,
      dominant_transcripts = if (n_unique == 1) qualified_tx else NA_character_,
      dominant_distribution = dominant_dist,
      mean_js = NA_real_,
      sum_js = NA_real_,
      score_js = 0
    ))
  }

  qualified_counts <- dominant_counts[qualified_tx]
  dominant_props <- qualified_counts / sum(qualified_counts)

  js_values <- sapply(qualified_tx, function(tx) {
    calculate_js(gene_psi[tx, valid_flags])
  })
  js_values[is.na(js_values)] <- 0

  score_js <- -sum(js_values * dominant_props * log2(dominant_props))

  list(
    n_total_celltypes = n_total_celltypes,
    n_valid_celltypes = n_valid_celltypes,
    n_filtered_celltypes = n_filtered_celltypes,
    n_unique_dominant = n_unique,
    n_valid_celltypes_switching = n_valid_switch,
    mean_dominant_psi = mean_dominant_psi,
    dominant_transcripts = paste(qualified_tx, collapse = ";"),
    dominant_distribution = dominant_dist,
    mean_js = mean(js_values),
    sum_js = sum(js_values),
    score_js = score_js
  )
}

# ------------------------------------------------------------------------------
# Calculate gene-level DTU metrics
# psi_matrix  : transcript x celltype PAI matrix
# expr_matrix : transcript x celltype expression matrix
# tx_to_gene  : data frame with transcript_id and gene_id
# ------------------------------------------------------------------------------
calculate_dtu_scores <- function(
    psi_matrix,
    expr_matrix,
    tx_to_gene,
    min_celltypes = 2,
    min_transcripts = 2
) {
  stopifnot(!is.null(rownames(psi_matrix)), !is.null(rownames(expr_matrix)))

  colnames(tx_to_gene) <- c("transcript_id", "gene_id")

  common_tx <- Reduce(intersect, list(
    rownames(psi_matrix),
    rownames(expr_matrix),
    tx_to_gene$transcript_id
  ))

  psi_matrix <- psi_matrix[common_tx, , drop = FALSE]
  expr_matrix <- expr_matrix[common_tx, , drop = FALSE]
  tx_to_gene <- tx_to_gene[match(common_tx, tx_to_gene$transcript_id), , drop = FALSE]

  genes <- unique(tx_to_gene$gene_id)
  res_list <- vector("list", length(genes))

  for (i in seq_along(genes)) {
    g <- genes[i]
    tx <- tx_to_gene$transcript_id[tx_to_gene$gene_id == g]
    n_tx <- length(tx)

    out <- data.frame(
      gene_id = g,
      n_transcripts = n_tx,
      n_total_celltypes = ncol(psi_matrix),
      n_valid_celltypes = NA_integer_,
      n_filtered_celltypes = NA_integer_,
      n_unique_dominant = NA_integer_,
      n_valid_celltypes_switching = NA_integer_,
      mean_dominant_psi = NA_real_,
      dominant_transcripts = NA_character_,
      dominant_distribution = NA_character_,
      mean_js = NA_real_,
      sum_js = NA_real_,
      score_js = NA_real_,
      stringsAsFactors = FALSE
    )

    if (n_tx < min_transcripts) {
      res_list[[i]] <- out
      next
    }

    gene_expr <- expr_matrix[tx, , drop = FALSE]
    gene_psi <- psi_matrix[tx, , drop = FALSE]

    n_valid <- sum(colSums(gene_expr, na.rm = TRUE) > 0)
    if (n_valid < min_celltypes) {
      out$n_valid_celltypes <- n_valid
      out$n_filtered_celltypes <- ncol(expr_matrix) - n_valid
      res_list[[i]] <- out
      next
    }

    stat <- detect_switch_js(gene_psi, gene_expr)

    out$n_total_celltypes <- stat$n_total_celltypes
    out$n_valid_celltypes <- stat$n_valid_celltypes
    out$n_filtered_celltypes <- stat$n_filtered_celltypes
    out$n_unique_dominant <- stat$n_unique_dominant
    out$n_valid_celltypes_switching <- stat$n_valid_celltypes_switching
    out$mean_dominant_psi <- stat$mean_dominant_psi
    out$dominant_transcripts <- stat$dominant_transcripts
    out$dominant_distribution <- stat$dominant_distribution
    out$mean_js <- stat$mean_js
    out$sum_js <- stat$sum_js
    out$score_js <- stat$score_js

    res_list[[i]] <- out
  }

  do.call(rbind, res_list)
}

# ------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------
run_dtu_js <- function(
    psi_matrix = mat_tx_pai_190_celltype_filter,
    expr_matrix = mat_tx_raw_sum_190_celltype_filter,
    tx_to_gene = tx_meta[, c("new_transcript_symbol", "new_gene_id")],
    min_celltypes = 2,
    min_transcripts = 2,
    output_csv = NULL
) {
  res <- calculate_dtu_scores(
    psi_matrix = psi_matrix,
    expr_matrix = expr_matrix,
    tx_to_gene = tx_to_gene,
    min_celltypes = min_celltypes,
    min_transcripts = min_transcripts
  )

  if (!is.null(output_csv)) {
    write.csv(res, output_csv, row.names = FALSE)
  }

  res
}

# ------------------------------------------------------------------------------
# Run
# ------------------------------------------------------------------------------
dtu_result_df <- run_dtu_js(
  psi_matrix = mat_tx_pai_190_celltype_filter,
  expr_matrix = mat_tx_nor_mean_190_celltype_filter,
  tx_to_gene = tx_meta[, c("new_transcript_symbol", "new_gene_id")],
  min_celltypes = 2,
  min_transcripts = 2,
  output_csv = "../data/DTU_JS_scores.csv"
)
