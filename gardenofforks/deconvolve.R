#!/usr/bin/env Rscript
# Batterie de déconvolution appelée par deconv.py (section 8 du pipeline).
#   - méthodes SANS référence (immunedeconv) : mcp_counter, xcell, quantiseq, epic
#   - méthodes AVEC référence single-cell (omnideconv) : dwls, bayesprism
# Entrées : --bulk counts.tsv (genes x samples), --config config.json, --outdir dir
# Sorties : deconv_<method>.csv (cell types x samples) par méthode réussie.

suppressMessages({library(jsonlite)})

args <- commandArgs(trailingOnly = TRUE)
getarg <- function(flag) { i <- which(args == flag); if (length(i)) args[i + 1] else NA }
bulk_path <- getarg("--bulk"); cfg_path <- getarg("--config"); outdir <- getarg("--outdir")

cfg <- fromJSON(cfg_path, simplifyVector = FALSE)
methods <- cfg$methods
enabled <- function(m) !is.null(methods[[m]]) && isTRUE(methods[[m]]$enabled)
params_of <- function(m) { p <- methods[[m]]; p$enabled <- NULL; p }

bulk <- as.matrix(read.table(bulk_path, header = TRUE, row.names = 1,
                             sep = "\t", check.names = FALSE))
storage.mode(bulk) <- "double"
cpm <- sweep(bulk, 2, pmax(colSums(bulk), 1), "/") * 1e6      # linéaire, pour immunedeconv

# n'appelle `fn` qu'avec les paramètres qu'elle accepte réellement (harmonise
# les nomenclatures : l'utilisateur met ce qu'il veut, on filtre) --------------
call_filtered <- function(fn, params, ...) {
  fixed <- list(...)
  valid <- intersect(names(params), names(formals(fn)))
  do.call(fn, c(fixed, params[valid]))
}

write_res <- function(mat, method) {
  df <- as.data.frame(as.matrix(mat))
  write.csv(df, file.path(outdir, paste0("deconv_", method, ".csv")))
  cat(sprintf("OK   %-12s -> %d types x %d tumeurs\n", method, nrow(df), ncol(df)))
}

# --------------------------------------------------------------------------
# 1. Méthodes sans référence (immunedeconv)
# --------------------------------------------------------------------------
first_gen <- c("mcp_counter", "xcell", "quantiseq", "epic")
if (any(sapply(first_gen, enabled))) suppressMessages(library(immunedeconv))
for (meth in first_gen) {
  if (!enabled(meth)) next
  tryCatch({
    res <- call_filtered(immunedeconv::deconvolute, params_of(meth),
                         gene_expression = cpm, method = meth)
    m <- as.data.frame(res[, -1, drop = FALSE]); rownames(m) <- res$cell_type
    write_res(m, meth)
  }, error = function(e) cat(sprintf("FAIL %-12s : %s\n", meth, conditionMessage(e))))
}

# --------------------------------------------------------------------------
# 2. Méthodes avec référence single-cell (omnideconv)
# --------------------------------------------------------------------------
second_gen <- c("dwls", "bayesprism")
ref <- cfg$reference
if (any(sapply(second_gen, enabled)) && !is.null(ref) && !is.null(ref$path)) {
  if (!file.exists(ref$path)) {
    cat(sprintf("SKIP second-gen : référence introuvable (%s)\n", ref$path))
  } else {
    cat(sprintf("Chargement de la référence single-cell (%s)…\n", basename(ref$path)))
    obj <- readRDS(ref$path)
    suppressMessages(library(Seurat))
    meta <- obj@meta.data
    ct_col <- ref$celltype_col; bt_col <- ref$batch_col
    ct_all <- as.character(meta[[ct_col]])
    keep <- !is.na(ct_all) & ct_all != ""
    # sous-échantillonnage par type cellulaire (tractabilité DWLS/BayesPrism)
    maxc <- if (!is.null(ref$max_cells_per_type)) as.integer(ref$max_cells_per_type) else 200L
    set.seed(0)
    idx <- unlist(lapply(split(which(keep), ct_all[keep]), function(ii)
      if (length(ii) > maxc) sample(ii, maxc) else ii))
    cat(sprintf("Référence : %d cellules retenues (<= %d / type), %d types.\n",
                length(idx), maxc, length(unique(ct_all[idx]))))
    sc <- tryCatch(as.matrix(SeuratObject::LayerData(obj, assay = "RNA", layer = "counts")[, idx]),
                   error = function(e)
                     as.matrix(Seurat::GetAssayData(obj, assay = "RNA", slot = "counts")[, idx]))
    ct <- ct_all[idx]
    bt <- if (!is.null(bt_col)) as.character(meta[[bt_col]])[idx] else rep("b1", length(idx))
    rm(obj); gc()

    suppressMessages(library(omnideconv))
    for (meth in second_gen) {
      if (!enabled(meth)) next
      cat(sprintf("Déconvolution %s (référence single-cell)…\n", meth))
      tryCatch({
        p <- params_of(meth)
        model <- tryCatch(call_filtered(
          omnideconv::build_model, p, single_cell_object = sc,
          cell_type_annotations = ct, method = meth, batch_ids = bt,
          bulk_gene_expression = bulk), error = function(e) NULL)
        dec <- call_filtered(
          omnideconv::deconvolute, p, bulk_gene_expression = bulk, model = model,
          method = meth, single_cell_object = sc, cell_type_annotations = ct,
          batch_ids = bt)
        write_res(t(as.matrix(dec)), meth)     # samples x types -> types x samples
      }, error = function(e) cat(sprintf("FAIL %-12s : %s\n", meth, conditionMessage(e))))
    }
  }
}
cat("deconvolve.R terminé.\n")
