# Shared I/O for the R reference runs.
read_ldac <- function(path) {
  lines <- readLines(path)
  lapply(strsplit(lines, " ", fixed = TRUE), function(f) {
    if (length(f) < 2) return(matrix(0L, 2, 0))
    kv <- do.call(rbind, strsplit(f[-1], ":", fixed = TRUE))
    rbind(as.integer(kv[, 1]), as.integer(kv[, 2]))          # row 1: 0-based term id, row 2: count
  })
}
to_triplet <- function(docs, V) {
  slam::simple_triplet_matrix(i = rep(seq_along(docs), vapply(docs, ncol, 1L)),
                              j = unlist(lapply(docs, function(d) d[1, ] + 1L)),
                              v = unlist(lapply(docs, function(d) d[2, ])), nrow = length(docs), ncol = V)
}
# phi: K x V. Written row-major as float64, like numpy's tofile.
save_result <- function(prefix, phi, meta, alpha = NULL) {
  con <- file(paste0(prefix, ".phi.bin"), "wb"); writeBin(as.vector(t(phi)), con, size = 8); close(con)
  meta$K <- nrow(phi); meta$V <- ncol(phi)
  fields <- vapply(names(meta), function(n) {
    v <- meta[[n]]; sprintf("\"%s\": %s", n, if (is.character(v)) sprintf("\"%s\"", v) else format(v, digits = 12))
  }, "")
  if (!is.null(alpha)) fields <- c(fields, sprintf("\"alpha\": [%s]", paste(format(alpha, digits = 10), collapse = ", ")))
  writeLines(paste0("{", paste(fields, collapse = ", "), "}"), paste0(prefix, ".json"))
}
args_list <- function() {
  a <- commandArgs(trailingOnly = TRUE); out <- list()
  for (i in seq(1, length(a), by = 2)) out[[sub("^-+", "", a[i])]] <- a[i + 1]
  out
}
