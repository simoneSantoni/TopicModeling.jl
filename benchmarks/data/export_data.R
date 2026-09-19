# Export the reference corpora shipped with R packages into plain files that the Julia,
# Python and R benchmark scripts all read, so every implementation sees identical input.
#   AP          2246 Associated Press articles (Blei et al. 2003; topicmodels::AssociatedPress)
#   poliblog5k  5000 political blog posts, 2008, with rating + day (stm)
#   gadarian    341 open-ended survey responses with treatment + pid_rep (stm)
suppressPackageStartupMessages({library(topicmodels); library(stm); library(slam)})
out <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(out)) out <- "."

write_ldac <- function(i, j, v, D, path) {
  o <- order(i, j); i <- i[o]; j <- j[o]; v <- v[o]
  tok <- paste0(j - 1L, ":", v)
  lines <- vapply(split(tok, factor(i, levels = seq_len(D))),
                  function(x) paste(c(length(x), x), collapse = " "), "")
  writeLines(lines, path)
}

data("AssociatedPress", package = "topicmodels")
ap <- AssociatedPress
write_ldac(ap$i, ap$j, ap$v, nrow(ap), file.path(out, "ap.ldac"))
writeLines(colnames(ap), file.path(out, "ap.vocab"))

stm_to_ldac <- function(docs, path) {
  D <- length(docs)
  i <- rep(seq_len(D), vapply(docs, ncol, 1L))
  j <- unlist(lapply(docs, function(d) d[1, ]))
  v <- unlist(lapply(docs, function(d) d[2, ]))
  write_ldac(i, j, v, D, path)
}

data("poliblog5k", package = "stm")
stm_to_ldac(poliblog5k.docs, file.path(out, "poliblog5k.ldac"))
writeLines(poliblog5k.voc, file.path(out, "poliblog5k.vocab"))
write.csv(poliblog5k.meta[, c("rating", "day", "blog")], file.path(out, "poliblog5k.meta.csv"), row.names = FALSE)

data("gadarian", package = "stm")
g <- textProcessor(gadarian$open.ended.response, metadata = gadarian, verbose = FALSE)
p <- prepDocuments(g$documents, g$vocab, g$meta, verbose = FALSE)
stm_to_ldac(p$documents, file.path(out, "gadarian.ldac"))
writeLines(p$vocab, file.path(out, "gadarian.vocab"))
write.csv(p$meta[, c("treatment", "pid_rep")], file.path(out, "gadarian.meta.csv"), row.names = FALSE)
cat("AP", dim(ap), "| poliblog5k", length(poliblog5k.docs), length(poliblog5k.voc),
    "| gadarian", length(p$documents), length(p$vocab), "\n")
