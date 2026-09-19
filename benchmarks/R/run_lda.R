# LDA reference runs in R: `lda` (collapsed Gibbs, C), topicmodels Gibbs (C++) and VEM (Blei's lda-c).
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
a <- args_list(); K <- as.integer(a$K); iters <- as.integer(a$iters); seed <- as.integer(a$seed)
alpha <- as.numeric(a$alpha); eta <- as.numeric(a$eta)
docs <- read_ldac(a$train); vocab <- readLines(a$vocab); V <- length(vocab)
if (a$impl == "lda") {
  library(lda); set.seed(seed)
  d <- lapply(docs, function(m) { storage.mode(m) <- "integer"; m })
  t <- system.time(fit <- lda.collapsed.gibbs.sampler(d, K, vocab, iters, alpha = alpha, eta = eta))[["elapsed"]]
  phi <- (fit$topics + eta) / rowSums(fit$topics + eta)
  save_result(a$out, phi, list(impl = "R lda", seconds = t, iters = iters, version = as.character(packageVersion("lda"))), rep(alpha, K))
} else {
  library(topicmodels); X <- to_triplet(docs, V); colnames(X) <- vocab
  if (a$impl == "topicmodels_gibbs") {
    ctl <- list(alpha = alpha, delta = eta, iter = iters, burnin = 0, thin = iters, seed = seed, best = TRUE)
    t <- system.time(fit <- LDA(X, K, method = "Gibbs", control = ctl))[["elapsed"]]
    save_result(a$out, exp(fit@beta), list(impl = "R topicmodels Gibbs", seconds = t, iters = iters, version = as.character(packageVersion("topicmodels"))), rep(alpha, K))
  } else {
    ctl <- list(alpha = alpha, estimate.alpha = TRUE, seed = seed, em = list(iter.max = iters, tol = 1e-4), var = list(iter.max = 100, tol = 1e-6))
    t <- system.time(fit <- LDA(X, K, method = "VEM", control = ctl))[["elapsed"]]
    save_result(a$out, exp(fit@beta), list(impl = "R topicmodels VEM", seconds = t, iters = fit@iter, version = as.character(packageVersion("topicmodels"))), rep(fit@alpha, K))
  }
}
cat(a$impl, "done in", round(t, 2), "s\n")
