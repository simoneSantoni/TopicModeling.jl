# CTM reference run: topicmodels::CTM, which wraps Blei & Lafferty's ctm-c variational EM.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
library(topicmodels)
a <- args_list(); K <- as.integer(a$K)
docs <- read_ldac(a$train); vocab <- readLines(a$vocab); X <- to_triplet(docs, length(vocab)); colnames(X) <- vocab
ctl <- list(seed = as.integer(a$seed), em = list(iter.max = as.integer(a$iters), tol = 1e-5), var = list(iter.max = 500, tol = 1e-6), cg = list(iter.max = 500, tol = 1e-5))
t <- system.time(fit <- CTM(X, K, method = "VEM", control = ctl))[["elapsed"]]
save_result(a$out, exp(fit@beta), list(impl = "R topicmodels CTM (ctm-c)", seconds = t, iters = fit@iter, loglik = sum(fit@loglikelihood), version = as.character(packageVersion("topicmodels"))))
cat("topicmodels CTM done in", round(t, 2), "s,", fit@iter, "EM iterations\n")
