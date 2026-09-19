# STM reference run: stm with its defaults (spectral initialisation, emtol 1e-5, pooled prevalence prior).
# The prevalence design matrix is built here (rating + b-spline of day) and exported, so that the Julia
# implementation is given exactly the same covariates.
source(file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))), "common.R"))
suppressPackageStartupMessages(library(stm))
a <- args_list(); K <- as.integer(a$K)
docs <- lapply(read_ldac(a$train), function(m) { m[1, ] <- m[1, ] + 1L; storage.mode(m) <- "integer"; m })
vocab <- readLines(a$vocab)
meta <- read.csv(a$meta)[as.integer(readLines(a$idx)), , drop = FALSE]
X <- model.matrix(~ rating + splines::bs(day, df = 5), data = meta)
write.table(X[, -1, drop = FALSE], a$xout, sep = ",", row.names = FALSE, col.names = FALSE)
t <- system.time(fit <- stm(docs, vocab, K, prevalence = X[, -1, drop = FALSE], init.type = "Spectral", max.em.its = as.integer(a$iters), verbose = FALSE, seed = as.integer(a$seed)))[["elapsed"]]
phi <- exp(fit$beta$logbeta[[1]])
save_result(a$out, phi, list(impl = "R stm", seconds = t, iters = fit$convergence$its, bound = tail(fit$convergence$bound, 1), converged = as.integer(fit$convergence$converged), version = as.character(packageVersion("stm"))))
write.table(fit$theta, paste0(a$out, ".theta.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
set.seed(1); ee <- estimateEffect(1:K ~ rating, fit, metadata = meta, uncertainty = "Global", nsims = 25)
eff <- t(vapply(summary(ee)$tables, function(tb) tb[2, 1:2], c(0, 0)))
write.table(eff, paste0(a$out, ".effect.csv"), sep = ",", row.names = FALSE, col.names = FALSE)
cat("stm done in", round(t, 2), "s,", fit$convergence$its, "EM iterations, bound", tail(fit$convergence$bound, 1), "\n")
