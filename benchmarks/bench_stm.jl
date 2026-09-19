# STM on poliblog5k (4500 train / 500 test blog posts, K = 20, prevalence ~ rating + bs(day, 5)):
# TopicModeling.jl vs the R package stm, given the identical design matrix (written by R/run_stm.R).
include(joinpath(@__DIR__, "common.jl"))
using DelimitedFiles, LinearAlgebra
train = load("poliblog5k.train"); test = load("poliblog5k.test"); K = 20
X = readdlm(joinpath(RESULTS, "stm_poliblog_X.csv"), ',', Float64)
fit(STM, train[1:300], 5; prevalence=X[1:300, :], iters=2)                        # compile
rphi, rmeta = read_phi(joinpath(RESULTS, "stm_poliblog_R"))
rows = [("stm", "R/C++", 1, metafield(rmeta, "seconds"), Int(metafield(rmeta, "iters")), metafield(rmeta, "bound"), score(rphi, 0.1, train, test))]
models = Dict{Int,STM}()
for T in (1, 8)
    m = fit(STM, train, K; prevalence=X, nthreads=T, keep_nu=true)
    models[T] = m
    pushfirst!(rows, ("**TopicModeling.jl**", "Julia", T, m.elapsed, m.iterations, m.trace[end], score(m.phi, 0.1, train, test)))
end
m = models[1]
open(joinpath(RESULTS, "stm_poliblog.md"), "w") do io
    println(io, "| implementation | language | threads | time (s) | EM iterations | final bound | held-out perplexity | NPMI |\n|---|---|---|---|---|---|---|---|")
    for (name, lang, T, secs, its, bound, s) in rows
        @printf(io, "| %s | %s | %d | %.1f | %d | %.0f | %.0f ± %.0f | %.3f |\n", name, lang, T, secs, its, bound, s.perplexity, s.perplexity_sd, s.npmi)
    end
    # Do the two implementations find the same model? Match topics, compare top words, θ and covariate effects.
    perm, dist = match_topics(rphi, m.phi)
    jt, rt = topword_sets(m.phi[perm, :]), topword_sets(rphi)
    overlap = [length(intersect(jt[k], rt[k])) / 10 for k in 1:K]
    rtheta = readdlm(joinpath(RESULTS, "stm_poliblog_R.theta.csv"), ',', Float64)
    θcor = [cor(m.theta[:, perm[k]], rtheta[:, k]) for k in 1:K]
    reff = readdlm(joinpath(RESULTS, "stm_poliblog_R.effect.csv"), ',', Float64)
    coef, se = estimate_effect(m; nsims=25, rng=Xoshiro(1))
    jeff = coef[2, perm]
    @printf(io, "\nAgreement between the two fitted models (topics matched by the Hungarian algorithm):\n\n")
    @printf(io, "- top-10 word overlap per topic: mean %.2f, min %.2f\n", mean(overlap), minimum(overlap))
    @printf(io, "- correlation of document-topic proportions θ per topic: median %.3f, min %.3f\n", median(θcor), minimum(θcor))
    @printf(io, "- effect of `rating` (Liberal vs Conservative) on topic prevalence, %d topics: correlation between implementations %.3f, mean |difference| %.4f (mean R standard error %.4f), same sign on %d/%d topics\n",
            K, cor(jeff, reff[:, 1]), mean(abs, jeff .- reff[:, 1]), mean(reff[:, 2]), count(sign.(jeff) .== sign.(reff[:, 1])), K)
end
print(read(joinpath(RESULTS, "stm_poliblog.md"), String))
