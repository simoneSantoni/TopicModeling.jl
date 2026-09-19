# LDA on the AP corpus (2021 train / 225 test documents, K = 50): Julia vs C++/Python/R.
# Reference runs must exist in results/ (see run_all.sh). Every topic matrix — ours and
# theirs — goes through the same `score` function.
include(joinpath(@__DIR__, "common.jl"))
train = load("ap.train"); test = load("ap.test"); K = 50
rows = Tuple[]
fit(LDA, train, K; iters=3, rng=Xoshiro(1)); fit(LDA, train, K; method=:vb, iters=2, rng=Xoshiro(1))   # compile
for T in (1, 8)
    runs = [fit(LDA, train, K; iters=1000, alpha=0.1, eta=0.01, nthreads=T, rng=Xoshiro(r)) for r in 1:3]
    m = runs[sortperm([r.elapsed for r in runs])[2]]                  # median of three, as for tomotopy
    println("julia gibbs T=$T times: ", round.([r.elapsed for r in runs]; digits=2))
    write_phi(joinpath(RESULTS, "lda_ap_julia_gibbs_t$T"), m.phi; impl="TopicModeling.jl Gibbs", seconds=m.elapsed, alpha=m.alpha)
    push!(rows, ("**TopicModeling.jl** Gibbs", "Julia", "collapsed Gibbs, 1000 sweeps", T, m.elapsed, score(m.phi, m.alpha, train, test)))
end
for T in (1, 8)
    m = fit(LDA, train, K; method=:vb, iters=100, tol=0.0, alpha=0.1, eta=0.01, nthreads=T, rng=Xoshiro(1))
    push!(rows, ("**TopicModeling.jl** VB", "Julia", "batch VB, 100 passes", T, m.elapsed, score(m.phi, m.alpha, train, test)))
end
refs = [("lda_ap_tomotopy_t1", "tomotopy", "C++", "collapsed Gibbs, 1000 sweeps", 1), ("lda_ap_tomotopy_t8", "tomotopy", "C++", "collapsed Gibbs, 1000 sweeps", 8),
        ("lda_ap_Rlda", "lda", "R/C", "collapsed Gibbs, 1000 sweeps", 1), ("lda_ap_topicmodels_gibbs", "topicmodels", "R/C++", "collapsed Gibbs, 1000 sweeps", 1),
        ("lda_ap_sklearn", "scikit-learn", "Python/Cython", "batch VB, 100 passes", 1), ("lda_ap_gensim", "gensim", "Python/NumPy", "online VB, 100 passes", 1),
        ("lda_ap_topicmodels_vem", "topicmodels", "R/C (lda-c)", "VEM, α estimated", 1)]
for (file, name, lang, algo, T) in refs
    isfile(joinpath(RESULTS, file * ".json")) || continue
    phi, meta = read_phi(joinpath(RESULTS, file))
    push!(rows, (name, lang, algo, T, metafield(meta, "seconds"), score(phi, metaalpha(meta, K, 0.1), train, test)))
end
open(joinpath(RESULTS, "lda_ap.md"), "w") do io
    println(io, "| implementation | language | algorithm | threads | time (s) | held-out perplexity | NPMI | diversity |\n|---|---|---|---|---|---|---|---|")
    for (name, lang, algo, T, secs, s) in rows
        @printf(io, "| %s | %s | %s | %d | %.1f | %.0f ± %.0f | %.3f | %.2f |\n", name, lang, algo, T, secs, s.perplexity, s.perplexity_sd, s.npmi, s.diversity)
    end
end
print(read(joinpath(RESULTS, "lda_ap.md"), String))
