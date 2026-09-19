# CTM on AP (K = 20): TopicModeling.jl vs R topicmodels (Blei & Lafferty's ctm-c) vs tomotopy,
# plus the literature check of Blei & Lafferty (2007, §4.2): with few observed words the CTM
# predicts the rest of a document better than LDA.
include(joinpath(@__DIR__, "common.jl"))
train = load("ap.train"); test = load("ap.test"); K = 20
keep = findall(d -> ntokens(d) >= 2, train.docs)
fit(CTM, train[1:200], 5; iters=2)                                                 # compile
rows = Tuple[]
ctm = nothing
for T in (1, 8)
    m = fit(CTM, train, K; nthreads=T); T == 1 && (global ctm = m)
    push!(rows, ("**TopicModeling.jl**", "Julia", "variational EM (Blei-Lafferty family)", T, m.elapsed, "$(m.iterations) EM its", score(m.phi, 0.1, train, test)))
end
for (file, name, lang, algo, T) in (("ctm_ap_topicmodels", "topicmodels (ctm-c)", "R/C", "variational EM (Blei-Lafferty)", 1),
                                    ("ctm_ap_tomotopy_t1", "tomotopy", "C++", "Gibbs, 1000 sweeps", 1), ("ctm_ap_tomotopy_t8", "tomotopy", "C++", "Gibbs, 1000 sweeps", 8))
    isfile(joinpath(RESULTS, file * ".json")) || continue
    phi, meta = read_phi(joinpath(RESULTS, file)); its = metafield(meta, "iters")
    push!(rows, (name, lang, algo, T, metafield(meta, "seconds"), "$(Int(its)) its", score(phi, 0.1, train, test)))
end
open(joinpath(RESULTS, "ctm_ap.md"), "w") do io
    println(io, "| implementation | language | algorithm | threads | time (s) | iterations | held-out perplexity | NPMI |\n|---|---|---|---|---|---|---|---|")
    for (name, lang, algo, T, secs, its, s) in rows
        @printf(io, "| %s | %s | %s | %d | %.1f | %s | %.0f ± %.0f | %.3f |\n", name, lang, algo, T, secs, its, s.perplexity, s.perplexity_sd, s.npmi)
    end
    # Literature check. Both models variational, both with a learnt prior, each folding in with its own prior.
    println(io, "\nCTM vs LDA, held-out perplexity of the unobserved part of each test document (own prior, mean of 5 splits):\n")
    println(io, "| K | share of words observed | LDA (VB, α learnt) | CTM | CTM − LDA |\n|---|---|---|---|---|")
    for k in (20, 40)
        c = k == K ? ctm : fit(CTM, train, k)
        l = fit(LDA, train, k; method=:vb, optimize_alpha=true, rng=Xoshiro(1))
        for frac in (0.1, 0.25, 0.5, 0.75)
            pl = mean(first(heldout_perplexity(l, test; frac, rng=Xoshiro(s))) for s in 1:5)
            pc = mean(first(heldout_perplexity(c, test; frac, rng=Xoshiro(s))) for s in 1:5)
            @printf(io, "| %d | %d%% | %.0f | %.0f | %+.0f |\n", k, round(Int, 100frac), pl, pc, pc - pl)
        end
    end
end
print(read(joinpath(RESULTS, "ctm_ap.md"), String))
