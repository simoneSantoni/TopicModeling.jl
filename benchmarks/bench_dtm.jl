# DTM. (1) Synthetic corpus with known drifting topics: recovery error of TopicModeling.jl, gensim's
# LdaSeqModel (port of Blei & Lafferty's code) and tomotopy's DTModel. (2) poliblog5k sliced by month.
# Usage: julia bench_dtm.jl prepare   -> writes the corpora for the reference runs
#        julia bench_dtm.jl           -> fits, scores, writes results/dtm.md
include(joinpath(@__DIR__, "common.jl"))
using DelimitedFiles
const CV = 0.01
function synthetic()
    simulate_dtm(; T=8, docs_per_slice=150, K=5, V=300, chain_variance=CV, doclen=100, alpha=0.1, rng=Xoshiro(2026))
end
function poliblog()
    c = load("poliblog5k"); meta = readdlm(joinpath(DATA, "poliblog5k.meta.csv"), ','; skipstart=1)
    month = clamp.(ceil.(Int, Float64.(meta[:, 2]) ./ 30.5), 1, 12)
    tr = parse.(Int, readlines(joinpath(DATA, "poliblog5k.train.idx"))); te = parse.(Int, readlines(joinpath(DATA, "poliblog5k.test.idx")))
    otr = tr[sortperm(month[tr])]; ote = te[sortperm(month[te])]
    return c[otr], month[otr], c[ote], month[ote]
end
function read_phi_t(prefix)
    meta = read(prefix * ".json", String); T = Int(metafield(meta, "T")); K = Int(metafield(meta, "K")); V = Int(metafield(meta, "V"))
    raw = Vector{Float64}(undef, T * K * V); read!(prefix * ".phi.bin", raw)
    A = reshape(raw, V, K, T)                                  # row-major T×K×V on disk
    return [permutedims(A[:, :, t]) for t in 1:T], meta
end
if get(ARGS, 1, "") == "prepare"
    corpus, times, _, _ = synthetic()
    write_ldac(joinpath(DATA, "dtm_synth.ldac"), corpus; vocab=joinpath(DATA, "dtm_synth.vocab")); write(joinpath(DATA, "dtm_synth.times"), join(times, '\n'))
    ptrain, ptimes, _, _ = poliblog()
    write_ldac(joinpath(DATA, "dtm_poliblog.train.ldac"), ptrain); write(joinpath(DATA, "dtm_poliblog.train.times"), join(ptimes, '\n'))
    println("prepared: ", corpus, " and ", ptrain); exit()
end
tvslices(truth, est, perm) = [mean(0.5 .* sum(abs, truth[t] .- est[t][perm, :]; dims=2)) for t in eachindex(truth)]
open(joinpath(RESULTS, "dtm.md"), "w") do io
    corpus, times, phi, _ = synthetic(); T = length(phi); avg = sum(phi) ./ T
    fit(DTM, corpus[1:200], times[1:200], 3; iters=2)                              # compile
    println(io, "Synthetic DTM (8 slices × 150 documents, K = 5, V = 300, chain variance 0.01). Error = total-variation distance between the estimated and the true topic at each slice, averaged over topics and slices; drift = the same between the first and the last slice of the true topics.\n")
    @printf(io, "True drift TV(φ₁, φ_T) = %.3f.\n\n", mean(0.5 .* sum(abs, phi[1] .- phi[end]; dims=2)))
    println(io, "| implementation | language | algorithm | threads | time (s) | mean TV to true φ_t | worst slice |\n|---|---|---|---|---|---|---|")
    for Th in (1, 8)
        m = fit(DTM, corpus, times, 5; chain_variance=CV, alpha=0.1, nthreads=Th, rng=Xoshiro(1))
        e = tvslices(phi, m.phi, match_topics(avg, topicword(m))[1])
        @printf(io, "| **TopicModeling.jl** | Julia | variational Kalman (concave reparametrisation), %d EM its | %d | %.1f | %.3f | %.3f |\n", m.iterations, Th, m.elapsed, mean(e), maximum(e))
    end
    for (file, name, lang, algo, Th) in (("dtm_synth_gensim", "gensim LdaSeqModel", "Python/NumPy", "variational Kalman (Blei-Lafferty port)", 1),
                                         ("dtm_synth_tomotopy_t1", "tomotopy DTModel", "C++", "SGLD/Gibbs, 1000 its", 1), ("dtm_synth_tomotopy_t8", "tomotopy DTModel", "C++", "SGLD/Gibbs, 1000 its", 8))
        isfile(joinpath(RESULTS, file * ".json")) || continue
        est, meta = read_phi_t(joinpath(RESULTS, file)); e = tvslices(phi, est, match_topics(avg, sum(est) ./ T)[1])
        @printf(io, "| %s | %s | %s | %d | %.1f | %.3f | %.3f |\n", name, lang, algo, Th, metafield(meta, "seconds"), mean(e), maximum(e))
    end
    l = fit(LDA, corpus, 5; iters=500, rng=Xoshiro(1)); e = tvslices(phi, fill(l.phi, T), match_topics(avg, l.phi)[1])
    @printf(io, "| static LDA baseline | Julia | collapsed Gibbs | 1 | %.1f | %.3f | %.3f |\n", l.elapsed, mean(e), maximum(e))

    ptrain, ptimes, ptest, ptest_times = poliblog(); K = 10
    println(io, "\npoliblog5k by month (12 slices, 4500 train / 500 test posts, K = 10). Held-out perplexity: each test post is scored with the topics of its own month, by the evaluator common to all implementations.\n")
    println(io, "| implementation | threads | time (s) | held-out perplexity | NPMI |\n|---|---|---|---|---|")
    function score_t(phis)
        s = [score(phis[t], 0.1, ptrain, ptest[findall(==(t), ptest_times)]; seeds=1:3) for t in 1:12 if count(==(t), ptest_times) > 0]
        w = [count(==(t), ptest_times) for t in 1:12 if count(==(t), ptest_times) > 0]
        return exp(sum(w .* log.(getfield.(s, :perplexity))) / sum(w)), mean(getfield.(s, :npmi))
    end
    for Th in (1, 8)
        m = fit(DTM, ptrain, ptimes, K; alpha=0.1, nthreads=Th, rng=Xoshiro(1)); pp, npmi = score_t(m.phi)
        @printf(io, "| **TopicModeling.jl** DTM (%d EM its) | %d | %.1f | %.0f | %.3f |\n", m.iterations, Th, m.elapsed, pp, npmi)
    end
    for (file, Th) in (("dtm_poliblog_tomotopy_t1", 1), ("dtm_poliblog_tomotopy_t8", 8))
        isfile(joinpath(RESULTS, file * ".json")) || continue
        est, meta = read_phi_t(joinpath(RESULTS, file)); pp, npmi = score_t(est)
        @printf(io, "| tomotopy DTModel (1000 its) | %d | %.1f | %.0f | %.3f |\n", Th, metafield(meta, "seconds"), pp, npmi)
    end
    l = fit(LDA, ptrain, K; iters=1000, rng=Xoshiro(1)); pp, npmi = score_t(fill(l.phi, 12))
    @printf(io, "| static LDA baseline (Julia Gibbs) | 8 | %.1f | %.0f | %.3f |\n", l.elapsed, pp, npmi)
end
print(read(joinpath(RESULTS, "dtm.md"), String))
