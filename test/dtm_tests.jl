# Dynamic topic model: kernel, invariance to the number of threads, held-out evaluation and
# argument validation. Runs on its own or from runtests.jl.
using TopicModeling
using TopicModeling: DTMTopicObjective, dtm_prior, _foreach_pooled
using Random, Statistics, Test

nondecreasing_trace(trace; rtol=1e-7) = all(diff(trace) .> -rtol .* abs.(trace[1:end-1]))

@testset "DTM details" begin
    # --- M-step objective -------------------------------------------------------------------
    V, T = 9, 5
    W, Sdiag, _ = dtm_prior(T, 0.01, 0.5, 5.0)
    @test length(W) == T && all(>(0), W) && all(>(0), Sdiag)
    n = 10 .* rand(Xoshiro(1), V, T); n[3, :] .= 0            # a term that never occurs
    obj = DTMTopicObjective(n, vec(sum(n; dims=1)), W)
    x0 = 3 .* randn(Xoshiro(2), V * T); g = zeros(V * T)
    f0 = obj(g, x0)
    fd = map(eachindex(x0)) do i
        e = zeros(V * T); e[i] = 1e-6
        (obj(similar(g), x0 .+ e) - obj(similar(g), x0 .- e)) / 2e-6
    end
    @test g ≈ fd atol = 1e-4
    # the value is that of the textbook formula, also far from the origin (no overflow)
    direct(x) = (m = reshape(x, V, T); sum(sum(n[:, t]) * log(sum(exp, m[:, t])) - sum(n[:, t] .* m[:, t]) +
                 0.5 * W[t] * sum(abs2, t == 1 ? m[:, 1] : m[:, t] .- m[:, t - 1]) for t in 1:T))
    @test f0 ≈ direct(x0)
    @test isfinite(obj(g, x0 .+ 800)) && all(isfinite, g)
    allocated(o, g, x) = @allocated o(g, x)
    allocated(obj, g, x0)
    @test allocated(obj, g, x0) == 0

    # --- task pool --------------------------------------------------------------------------
    for nworkers in (1, 3, 50)
        hits = zeros(Int, 20); owner = zeros(Int, 20)
        _foreach_pooled(20:-1:1, nworkers) do i, p
            hits[i] += 1; owner[i] = p
        end
        @test all(==(1), hits) && all(p -> 1 <= p <= min(nworkers, 20), owner)
    end
    @test_throws CompositeException _foreach_pooled((i, p) -> error("boom"), 1:4, 2)

    # --- fit --------------------------------------------------------------------------------
    corpus, times, _, _ = simulate_dtm(; T=5, docs_per_slice=40, K=3, V=80, chain_variance=0.02, doclen=40, rng=Xoshiro(4))
    train = corpus[1:2:end]; ttrain = times[1:2:end]
    test = corpus[2:2:end]; ttest = times[2:2:end]
    # Given the initial topics the fit is exactly the same for any number of threads: E-step
    # blocks are merged in a fixed order and an M-step does not depend on the L-BFGS workspace
    # it gets. (The default `init=:lda` is a threaded Gibbs run, a different chain for each
    # `nthreads`, hence the matrix `init`.)
    init = fit(LDA, train, 3; iters=30, alpha=0.1, optimize_alpha=false, nthreads=1, rng=Xoshiro(5)).phi
    m1 = fit(DTM, train, ttrain, 3; alpha=0.1, chain_variance=0.02, iters=6, tol=0.0, init, nthreads=1)
    m3 = fit(DTM, train, ttrain, 3; alpha=0.1, chain_variance=0.02, iters=6, tol=0.0, init, nthreads=3)
    @test m1.trace == m3.trace
    @test m1.phi == m3.phi && m1.theta == m3.theta
    @test length(m1.trace) == 6 && nondecreasing_trace(m1.trace)
    @test all(sum(m1.theta; dims=2) .≈ 1) && all(all(sum(φ; dims=2) .≈ 1) for φ in m1.phi)
    @test m1 isa DTM{Int} && m1.periods == 1:5 && m1.times == ttrain
    @test startswith(sprint(show, m1), "DTM(K=3, V=80, D=100)")
    d1 = fit(DTM, train, ttrain, 3; iters=2, nthreads=2, rng=Xoshiro(6))
    d2 = fit(DTM, train, ttrain, 3; iters=2, nthreads=2, rng=Xoshiro(6))
    @test d1.phi == d2.phi && nondecreasing_trace(d1.trace)

    # Slices are equally spaced whatever their labels; labels only need to be sortable.
    a = fit(DTM, train, ttrain, 3; init, iters=2, tol=0.0)
    labels = ["1999", "2001", "2010", "2011", "2030"]
    b = fit(DTM, train, labels[ttrain], 3; init, iters=2, tol=0.0)
    @test b isa DTM{String} && b.periods == labels
    @test a.phi == b.phi && a.trace == b.trace
    @test transform(b, test[1:4], labels[ttest[1:4]]) == transform(a, test[1:4], ttest[1:4])
    @test fit(DTM, train, Any[t for t in ttrain], 3; init, iters=1) isa DTM{Int}

    # --- held-out evaluation ------------------------------------------------------------------
    ppl, ll = heldout_perplexity(m1, test, ttest; rng=Xoshiro(1), nthreads=1)
    @test isfinite(ppl) && 1 < ppl < 80 && ll ≈ -log(ppl)
    @test heldout_perplexity(m1, test, ttest; rng=Xoshiro(1), nthreads=3) == (ppl, ll)
    @test_throws ArgumentError heldout_perplexity(m1, test)
    @test_throws ArgumentError heldout_perplexity(m1, test, fill(99, ndocs(test)))
    @test_throws DimensionMismatch heldout_perplexity(m1, test, ttest[1:3])
    other, _, _ = simulate_lda(; D=5, K=2, V=30, rng=Xoshiro(1))
    @test_throws DimensionMismatch heldout_perplexity(m1, other, ones(Int, 5))
    @test_throws DimensionMismatch transform(m1, other, ones(Int, 5))

    # --- argument validation ------------------------------------------------------------------
    @test_throws ArgumentError fit(DTM, train, ttrain, 1; init=fill(1 / 80, 1, 80))
    @test_throws ArgumentError fit(DTM, train, ttrain, 1)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, chain_variance=0.0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, chain_variance=-1.0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, obs_variance=0.0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, init_variance=0.0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, alpha=0.0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, alpha=[0.1, -0.1, 0.1])
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init, nthreads=0)
    @test_throws ArgumentError fit(DTM, train, ttrain, 3; init=:nonsense)
    @test_throws DimensionMismatch fit(DTM, train, ttrain, 3; init=fill(1 / 80, 2, 80))
    @test_throws DimensionMismatch fit(DTM, train, ttrain[1:10], 3; init)
end
