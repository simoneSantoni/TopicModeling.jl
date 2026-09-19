using TopicModeling
using TopicModeling: hungarian, LBFGS, minimize!, DTMTopicObjective, dtm_prior, CTMDocObjective,
                     LogisticNormalDoc, load_doc!, hessian!, randdirichlet, logsumexp
using LinearAlgebra, Random, Statistics, Test

# Central finite differences of an `fg!(g, x)`-style objective.
function fdgrad(f, x; h=1e-6)
    scratch = similar(x)
    return [(e = zeros(length(x)); e[i] = h; (f(scratch, x .+ e) - f(scratch, x .- e)) / 2h) for i in eachindex(x)]
end
nondecreasing(trace; rtol=1e-7) = all(diff(trace) .> -rtol .* abs.(trace[1:end-1]))

@testset "TopicModeling" begin

@testset "utilities" begin
    @test logsumexp([1000.0, 1000.0]) ≈ 1000 + log(2)
    C = rand(Xoshiro(1), 5, 5)
    a = hungarian(C)
    @test sort(a) == 1:5
    brute = minimum(sum(C[i, p[i]] for i in 1:5) for p in Iterators.product(ntuple(_ -> 1:5, 5)...) if allunique(p))
    @test sum(C[i, a[i]] for i in 1:5) ≈ brute
    θ = randdirichlet(Xoshiro(2), fill(0.3, 8))
    @test sum(θ) ≈ 1 && all(>=(0), θ)
    # L-BFGS on the Rosenbrock function
    rosen(g, x) = (g[1] = -2(1 - x[1]) - 400x[1] * (x[2] - x[1]^2); g[2] = 200(x[2] - x[1]^2); (1 - x[1])^2 + 100(x[2] - x[1]^2)^2)
    x = [-1.2, 1.0]
    f, _, ok = minimize!(rosen, x, LBFGS(2); maxiter=500, gtol=1e-8)
    @test ok
    @test x ≈ [1.0, 1.0] atol = 1e-5
end

@testset "corpus" begin
    c = Corpus(["the cat sat on the mat", "the dog sat", "cats and dogs"]; stopwords=["the", "on", "and"])
    @test ndocs(c) == 3 && "cat" in c.vocab && !("the" in c.vocab)
    @test ntokens(c) == 3 + 2 + 2
    @test sum(dtm(c)) == ntokens(c)
    @test ndocs(Corpus(dtm(c), c.vocab)) == 3 && ntokens(Corpus(dtm(c), c.vocab)) == ntokens(c)
    mktempdir() do dir
        write_ldac(joinpath(dir, "c.ldac"), c; vocab=joinpath(dir, "c.vocab"))
        c2 = read_ldac(joinpath(dir, "c.ldac"); vocab=joinpath(dir, "c.vocab"))
        @test c2.vocab == c.vocab && all(c2[i].terms == c[i].terms && c2[i].counts == c[i].counts for i in 1:3)
    end
    big, _, _ = simulate_lda(; D=50, K=3, V=40, rng=Xoshiro(1))
    obs, held = split_documents(big; rng=Xoshiro(1))
    @test all(ntokens(obs[i]) + ntokens(held[i]) == ntokens(big[i]) for i in 1:50)
    pruned, kept = prune(big; min_df=5)
    @test all(>=(5), docfreq(pruned)) && pruned.vocab == big.vocab[kept]
end

# Griffiths & Steyvers (2004): ten "bar" topics on a 5×5 grid must be recovered.
@testset "LDA recovers the Griffiths-Steyvers bars" begin
    bars = bars_topics(5)
    corpus, _, theta = simulate_lda(; D=600, topics=bars, doclen=100, alpha=1.0, rng=Xoshiro(1))
    g = fit(LDA, corpus, 10; method=:gibbs, iters=300, alpha=1.0, eta=0.1, optimize_alpha=false, rng=Xoshiro(2))
    perm, dist = match_topics(bars, g.phi)
    @test maximum(dist) < 0.15
    @test mean(abs, g.theta[:, perm] .- theta) < 0.06
    @test all(sum(g.phi; dims=2) .≈ 1) && all(sum(g.theta; dims=2) .≈ 1)
    # Variational Bayes has merged-topic local optima (as do sklearn and gensim: see
    # benchmarks/RESULTS.md). What must hold: the ELBO never decreases, and it ranks the
    # optima correctly, so the best of a few restarts recovers the bars.
    vs = [fit(LDA, corpus, 10; method=:vb, alpha=1.0, eta=0.1, doc_tol=1e-5, tol=1e-8, iters=400, rng=Xoshiro(s)) for s in 1:4]
    @test all(v -> nondecreasing(v.trace), vs)
    best = vs[argmax([v.trace[end] for v in vs])]
    @test maximum(match_topics(bars, best.phi)[2]) < 0.15

    # The threaded sampler (document × word block partitioning) is as good as the serial one. Single
    # chains can sit in a merged-topic mode, so compare the best-likelihood chain of each.
    big, truth, _ = simulate_lda(; D=1500, K=8, V=300, doclen=80, rng=Xoshiro(3))
    for T in (1, 4)
        chains = [fit(LDA, big, 8; iters=200, nthreads=T, rng=Xoshiro(s)) for s in 1:4]
        best = chains[argmax([c.trace[end] for c in chains])]
        @test mean(match_topics(truth, best.phi)[2]) < 0.06
        @test sum(length, best.z) == ntokens(big)
    end
    s1 = fit(LDA, big, 8; iters=20, rng=Xoshiro(4))
    @test size(transform(s1, big[1:10])) == (10, 8)
    # Online VB is meant for streams and is weak on 1500 documents (so is sklearn's); it
    # must still beat an uninformed model by a wide margin.
    online = fit(LDA, big, 8; method=:vb, batchsize=256, iters=15, rng=Xoshiro(5))
    @test all(isfinite, online.trace) && all(sum(online.phi; dims=2) .≈ 1)
    @test mean(match_topics(truth, online.phi)[2]) < 0.6 * mean(match_topics(truth, fill(1 / 300, 8, 300))[2])
end

@testset "evaluation" begin
    corpus, phi, _ = simulate_lda(; D=400, K=5, V=200, doclen=80, rng=Xoshiro(1))
    train, test, _, _ = train_test_split(corpus; test=0.2, rng=Xoshiro(2))
    shuffled = phi[:, randperm(Xoshiro(3), 200)]
    @test first(heldout_perplexity(phi, 0.1, test; rng=Xoshiro(1))) < first(heldout_perplexity(shuffled, 0.1, test; rng=Xoshiro(1)))
    @test mean(coherence(phi, train)) > mean(coherence(shuffled, train))
    @test all(-1 .<= coherence(phi, train) .<= 1)
    @test 0 < topic_diversity(phi) <= 1
    @test match_topics(phi, phi[[3, 1, 5, 2, 4], :])[1] == [2, 4, 1, 5, 3]
end

@testset "STM" begin
    rng = Xoshiro(7)
    D, K, V = 1200, 5, 400
    x = Float64.(rand(rng, D) .< 0.5)
    Γ = zeros(2, K - 1); Γ[2, 1] = 1.5; Γ[2, 2] = -1.0
    corpus, phi, theta = simulate_logistic_normal(; D, K, V, mu=hcat(ones(D), x) * Γ, Sigma=0.5 * Matrix(I, K - 1, K - 1), doclen=80:140, rng)
    # gradient and Hessian of the collapsed document objective
    o = LogisticNormalDoc(K); o.siginv = Matrix(2.0I, K - 1, K - 1); o.mu .= 0.1
    load_doc!(o, corpus[1], phi .+ 1e-4)
    x0 = randn(rng, K - 1); g = zeros(K - 1); o(g, x0)
    @test g ≈ fdgrad(o, x0) atol = 1e-4
    H = zeros(K - 1, K - 1); hessian!(H, o, x0, zeros(K, length(corpus[1].terms)))
    Hfd = reduce(hcat, [(e = zeros(K - 1); e[j] = 1e-5; gp = zeros(K - 1); gm = zeros(K - 1); o(gp, x0 .+ e); o(gm, x0 .- e); (gp .- gm) ./ 2e-5) for j in 1:K-1])
    @test H ≈ Hfd atol = 1e-3

    m = fit(STM, corpus, K; prevalence=reshape(x, :, 1), keep_nu=true, tol=1e-6, rng=Xoshiro(1))
    @test m.converged && nondecreasing(m.trace; rtol=1e-6)
    perm, dist = match_topics(phi, m.phi)
    lda = fit(LDA, corpus, K; iters=300, rng=Xoshiro(1))
    @test mean(dist) < 0.15
    @test mean(dist) < mean(match_topics(phi, lda.phi)[2])     # the right model beats LDA on logistic-normal data
    @test mean(abs, m.theta[:, perm] .- theta) < 0.05
    # spectral initialisation: one anchor per true topic
    _, anchors = spectral_init(corpus, K)
    @test sort([argmax(phi[:, a]) for a in anchors]) == 1:K
    coef, se = estimate_effect(m; nsims=10, rng=Xoshiro(2))
    truth = hcat(ones(D), x) \ theta                  # effect of x on the true proportions
    @test coef[2, perm] ≈ truth[2, :] atol = 0.04
    @test coef[2, perm[1]] > 0.15 && coef[2, perm[2]] < -0.05
    @test size(transform(m, corpus[1:5]; prevalence=reshape(x[1:5], :, 1))) == (5, K)
end

@testset "DTM" begin
    W, _, _ = dtm_prior(4, 0.01, 0.5, 5.0)
    n = 10 .* rand(Xoshiro(1), 7, 4)
    obj = DTMTopicObjective(n, vec(sum(n; dims=1)), W)
    x0 = randn(Xoshiro(2), 28); g = zeros(28); obj(g, x0)
    @test g ≈ fdgrad(obj, x0) atol = 1e-4

    corpus, times, phi, _ = simulate_dtm(; T=8, docs_per_slice=120, K=4, V=250, chain_variance=0.03, doclen=100, rng=Xoshiro(2))
    m = fit(DTM, corpus, times, 4; chain_variance=0.03, alpha=0.1, rng=Xoshiro(3))
    @test nondecreasing(m.trace)
    @test length(m.phi) == 8 && all(all(sum(φ; dims=2) .≈ 1) for φ in m.phi)
    avg = sum(phi) ./ 8
    perm, _ = match_topics(avg, topicword(m))
    static = fit(LDA, corpus, 4; iters=300, rng=Xoshiro(3))
    sperm, _ = match_topics(avg, static.phi)
    tv(a, b) = mean(0.5 .* sum(abs, a .- b; dims=2))
    dyn_err = mean(tv(phi[t], m.phi[t][perm, :]) for t in 1:8)
    static_err = mean(tv(phi[t], static.phi[sperm, :]) for t in 1:8)
    @test dyn_err < 0.12
    @test dyn_err < static_err                          # tracking drift must beat ignoring it
    @test length(topwords(m, 1; n=5)) == 4
    θ = transform(m, corpus[1:20], times[1:20])
    @test size(θ) == (20, 4) && all(sum(θ; dims=2) .≈ 1)
    @test_throws ArgumentError transform(m, corpus[1:2], [99, 99])
end

@testset "CTM" begin
    rng = Xoshiro(11)
    K, V, D = 4, 300, 1500
    C = Matrix(1.0I, K - 1, K - 1); C[1, 2] = C[2, 1] = 0.8
    corpus, phi, theta = simulate_logistic_normal(; D, K, V, Sigma=1.5 .* C, doclen=100:160, rng)
    o = LogisticNormalDoc(K); A = randn(rng, K - 1, K - 1); o.siginv = inv(A * A' + I); o.mu .= 0.2
    load_doc!(o, corpus[1], phi .+ 1e-4)
    obj = CTMDocObjective(o, diag(o.siginv)); x0 = randn(rng, 2(K - 1)); g = zeros(2(K - 1)); obj(g, x0)
    @test g ≈ fdgrad(obj, x0) atol = 1e-4

    m = fit(CTM, corpus, K; rng=Xoshiro(1))
    @test m.converged && nondecreasing(m.trace)
    perm, dist = match_topics(phi, m.phi)
    @test maximum(dist) < 0.15
    ĉ = cor(m.theta[:, perm]); c0 = cor(theta)
    @test ĉ[1, 2] ≈ c0[1, 2] atol = 0.1                 # the planted correlation is recovered
    @test isposdef(Symmetric(m.Sigma))
    @test size(transform(m, corpus[1:7])) == (7, K)
    @test first(heldout_perplexity(m, corpus[1:100]; rng=Xoshiro(1))) < V
end

include("vb_tests.jl")
include("gibbs_tests.jl")

include("logisticnormal_tests.jl")
include("dtm_tests.jl")
include("robustness_tests.jl")
end
