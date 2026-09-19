# Variational-Bayes LDA: the per-document kernels, threading, reproducibility, warm starts.

vb_allocated(f, args...) = (f(args...); @allocated f(args...))
vb_allocated_init(ws, doc, topics, α, init) =
    (TopicModeling.infer_doc!(ws, doc, topics, α; init); @allocated TopicModeling.infer_doc!(ws, doc, topics, α; init))
vb_allocated_bound(ws, doc, α) = (TopicModeling.doc_bound(ws, doc, α); @allocated TopicModeling.doc_bound(ws, doc, α))

@testset "LDA variational Bayes" begin
    corpus, truth, _ = simulate_lda(; D=300, K=5, V=120, doclen=60, alpha=0.3, rng=Xoshiro(1))
    K = 5
    α = fill(0.3, K)

    @testset "document kernels" begin
        topics = truth .+ 1e-3
        doc = corpus[1]
        ws = TopicModeling.DocVB(K)
        sstats = zeros(K, nterms(corpus))
        init = fill(0.3 + ntokens(doc) / K, K)
        cold = TopicModeling.infer_doc!(ws, doc, topics, α)                # also warms up
        γcold = copy(ws.gamma)
        @test sum(ws.gamma) ≈ sum(α) + ntokens(doc)
        TopicModeling.infer_doc!(ws, doc, topics, α; init)
        @test ws.gamma == γcold                                           # same start, same ascent
        warm = TopicModeling.infer_doc!(ws, doc, topics, α; init=view(reshape(γcold, K, 1), :, 1))
        @test warm <= 2 < cold                                            # restart at the fixed point
        @test sum(ws.gamma) ≈ sum(α) + ntokens(doc)
        @test ws.gamma ≈ γcold atol = 1e-2
        @test_throws DimensionMismatch TopicModeling.infer_doc!(ws, doc, topics, α; init=ones(K + 1))
        @test (@inferred TopicModeling.infer_doc!(ws, doc, topics, α)) isa Int
        @test (@inferred TopicModeling.infer_doc!(ws, doc, topics, α; init=γcold)) isa Int
        TopicModeling.accumulate_sstats!(sstats, ws, doc)
        TopicModeling.doc_bound(ws, doc, α)
        G = repeat(γcold, 1, 2)
        # Measured behind a function barrier: at testset scope Julia 1.10 charges the boxing of
        # the result and the keyword call to the expression, which says nothing about the kernel.
        @test vb_allocated(TopicModeling.infer_doc!, ws, doc, topics, α) == 0
        @test vb_allocated_init(ws, doc, topics, α, γcold) == 0
        @test vb_allocated_init(ws, doc, topics, α, view(G, :, 2)) == 0
        @test vb_allocated(TopicModeling.accumulate_sstats!, sstats, ws, doc) == 0
        @test vb_allocated_bound(ws, doc, α) == 0
        # every token's responsibilities sum to one, so the statistics sum to the document length
        fill!(sstats, 0.0); TopicModeling.accumulate_sstats!(sstats, ws, doc)
        @test sum(sstats) ≈ ntokens(doc)
    end

    @testset "topic expectations and bound" begin
        λ = rand(Xoshiro(2), K, 37) .+ 0.05
        E = TopicModeling.dirichlet_expectation!(similar(λ), λ)
        ref = TopicModeling.topic_bound(λ, E, 0.1)
        for T in (1, 3)
            cols = TopicModeling.chunks(37, T)
            E2 = similar(λ); X2 = similar(λ); rowsum = zeros(K); ψ = zeros(K)
            TopicModeling.topic_expectations!(E2, X2, λ, rowsum, ψ, cols)
            @test E2 == E && X2 == exp.(E)                                # bitwise, any chunking
            @test TopicModeling.topic_bound(λ, E2, 0.1, rowsum, cols) ≈ ref rtol = 1e-12
            ss = [rand(Xoshiro(t), K, 37) for t in 1:T]
            expected = foldl(+, ss)
            @test TopicModeling.merge_sstats!(ss, T, cols) == expected
            @test TopicModeling.update_lambda!(copy(λ), ss[1], 0.1, cols) == 0.1 .+ ss[1]
            @test TopicModeling.update_lambda!(copy(λ), ss[1], 0.1, cols, 0.25, 7.0) ==
                  @. (1 - 0.25) * λ + 0.25 * (0.1 + 7.0 * ss[1])
        end
        # the Dirichlet entropy/cross-entropy terms vanish when q(β) equals the prior
        η = 0.1; λ0 = fill(η, K, 37)
        @test TopicModeling.topic_bound(λ0, TopicModeling.dirichlet_expectation!(similar(λ0), λ0), η) ≈ 0 atol = 1e-9
    end

    @testset "batch" begin
        kw = (; method=:vb, alpha=0.3, eta=0.05, iters=40, tol=0.0)
        m1 = fit(LDA, corpus, K; kw..., nthreads=1, rng=Xoshiro(3))
        m3 = fit(LDA, corpus, K; kw..., nthreads=3, rng=Xoshiro(3))
        @test nondecreasing(m1.trace)
        @test m3.phi ≈ m1.phi atol = 1e-8
        @test m3.theta ≈ m1.theta atol = 1e-8
        @test m3.trace ≈ m1.trace rtol = 1e-9
        again = fit(LDA, corpus, K; kw..., nthreads=3, rng=Xoshiro(3))
        @test again.phi == m3.phi && again.theta == m3.theta && again.trace == m3.trace
        @test all(sum(m3.phi; dims=2) .≈ 1) && all(sum(m3.theta; dims=2) .≈ 1)

        for ws in (1, 10)                                                 # warm starts, from pass ws + 1
            w1 = fit(LDA, corpus, K; kw..., warm_start=ws, nthreads=1, rng=Xoshiro(3))
            w3 = fit(LDA, corpus, K; kw..., warm_start=ws, nthreads=3, rng=Xoshiro(3))
            @test nondecreasing(w1.trace) && nondecreasing(w3.trace)
            @test w1.trace[1:ws] == m1.trace[1:ws] && w1.trace != m1.trace
            @test w3.phi ≈ w1.phi atol = 1e-8
            @test w3.trace ≈ w1.trace rtol = 1e-9
            @test w3.theta == fit(LDA, corpus, K; kw..., warm_start=ws, nthreads=3, rng=Xoshiro(3)).theta
        end

        # A converged fit: φ, θ and the last ELBO belong to the same pass, and `transform`
        # infers with exp(E[log β]) as training did, so it reproduces θ.
        conv = fit(LDA, corpus, K; method=:vb, alpha=0.3, eta=0.05, iters=500, tol=1e-7, rng=Xoshiro(3))
        @test conv.iterations < 500
        @test conv.phi ≈ conv.lambda ./ sum(conv.lambda; dims=2)
        @test transform(conv, corpus) ≈ conv.theta atol = 1e-3
        @test maximum(abs, transform(conv, corpus) .- conv.theta) <
              maximum(abs, first(TopicModeling.infer_theta(corpus, conv.phi, conv.alpha)) .- conv.theta)
        opt = fit(LDA, corpus, K; kw..., alpha=0.1, optimize_alpha=true, rng=Xoshiro(3))
        @test nondecreasing(opt.trace) && 0.15 < mean(opt.alpha) < 0.6
    end

    @testset "online" begin
        kw = (; method=:vb, alpha=0.1, eta=0.05, iters=6, batchsize=50, optimize_alpha=true)
        o1 = fit(LDA, corpus, K; kw..., nthreads=1, rng=Xoshiro(4))
        o3 = fit(LDA, corpus, K; kw..., nthreads=3, rng=Xoshiro(4))
        @test o3.phi ≈ o1.phi atol = 1e-8
        @test o3.trace ≈ o1.trace rtol = 1e-9
        @test o3.phi == fit(LDA, corpus, K; kw..., nthreads=3, rng=Xoshiro(4)).phi
        @test all(isfinite, o1.trace) && o1.trace[end] > o1.trace[1]
        # damped Newton steps (ρ_t < 0.1 here): α moves smoothly and stays positive
        @test all(>(0), o1.alpha) && maximum(o1.alpha) / minimum(o1.alpha) < 10
    end

    @testset "arguments" begin
        small = corpus[1:40]
        @test_throws ArgumentError fit(LDA, small, 3; method=:vb, iters=0)
        @test_throws ArgumentError fit(LDA, small, 3; method=:vb, alpha=0.0)
        @test_throws ArgumentError fit(LDA, small, 3; method=:vb, alpha=[0.1, -0.1, 0.1])
        @test_throws ArgumentError fit(LDA, small, 3; method=:vb, eta=0.0)
        @test_throws ArgumentError fit(LDA, small, 3; method=:vb, warm_start=-1)
        one = fit(LDA, small, 3; method=:vb, iters=1, rng=Xoshiro(1))
        @test all(isfinite, one.theta) && all(sum(one.theta; dims=2) .≈ 1)
        seeded = fit(LDA, corpus, K; method=:vb, init=truth, iters=3, alpha=0.3, rng=Xoshiro(1))
        @test mean(match_topics(truth, seeded.phi)[2]) < 0.1
        g = fit(LDA, small, 3; iters=5, rng=Xoshiro(1))                   # Gibbs models transform with phi
        @test transform(g, small) == first(TopicModeling.infer_theta(small, g.phi, g.alpha))
    end
end
