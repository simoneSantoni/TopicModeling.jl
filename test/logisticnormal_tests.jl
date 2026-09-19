# Numerical kernels behind the logistic-normal models (STM, CTM): L-BFGS memory, the
# allocation-free Cholesky/inverse of the Laplace step, the sparse co-occurrence Gram matrix,
# the EM stopping rule, reproducibility and argument validation.

# Ill-conditioned and non-convex (concave where |x_i| > π/2): from `WAVY_X0` some steps have
# sᵀy ≤ 0, so L-BFGS must reject pairs, also when its memory is already full.
function wavy(g, x)
    n = length(x); f = 0.0
    for i in 1:n
        w = 10.0^(2 - i); f += w * (1 - cos(x[i])); g[i] = w * sin(x[i])
    end
    for i in 1:(n - 1)
        d = x[i] - x[i + 1]; f += 0.025 * d^2; g[i] += 0.05 * d; g[i + 1] -= 0.05 * d
    end
    return f
end
const WAVY_X0 = [2.13, 2.6, 1.06, 0.02, -0.54, -1.12]
function cholinv_allocated(H, A, dsave, W)
    copyto!(H, A)
    return @allocated TopicModeling._cholesky_inverse!(TopicModeling._safe_cholesky!(H, dsave), W)
end
lbfgs_allocated(ws, x) = (copyto!(x, WAVY_X0); @allocated TopicModeling.minimize!(wavy, x, ws; maxiter=500, gtol=1e-8))

@testset "logistic-normal kernels" begin

@testset "L-BFGS rejects pairs without touching the memory" begin
    # Invariant of the memory, whenever the optimiser is stopped: rho_c · s_cᵀy_c = 1 and
    # s_cᵀy_c > 0 for every live column. A rejected pair written over the oldest live column
    # (the bug fixed here) breaks it for maxiter = 4:7 with m = 2.
    for m in (1, 2, 3), maxiter in 2:15
        ws = TopicModeling.LBFGS(6; m)
        TopicModeling.minimize!(wavy, copy(WAVY_X0), ws; maxiter, gtol=1e-8)
        for c in 1:m
            ws.rho[c] == 0 && continue
            sy = dot(ws.S[:, c], ws.Y[:, c])
            @test sy > 0 && ws.rho[c] * sy ≈ 1
        end
    end
    # the very first step from WAVY_X0 is one of the rejected ones
    g0 = zeros(6); g1 = zeros(6); wavy(g0, WAVY_X0)
    x1 = WAVY_X0 .- min(1, 1 / norm(g0)) .* g0; wavy(g1, x1)
    @test dot(x1 .- WAVY_X0, g1 .- g0) < 0
    for m in (2, 10)
        x = copy(WAVY_X0)
        f, _, ok = TopicModeling.minimize!(wavy, x, TopicModeling.LBFGS(6; m); maxiter=500, gtol=1e-8)
        @test ok && f < 1e-12 && norm(x) < 1e-5
    end
    ws = TopicModeling.LBFGS(6; m=10); x = zeros(6)
    lbfgs_allocated(ws, x)
    @test lbfgs_allocated(ws, x) == 0
    @test (@inferred TopicModeling.minimize!(wavy, copy(WAVY_X0), ws)) isa Tuple{Float64,Int,Bool}
end

@testset "Cholesky inverse and its repair" begin
    for n in (1, 5, 19, 40)                       # 40 > _CHOL_SMALL takes the LAPACK path
        B = randn(Xoshiro(n), n, n); A = B'B + I
        H = copy(A); W = zeros(n, n); dsave = zeros(n)
        @test TopicModeling._safe_cholesky!(H, dsave) === H
        @test UpperTriangular(H)' * UpperTriangular(H) ≈ A
        ld = TopicModeling._cholesky_inverse!(H, W)
        @test ld ≈ logdet(A) && H ≈ inv(A) && issymmetric(H)
        @test cholinv_allocated(H, A, dsave, W) == 0
    end
    for n in (3, 40)
        # Indefinite, and only the last pivot fails, after the earlier columns (and their
        # diagonal) have been overwritten by the first attempt.
        B = randn(Xoshiro(n), n, n); A = B'B ./ n + I; A[n, n] = -5.0
        @test !isposdef(Symmetric(A))
        R = copy(A)                               # the intended repair: diagonally dominant
        for i in 1:n
            R[i, i] = max(A[i, i], sum(abs, A[i, :]) - abs(A[i, i]))
        end
        H = copy(A); dsave = zeros(n)
        TopicModeling._safe_cholesky!(H, dsave)
        E = UpperTriangular(H)' * UpperTriangular(H) - R
        @test E ≈ E[1, 1] * I atol = 1e-10        # at most a small multiple of I is added
        @test -1e-10 <= E[1, 1] < 1e-4
        ld = TopicModeling._cholesky_inverse!(H, zeros(n, n))
        @test H ≈ inv(R + E[1, 1] * I) && ld ≈ logdet(R + E[1, 1] * I)
        # NaN anywhere: terminates, and the factor is a finite diagonal matrix
        for pos in ((1, 1), (n, n), (1, n))
            H = copy(A); H[pos...] = NaN; H[reverse(pos)...] = NaN
            TopicModeling._safe_cholesky!(H, dsave)
            @test all(isfinite, UpperTriangular(H)) && all(>(0), diag(H))
            @test isfinite(TopicModeling._cholesky_inverse!(H, zeros(n, n))) && all(isfinite, H)
        end
    end
end

@testset "co-occurrence Gram matrix" begin
    corpus, _, _ = simulate_lda(; D=60, K=3, V=50, doclen=5:40, rng=Xoshiro(1))
    docs = vcat(corpus.docs, [Document(Int32[7], Int32[1]), Document(Int32[3, 9], Int32[1, 1])])
    c = Corpus(docs, corpus.vocab)
    terms = [t for t in findall(>(0), termfreq(c)) if t % 7 != 0]      # drop some terms
    # dense reference: Σ_d (h_d h_dᵀ − diag(h_d)) / (n_d (n_d − 1)) over documents with n_d ≥ 2
    Qref = zeros(length(terms), length(terms))
    for d in c.docs
        n = ntokens(d)
        n >= 2 || continue
        h = zeros(nterms(c)); h[d.terms] = d.counts
        h = h[terms]
        Qref .+= (h * h' - Diagonal(h)) ./ (n * (n - 1))
    end
    Q1 = TopicModeling.cooccurrence_gram(c, terms; nthreads=1)
    @test Q1 ≈ Qref atol = 1e-13
    @test issymmetric(Q1)
    @test TopicModeling.cooccurrence_gram(c, terms; nthreads=3) == Q1
    phi1, a1 = spectral_init(c, 3; nthreads=1)
    phi3, a3 = spectral_init(c, 3; nthreads=3)
    @test a1 == a3 && phi1 == phi3 && all(sum(phi1; dims=2) .≈ 1)
end

@testset "EM stopping rule" begin
    calm = TopicModeling._calm_iterations
    @test calm(0, [-100.0], 1e-5) == 0
    @test calm(0, [-100.0, -100.0 + 1e-4], 1e-5) == 1
    @test calm(1, [-100.0, -100.0 + 1e-4], 1e-5) == 2
    @test calm(1, [-100.0, -99.0], 1e-5) == 0
    @test calm(1, [-100.0, -100.0], 0.0) == 0               # tol = 0 runs all iterations
    # An isolated dip of the (non-monotone) STM bound between larger increases used to end EM
    # with converged = true; now the count is reset by the next iteration.
    trace = [-100.0, -99.0, -98.99, -98.99 - 1e-5, -98.98, -98.975]
    counts = accumulate((c, n) -> calm(c, trace[1:n], 1e-6), 1:length(trace); init=0)
    @test counts == [0, 0, 0, 1, 0, 0]
    counts = accumulate((c, n) -> calm(c, [-100.0, -99.0, -99.0 - 1e-5, -99.0 - 1.5e-5][1:n], 1e-6), 1:4; init=0)
    @test counts == [0, 0, 1, 2]                            # a bound that has stalled does stop
end

@testset "STM and CTM: reproducibility, threads, arguments" begin
    rng = Xoshiro(5)
    D, K, V = 150, 4, 120
    x = randn(rng, D)
    Γ = zeros(2, K - 1); Γ[2, 1] = 1.0
    corpus, _, _ = simulate_logistic_normal(; D, K, V, mu=hcat(ones(D), x) * Γ, doclen=30:60, rng)
    X = reshape(x, :, 1)
    for fitter in ((; kw...) -> fit(STM, corpus, K; prevalence=X, keep_nu=true, kw...), (; kw...) -> fit(CTM, corpus, K; kw...))
        a = fitter(; iters=6, tol=0.0, nthreads=3)
        b = fitter(; iters=6, tol=0.0, nthreads=3)
        s = fitter(; iters=6, tol=0.0, nthreads=1)
        @test a.phi == b.phi && a.theta == b.theta && a.trace == b.trace
        @test maximum(abs, a.phi .- s.phi) < 1e-6
        @test a.trace ≈ s.trace rtol = 1e-9
        @test a.iterations == 6 && !a.converged
        @test all(isfinite, a.trace) && all(sum(a.phi; dims=2) .≈ 1)
        m = fitter(; iters=300, tol=1e-5)
        @test m.converged && m.iterations < 300
        Δ = abs.(diff(m.trace)) ./ abs.(m.trace[1:end-1])
        @test all(Δ[end-1:end] .< 1e-5) && findfirst(i -> Δ[i] < 1e-5 && Δ[i+1] < 1e-5, 1:length(Δ)-1) == length(Δ) - 1
        # exact zeros in the initial topics (as the unsmoothed M-step produces) are harmless
        init = copy(s.phi); init[:, 1:2:end] .= 0.0; init[1, :] .+= 1e-3; init0 = copy(init)
        z = fitter(; init, iters=3, tol=0.0)
        @test all(isfinite, z.trace) && all(isfinite, z.phi) && init == init0
    end
    m = fit(STM, corpus, K; prevalence=X, keep_nu=true, iters=3, tol=0.0)
    @test length(m.nu) == D && m.nu[1] != m.nu[2] && all(ν -> isposdef(Symmetric(ν)), m.nu)

    @test_throws ArgumentError fit(STM, corpus, K; gamma_prior=:lasso)
    @test_throws ArgumentError fit(STM, corpus, K; sigma_prior=1.5)
    @test_throws ArgumentError fit(STM, corpus, K; sigma_prior=-0.1)
    @test_throws ArgumentError fit(STM, corpus, K; nthreads=0)
    @test_throws ArgumentError fit(STM, corpus, K; prevalence=fill(NaN, D, 1))
    @test_throws "one row per document" fit(STM, corpus, K; prevalence=ones(D - 1, 1))
    @test_throws DimensionMismatch fit(STM, corpus, K; prevalence=ones(D + 1, 2))
    @test_throws ArgumentError fit(CTM, corpus, K; shrinkage=2)
    @test_throws ArgumentError fit(CTM, corpus, K; nthreads=0)
end

end
