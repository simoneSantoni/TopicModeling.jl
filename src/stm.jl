# Structural Topic Model (Roberts, Stewart & Airoldi 2016, JASA) with topic-prevalence
# covariates:
#
#     η_d ~ N(X_d Γ, Σ),  θ_d = softmax([η_d; 0]),  z_dn ~ θ_d,  w_dn ~ β_{z_dn}
#
# Inference is the partially collapsed variational EM of the paper: for each document a
# Laplace approximation q(η_d) = N(λ_d, ν_d), with λ_d the MAP of the collapsed objective
# and ν_d the inverse Hessian there; closed-form updates for Γ, Σ and β.

"""
    STM

Fitted Structural Topic Model; obtain with `fit(STM, corpus, K; prevalence=X, ...)`.

Fields: `phi` (K×V), `theta` (D×K), `lambda` (D×(K-1) variational means of η), `nu`
(per-document (K-1)×(K-1) variational covariances; empty unless `keep_nu=true`), `gamma`
(P×(K-1) prevalence coefficients), `Sigma` ((K-1)×(K-1)), `mu` (D×(K-1) prior means
`XΓ`), `X` (D×P design matrix, intercept first), `trace` (approximate ELBO per iteration).
"""
struct STM <: AbstractTopicModel
    K::Int
    phi::Matrix{Float64}
    theta::Matrix{Float64}
    lambda::Matrix{Float64}
    nu::Vector{Matrix{Float64}}
    gamma::Matrix{Float64}
    Sigma::Matrix{Float64}
    mu::Matrix{Float64}
    X::Matrix{Float64}
    vocab::Vector{String}
    trace::Vector{Float64}
    iterations::Int
    converged::Bool
    elapsed::Float64
end

topicword(m::STM) = m.phi
doctopic(m::STM) = m.theta

# Collapsed per-document objective (negative log posterior of η up to a constant):
#   f(η) = −Σ_w c_w log Σ_k exp(η_k) β_kw + N log Σ_k exp(η_k) + ½ (η−μ)ᵀ Σ⁻¹ (η−μ),  η_K ≡ 0.
mutable struct LogisticNormalDoc
    K::Int
    n::Int
    N::Float64
    B::Matrix{Float64}            # K × n gathered topic columns
    counts::Vector{Float64}
    mu::Vector{Float64}
    siginv::Matrix{Float64}
    a::Vector{Float64}            # exp(η̃ − max)
    s::Vector{Float64}            # Σ_k a_k β_kw per term
    d::Vector{Float64}            # η − μ
    Sd::Vector{Float64}           # Σ⁻¹ (η − μ)
    acc::Vector{Float64}
end

function LogisticNormalDoc(K::Int)
    return LogisticNormalDoc(K, 0, 0.0, zeros(K, 0), Float64[], zeros(K - 1), zeros(K - 1, K - 1),
                             zeros(K), Float64[], zeros(K - 1), zeros(K - 1), zeros(K))
end

function load_doc!(o::LogisticNormalDoc, doc::Document, beta::Matrix{Float64})
    K = o.K
    n = length(doc.terms)
    if size(o.B, 2) < n
        o.B = Matrix{Float64}(undef, K, max(n, 2 * size(o.B, 2)))
        resize!(o.counts, size(o.B, 2)); resize!(o.s, size(o.B, 2))
    end
    N = 0.0
    @inbounds for i in 1:n
        w = doc.terms[i]
        o.counts[i] = doc.counts[i]
        N += doc.counts[i]
        @simd for k in 1:K
            o.B[k, i] = beta[k, w]
        end
    end
    o.n = n; o.N = N
    return o
end

# o.a = exp(η̃ − max η̃) (unnormalised θ); returns (Σ_k a_k, max η̃).
@inline function _exp_shifted!(o::LogisticNormalDoc, x::AbstractVector{Float64})
    K = o.K
    m = 0.0
    @inbounds for k in 1:(K - 1)
        m = max(m, x[k])
    end
    suma = 0.0
    @inbounds for k in 1:K
        o.a[k] = exp((k < K ? x[k] : 0.0) - m)
        suma += o.a[k]
    end
    return suma, m
end

# ll + c log s with few calls of `log`: most terms of a document occur once or twice, and their
# s are multiplied into the running product `p`, flushed before it can under- or overflow
# (s ≤ K, and s > 1e-50 is checked, so one more factor never leaves the floating-point range).
@inline function _add_clog(ll::Float64, p::Float64, c::Float64, s::Float64)
    if c == 1.0 && s > 1e-50
        p *= s
    elseif c == 2.0 && s > 1e-50
        p *= s * s
    else
        return ll + c * log(s), p
    end
    if !(1e-200 < p < 1e200)
        ll += log(p); p = 1.0
    end
    return ll, p
end

# Fill o.a (unnormalised θ), o.s, and return (Σ_w c_w log s_w − N log Σ a) = log p(w | η).
function _loglik_terms!(o::LogisticNormalDoc, x::AbstractVector{Float64})
    K = o.K
    suma, _ = _exp_shifted!(o, x)
    a, B = o.a, o.B
    ll = 0.0; p = 1.0
    @inbounds for i in 1:o.n
        s = 1e-300
        @simd for k in 1:K
            s += a[k] * B[k, i]
        end
        o.s[i] = s
        ll, p = _add_clog(ll, p, o.counts[i], s)
    end
    return ll + log(p) - o.N * log(suma), suma
end

# The same in one pass with o.acc[k] = Σ_w c_w β_kw / s_w, which the gradients need. Returns
# (Σ_w c_w log s_w, Σ a, max η̃) and leaves the normalisation to the caller (STM and CTM differ).
function _loglik_grad_terms!(o::LogisticNormalDoc, x::AbstractVector{Float64})
    K = o.K
    suma, m = _exp_shifted!(o, x)
    a, B, acc = o.a, o.B, o.acc
    fill!(acc, 0.0)
    ll = 0.0; p = 1.0
    @inbounds for i in 1:o.n
        s = 1e-300
        @simd for k in 1:K
            s += a[k] * B[k, i]
        end
        o.s[i] = s
        c = o.counts[i]
        r = c / s
        @simd for k in 1:K
            acc[k] += r * B[k, i]
        end
        ll, p = _add_clog(ll, p, c, s)
    end
    return ll + log(p), suma, m
end

function (o::LogisticNormalDoc)(g::Vector{Float64}, x::Vector{Float64})
    K1 = o.K - 1
    ll, suma, _ = _loglik_grad_terms!(o, x)
    @inbounds for k in 1:K1
        o.d[k] = x[k] - o.mu[k]
    end
    mul!(o.Sd, o.siginv, o.d)
    quad = 0.0
    @inbounds for k in 1:K1
        quad += o.d[k] * o.Sd[k]
        g[k] = -o.a[k] * o.acc[k] + o.N * o.a[k] / suma + o.Sd[k]
    end
    return -(ll - o.N * log(suma)) + 0.5 * quad
end

# Hessian of f at x (first K−1 coordinates), written into H (full storage). `Phi` is K × n
# scratch. Returns log p(w | η = x) and leaves o.a and o.s evaluated at x.
function hessian!(H::Matrix{Float64}, o::LogisticNormalDoc, x::Vector{Float64}, Phi::Matrix{Float64})
    K = o.K
    K1 = K - 1
    ll, suma = _loglik_terms!(o, x)
    fill!(o.acc, 0.0)
    @inbounds for i in 1:o.n
        sq = sqrt(o.counts[i]) / o.s[i]
        r = o.counts[i] / o.s[i]
        for k in 1:K
            e = o.a[k] * o.B[k, i]
            Phi[k, i] = sq * e                 # √c_w φ_wk
            o.acc[k] += r * e                  # Σ_w c_w φ_wk
        end
    end
    P = view(Phi, 1:K1, 1:o.n)
    BLAS.syrk!('U', 'N', 1.0, P, 0.0, H)       # Σ_w c_w φ_w φ_wᵀ
    @inbounds for j in 1:K1
        θj = o.a[j] / suma
        for i in 1:j
            θi = o.a[i] / suma
            H[i, j] += -o.N * θi * θj + o.siginv[i, j]
        end
        H[j, j] += o.N * θj - o.acc[j]
    end
    @inbounds for j in 1:K1, i in (j + 1):K1
        H[i, j] = H[j, i]
    end
    return ll
end

# Below this order a hand-written Cholesky beats LAPACK, whose per-call overhead dominates for
# the (K−1)×(K−1) Hessians and does not scale when many tasks call it at once.
const _CHOL_SMALL = 32

# In-place Cholesky H = UᵀU of the upper triangle; the strict lower triangle is not touched.
# Returns false if H is not (numerically) positive definite; U is then garbage.
function _cholesky_upper!(H::Matrix{Float64})
    n = size(H, 1)
    if n > _CHOL_SMALL
        _, info = LAPACK.potrf!('U', H)
        info == 0 || return false
    else
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                t = H[i, j]
                @simd for k in 1:(i - 1)
                    t -= H[k, i] * H[k, j]
                end
                H[i, j] = t / H[i, i]
            end
            s = H[j, j]
            @simd for k in 1:(j - 1)
                s -= H[k, j] * H[k, j]
            end
            s > 0 || return false
            H[j, j] = sqrt(s)
        end
    end
    @inbounds for j in 1:n
        isfinite(H[j, j]) || return false
    end
    return true
end

# Cholesky factor of the (symmetric, fully stored) Hessian in its upper triangle. If H is not
# positive definite (the objective is not convex) it is first made diagonally dominant, as the
# reference implementation does, then shifted. The strict lower triangle and `dsave` keep the
# original so that every attempt starts from an uncorrupted matrix.
function _safe_cholesky!(H::Matrix{Float64}, dsave::Vector{Float64})
    n = size(H, 1)
    @inbounds for j in 1:n
        dsave[j] = H[j, j]
    end
    _cholesky_upper!(H) && return H
    dmax = 1.0
    @inbounds for i in 1:n
        off = 0.0
        for j in 1:(i - 1)
            off += abs(H[i, j])
        end
        for j in (i + 1):n
            off += abs(H[j, i])
        end
        dsave[i] < off && (dsave[i] = off)
        dmax = max(dmax, abs(dsave[i]))
    end
    shift = 0.0
    for _ in 1:40
        @inbounds for j in 1:n
            for i in 1:(j - 1)
                H[i, j] = H[j, i]
            end
            H[j, j] = dsave[j] + shift
        end
        _cholesky_upper!(H) && return H
        shift = shift == 0 ? 1e-8 * dmax : 10 * shift
    end
    @inbounds for j in 1:n                      # non-finite input: fall back to a diagonal matrix
        for i in 1:(j - 1)
            H[i, j] = 0.0
        end
        H[j, j] = isfinite(dsave[j]) && dsave[j] > 0 ? sqrt(dsave[j]) : 1.0
    end
    return H
end

# H holds a Cholesky factor U in its upper triangle. Overwrite H with the full symmetric
# inverse (UᵀU)⁻¹ and return log|UᵀU|; `W` is n×n scratch. Allocation-free.
function _cholesky_inverse!(H::Matrix{Float64}, W::Matrix{Float64})
    n = size(H, 1)
    ld = 0.0
    @inbounds for j in 1:n
        ld += log(H[j, j])
    end
    if n > _CHOL_SMALL
        LAPACK.potri!('U', H)
    else
        @inbounds for c in 1:n                  # W = U⁻¹: back substitution, column sweeps
            for i in 1:(c - 1)
                W[i, c] = 0.0
            end
            W[c, c] = 1.0
            for k in c:-1:1
                xk = W[k, c] / H[k, k]
                W[k, c] = xk
                @simd for i in 1:(k - 1)
                    W[i, c] -= H[i, k] * xk
                end
            end
        end
        @inbounds for j in 1:n, i in 1:j
            H[i, j] = 0.0
        end
        @inbounds for k in 1:n, j in 1:k        # (UᵀU)⁻¹ = U⁻¹U⁻ᵀ = Σ_k w_k w_kᵀ
            wjk = W[j, k]
            @simd for i in 1:j
                H[i, j] += W[i, k] * wjk
            end
        end
    end
    @inbounds for j in 1:n, i in (j + 1):n
        H[i, j] = H[j, i]
    end
    return 2 * ld
end

# Variational Bayesian linear regression (Drugowitsch 2013) with an unpenalised intercept
# in column 1: y ~ N(Xw, τ⁻¹), w_j ~ N(0, (τα)⁻¹), Gamma hyper-priors on τ and α. This is
# the "Pooled" prevalence prior of stm: coefficients are shrunk by a learnt amount.
function vb_regression(X::Matrix{Float64}, y::AbstractVector{Float64}, XtX::Matrix{Float64};
                       a0=1e-2, b0=1e-4, c0=1e-2, d0=1e-4, maxiter::Int=1000, tol::Float64=1e-4)
    N, P = size(X)
    Xty = X' * y
    P == 1 && return XtX \ Xty
    w = zeros(P)
    Eα = c0 / d0
    aN = a0 + N / 2
    cN = c0 + (P - 1) / 2
    for _ in 1:maxiter
        A = copy(XtX)
        @inbounds for j in 2:P
            A[j, j] += Eα
        end
        F = cholesky!(Symmetric(A))
        wnew = F \ Xty
        Vd = diag(inv(F))
        r = y .- X * wnew
        w2 = sum(abs2, @view wnew[2:end])
        bN = b0 + 0.5 * (sum(abs2, r) + Eα * w2)
        dN = d0 + 0.5 * (aN / bN * w2 + sum(@view Vd[2:end]))
        Eα = cN / dN
        done = sum(abs, wnew .- w) < tol
        w = wnew
        done && break
    end
    return w
end

# EM stopping rule shared by STM and CTM: the number of consecutive iterations whose relative
# bound change was below `tol`; EM stops at two. The Laplace bound of STM is not monotone, and
# an isolated dip between larger increases must not pass for convergence, as it would with a
# test on a single |Δ|. (Asking for an increase would not do either: on small corpora the STM
# bound can creep down for hundreds of iterations.)
function _calm_iterations(calm::Int, trace::Vector{Float64}, tol::Float64)
    n = length(trace)
    n > 1 || return 0
    return abs(trace[n] - trace[n - 1]) < tol * abs(trace[n - 1]) ? calm + 1 : 0
end

# ss[1] += ss[2] + … over column chunks. Each entry sees the additions in the same order
# whatever the number of threads.
function _merge_ss!(ss::Vector{Matrix{Float64}}, nthreads::Int)
    A = ss[1]
    length(ss) > 1 || return A
    Threads.@threads for cols in chunks(size(A, 2), nthreads)
        for t in 2:length(ss)
            B = ss[t]
            @inbounds for j in cols
                @simd for k in axes(A, 1)
                    A[k, j] += B[k, j]
                end
            end
        end
    end
    return A
end

# Per-task buffers of the E-step, allocated once per fit.
struct STMWorkspace
    obj::LogisticNormalDoc
    opt::LBFGS
    x::Vector{Float64}
    H::Matrix{Float64}
    W::Matrix{Float64}
    dsave::Vector{Float64}
    Phi::Matrix{Float64}
end

STMWorkspace(K::Int, maxn::Int) =
    STMWorkspace(LogisticNormalDoc(K), LBFGS(K - 1; m=10), zeros(K - 1), zeros(K - 1, K - 1),
                 zeros(K - 1, K - 1), zeros(K - 1), zeros(K, maxn))

"""
    fit(STM, corpus, K; prevalence=nothing, kwargs...)

`prevalence` is a D×P matrix of document covariates (no intercept column; one is added).
With `prevalence=nothing` the model is a correlated topic model fitted by the STM algorithm.

Keywords: `init=:spectral` (or `:lda`, `:random`, or a K×V matrix), `iters=500`, `tol=1e-6`,
`gamma_prior=:pooled` (or `:ols`), `sigma_prior=0.0` (weight of the diagonal in the Σ
update, in [0, 1]), `doc_maxiter=500` (cap on the L-BFGS iterations of one document's
E-step), `keep_nu=false`, `nthreads`, `rng`, `verbose`.

EM stops when the relative change of the bound is below `tol` on two consecutive iterations
(the Laplace-approximate bound is not exactly monotone, and a single small dip is not
convergence). R's stm stops at the first change below `emtol=1e-5`, which is usually well
before the topics have settled: pass `tol=1e-5` for parity runs.
"""
function StatsAPI.fit(::Type{STM}, corpus::Corpus, K::Integer;
                      prevalence::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      init=:spectral, iters::Int=500, tol::Float64=1e-6,
                      gamma_prior::Symbol=:pooled, sigma_prior::Real=0.0, keep_nu::Bool=false,
                      doc_maxiter::Int=500, nthreads::Int=Threads.nthreads(),
                      rng::AbstractRNG=Random.default_rng(), verbose::Bool=false)
    t0 = time()
    K = Int(K)
    K >= 2 || throw(ArgumentError("need at least two topics"))
    gamma_prior in (:pooled, :ols) || throw(ArgumentError("gamma_prior must be :pooled or :ols, got :$gamma_prior"))
    0 <= sigma_prior <= 1 || throw(ArgumentError("sigma_prior must be in [0, 1]"))
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    D, V = ndocs(corpus), nterms(corpus)
    all(d -> !isempty(d), corpus.docs) || throw(ArgumentError("STM requires non-empty documents"))
    K1 = K - 1
    if prevalence !== nothing
        size(prevalence, 1) == D ||
            throw(DimensionMismatch("prevalence must have one row per document: got $(size(prevalence, 1)) rows for $D documents"))
        all(isfinite, prevalence) || throw(ArgumentError("prevalence must be finite (no NaN or Inf)"))
    end
    X = prevalence === nothing ? ones(D, 1) : hcat(ones(D), Matrix{Float64}(prevalence))
    P = size(X, 2)
    XtX = X' * X

    beta = if init isa AbstractMatrix
        size(init) == (K, V) || throw(DimensionMismatch("init must be K×V"))
        Matrix{Float64}(init)                      # a copy: the buffer is recycled below
    elseif init === :spectral
        first(spectral_init(corpus, K; nthreads))
    elseif init === :lda
        fit(LDA, corpus, K; method=:gibbs, iters=50, alpha=50 / K, eta=0.1, optimize_alpha=false, rng, nthreads).phi
    elseif init === :random
        normalize_rows!(rand(rng, K, V) .+ 0.1)
    else
        throw(ArgumentError("unknown init $init"))
    end

    lambda = zeros(D, K1)
    mu = zeros(D, K1)
    Sigma = Matrix(20.0I, K1, K1)
    Gamma = zeros(P, K1)
    nus = keep_nu ? [zeros(K1, K1) for _ in 1:D] : Matrix{Float64}[]

    parts = balanced_chunks([length(d.terms) + K for d in corpus.docs], nthreads)
    T = length(parts)
    beta_ss = [zeros(K, V) for _ in 1:T]
    sigma_ss = [zeros(K1, K1) for _ in 1:T]
    bounds = zeros(T)
    maxn = maximum(d -> length(d.terms), corpus.docs)
    ws = [STMWorkspace(K, maxn) for _ in 1:T]

    blas_threads = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    trace = Float64[]
    converged = false
    its = 0
    calm = 0
    try
        for it in 1:iters
            its = it
            F = cholesky(Symmetric(Sigma))
            siginv = inv(F)
            stm_estep!(corpus, beta, lambda, mu, siginv, logdet(F), parts, beta_ss, sigma_ss,
                       bounds, nus, ws, doc_maxiter)
            _merge_ss!(beta_ss, T)
            for t in 2:T
                sigma_ss[1] .+= sigma_ss[t]
            end
            push!(trace, sum(bounds))
            verbose && @printf("EM %4d  bound = %.4f\n", it, trace[end])

            # M-step.
            for k in 1:K1
                y = view(lambda, :, k)
                Gamma[:, k] = gamma_prior === :ols ? (XtX + 1e-8I) \ (X' * y) : vb_regression(X, y, XtX)
            end
            mul!(mu, X, Gamma)
            R = lambda .- mu
            Sigma = (sigma_ss[1] .+ R' * R) ./ D
            Sigma = (Sigma .+ Sigma') ./ 2
            if sigma_prior > 0
                Sigma = (1 - sigma_prior) .* Sigma .+ sigma_prior .* Diagonal(diag(Sigma))
            end
            # β ∝ expected counts, unsmoothed as in stm. The old β is the next accumulator.
            beta, beta_ss[1] = normalize_rows!(beta_ss[1]), beta

            calm = _calm_iterations(calm, trace, tol)
            if calm >= 2
                converged = true
                break
            end
        end
    finally
        BLAS.set_num_threads(blas_threads)
    end

    theta = Matrix{Float64}(undef, D, K)
    ηfull = zeros(K)
    for d in 1:D
        ηfull[1:K1] = view(lambda, d, :)
        theta[d, :] = softmax(ηfull)
    end
    return STM(K, beta, theta, lambda, nus, Gamma, Sigma, mu, X, corpus.vocab, trace, its, converged, time() - t0)
end

function stm_estep!(corpus, beta, lambda, mu, siginv, logdetSigma, parts, beta_ss, sigma_ss,
                    bounds, nus, ws, doc_maxiter)
    K = size(beta, 1)
    K1 = K - 1
    Threads.@threads for t in 1:length(parts)
        fill!(beta_ss[t], 0.0); fill!(sigma_ss[t], 0.0); bounds[t] = 0.0
        w = ws[t]
        obj, opt, x, H = w.obj, w.opt, w.x, w.H
        obj.siginv = siginv
        bs = beta_ss[t]
        bound = 0.0
        for d in parts[t]
            doc = corpus.docs[d]
            load_doc!(obj, doc, beta)
            @inbounds for k in 1:K1
                obj.mu[k] = mu[d, k]
                x[k] = lambda[d, k]                 # warm start from the previous EM iteration
            end
            minimize!(obj, x, opt; maxiter=doc_maxiter, gtol=1e-6, ftol=1e-13)
            ll = hessian!(H, obj, x, w.Phi)         # leaves o.a and o.s at the mode
            logdetH = _cholesky_inverse!(_safe_cholesky!(H, w.dsave), w.W)   # H is now ν
            # Expected topic-word counts at the mode, and this document's share of the bound:
            # log p(w|η) − ½(η−μ)ᵀΣ⁻¹(η−μ) − ½log|Σ| + ½log|ν|.
            @inbounds for i in 1:obj.n
                v = doc.terms[i]
                r = obj.counts[i] / obj.s[i]
                @simd for k in 1:K
                    bs[k, v] += r * obj.a[k] * obj.B[k, i]
                end
            end
            @inbounds for k in 1:K1
                obj.d[k] = x[k] - obj.mu[k]
            end
            mul!(obj.Sd, siginv, obj.d)
            bound += ll - 0.5 * dot(obj.d, obj.Sd) - 0.5 * logdetSigma - 0.5 * logdetH
            sigma_ss[t] .+= H
            @inbounds for k in 1:K1
                lambda[d, k] = x[k]
            end
            isempty(nus) || copyto!(nus[d], H)
        end
        bounds[t] = bound
    end
    return nothing
end

"""
    transform(model::STM, corpus; prevalence=nothing) -> D×K

Topic proportions for new documents: the MAP of η under the fitted prior, with prior mean
`XΓ` when covariates are given and the average fitted mean otherwise.
"""
function transform(m::STM, c::Corpus; prevalence::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                   nthreads::Int=Threads.nthreads())
    check_vocabulary(c, m.phi)
    K = m.K
    K1 = K - 1
    D = ndocs(c)
    prevalence === nothing || size(prevalence, 1) == D ||
        throw(DimensionMismatch("prevalence must have one row per document: got $(size(prevalence, 1)) rows for $D documents"))
    mu = prevalence === nothing ? repeat(mean(m.mu; dims=1), D) : hcat(ones(D), Matrix{Float64}(prevalence)) * m.gamma
    siginv = inv(cholesky(Symmetric(m.Sigma)))
    theta = zeros(D, K)
    Threads.@threads for part in balanced_chunks([length(d.terms) + K for d in c.docs], nthreads)
        obj = LogisticNormalDoc(K); obj.siginv = siginv
        opt = LBFGS(K1; m=10); x = zeros(K1); ηfull = zeros(K)
        for d in part
            load_doc!(obj, c.docs[d], m.phi)
            obj.mu .= view(mu, d, :)
            x .= obj.mu
            minimize!(obj, x, opt; maxiter=500, gtol=1e-6, ftol=1e-13)
            ηfull[1:K1] = x
            theta[d, :] = softmax(ηfull)
        end
    end
    return theta
end

"""
    estimate_effect(model; nsims=25, rng) -> (coef, se)

Regress topic proportions on the prevalence design matrix `model.X` with the "method of
composition" of stm's `estimateEffect`: draw η_d ~ N(λ_d, ν_d), map to θ, run OLS, and pool
point estimates and covariances over draws. Requires a model fitted with `keep_nu=true`.
Returns P×K matrices of coefficients and standard errors.
"""
function estimate_effect(m::STM; nsims::Int=25, rng::AbstractRNG=Random.default_rng())
    isempty(m.nu) && throw(ArgumentError("fit the model with keep_nu=true to propagate uncertainty"))
    D, K = size(m.theta)
    K1 = K - 1
    X = m.X
    P = size(X, 2)
    XtXinv = inv(Symmetric(X' * X))
    Ls = [cholesky(Symmetric(ν)).L for ν in m.nu]
    draws = zeros(nsims, P, K); vars = zeros(nsims, P, K)
    θ = zeros(D, K); ηfull = zeros(K)
    for s in 1:nsims
        for d in 1:D
            ηfull[1:K1] = view(m.lambda, d, :) .+ Ls[d] * randn(rng, K1)
            θ[d, :] = softmax(ηfull)
        end
        B = XtXinv * (X' * θ)
        resid = θ .- X * B
        for k in 1:K
            σ2 = sum(abs2, view(resid, :, k)) / (D - P)
            draws[s, :, k] = B[:, k]
            vars[s, :, k] = σ2 .* diag(XtXinv)
        end
    end
    coef = dropdims(mean(draws; dims=1); dims=1)
    within = dropdims(mean(vars; dims=1); dims=1)
    between = nsims > 1 ? dropdims(var(draws; dims=1); dims=1) : zero(within)
    return coef, sqrt.(within .+ (1 + 1 / nsims) .* between)
end
