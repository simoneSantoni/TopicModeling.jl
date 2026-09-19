# Correlated Topic Model (Blei & Lafferty 2007, Annals of Applied Statistics).
#
#     η_d ~ N(μ, Σ),  θ_d = softmax([η_d; 0]),  z ~ θ_d,  w ~ β_z
#
# Variational EM with the mean-field family of the paper, q(η_d) = N(λ_d, diag(ν²_d)), and
# the same first-order bound on E[log Σ_k exp η_k]. The token responsibilities φ and the bound
# parameter ζ have closed-form optima, so they are substituted out and each document is a
# smooth problem in (λ, log ν²):
#
#   L_d = Σ_w c_w log Σ_k exp(λ_k) β_kw − N log Σ_k exp(λ_k + ν²_k/2)
#         − ½ (λ−μ)ᵀΣ⁻¹(λ−μ) − ½ Σ_k ν²_k (Σ⁻¹)_kk + ½ Σ_k log ν²_k − ½ log|Σ| + (K−1)/2.

"""
    CTM

Fitted Correlated Topic Model; obtain with `fit(CTM, corpus, K; ...)`.

Fields: `phi` (K×V), `theta` (D×K), `lambda`/`nu2` (D×(K-1) variational means and
variances of η), `mu`, `Sigma` (logistic-normal parameters over the first K-1 topics; the
K-th is the reference with η ≡ 0), `trace` (ELBO per iteration).
"""
struct CTM <: AbstractTopicModel
    K::Int
    phi::Matrix{Float64}
    theta::Matrix{Float64}
    lambda::Matrix{Float64}
    nu2::Matrix{Float64}
    mu::Vector{Float64}
    Sigma::Matrix{Float64}
    vocab::Vector{String}
    trace::Vector{Float64}
    iterations::Int
    converged::Bool
    elapsed::Float64
end

topicword(m::CTM) = m.phi
doctopic(m::CTM) = m.theta

"Correlation matrix of the first K-1 topics' log-odds against the reference topic."
function topic_correlations(m::CTM)
    s = sqrt.(diag(m.Sigma))
    return m.Sigma ./ (s * s')
end

# Negative per-document bound in x = [λ; log ν²], without the −½log|Σ| + (K−1)/2 constant.
struct CTMDocObjective
    doc::LogisticNormalDoc
    siginv_diag::Vector{Float64}
end

function (c::CTMDocObjective)(g::Vector{Float64}, x::Vector{Float64})
    o = c.doc
    K1 = o.K - 1
    # Σ_w c_w log Σ_k exp(λ_k) β_kw: the kernel works with a_k = exp(λ_k − m).
    ll, _, m = _loglik_grad_terms!(o, x)
    data = ll + o.N * m
    # Z = Σ_k exp(λ_k + ν²_k/2), reference topic contributing exp(0). `g` is scratch for ν² and
    # the exponentials until the gradient is written, so each is evaluated once.
    mz = 0.0
    @inbounds for k in 1:K1
        ν2 = exp(x[K1 + k])
        g[K1 + k] = ν2
        g[k] = x[k] + 0.5 * ν2
        mz = max(mz, g[k])
    end
    Z = exp(-mz)
    @inbounds for k in 1:K1
        g[k] = exp(g[k] - mz)
        Z += g[k]
    end
    logZ = mz + log(Z)
    @inbounds for k in 1:K1
        o.d[k] = x[k] - o.mu[k]
    end
    mul!(o.Sd, o.siginv, o.d)
    quad = 0.0; trace = 0.0; ent = 0.0
    @inbounds for k in 1:K1
        ν2 = g[K1 + k]
        e = g[k] / Z                           # share of topic k in Z
        quad += o.d[k] * o.Sd[k]
        trace += ν2 * c.siginv_diag[k]
        ent += x[K1 + k]
        g[k] = -(o.a[k] * o.acc[k] - o.N * e - o.Sd[k])
        g[K1 + k] = -(-0.5 * o.N * e * ν2 - 0.5 * c.siginv_diag[k] * ν2 + 0.5)
    end
    return -(data - o.N * logZ - 0.5 * quad - 0.5 * trace + 0.5 * ent)
end

"""
    fit(CTM, corpus, K; kwargs...)

Keywords: `init=:spectral` (or `:lda`, `:random`, a K×V matrix), `iters=500`, `tol=1e-5`,
`shrinkage=0.0` (weight of the diagonal in the Σ update, in [0, 1]), `doc_maxiter=500` (cap
on the L-BFGS iterations of one document's E-step), `nthreads`, `rng`, `verbose`.

EM stops when the relative change of the ELBO is below `tol` on two consecutive iterations.
"""
function StatsAPI.fit(::Type{CTM}, corpus::Corpus, K::Integer; init=:spectral, iters::Int=500,
                      tol::Float64=1e-5, shrinkage::Real=0.0, doc_maxiter::Int=500,
                      nthreads::Int=Threads.nthreads(), rng::AbstractRNG=Random.default_rng(),
                      verbose::Bool=false)
    t0 = time()
    K = Int(K)
    K >= 2 || throw(ArgumentError("need at least two topics"))
    0 <= shrinkage <= 1 || throw(ArgumentError("shrinkage must be in [0, 1]"))
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1"))
    D, V = ndocs(corpus), nterms(corpus)
    all(d -> !isempty(d), corpus.docs) || throw(ArgumentError("CTM requires non-empty documents"))
    K1 = K - 1
    beta = if init isa AbstractMatrix
        size(init) == (K, V) || throw(DimensionMismatch("init must be K×V"))
        Matrix{Float64}(init)                      # a copy: the buffer is recycled below
    elseif init === :spectral
        first(spectral_init(corpus, K; nthreads))
    elseif init === :lda
        fit(LDA, corpus, K; method=:gibbs, iters=100, rng, nthreads).phi
    elseif init === :random
        normalize_rows!(rand(rng, K, V) .+ 0.1)
    else
        throw(ArgumentError("unknown init $init"))
    end

    lambda = zeros(D, K1); lognu2 = fill(log(1.0), D, K1)
    mu = zeros(K1); Sigma = Matrix(1.0I, K1, K1)
    parts = balanced_chunks([length(d.terms) + K for d in corpus.docs], nthreads)
    T = length(parts)
    beta_ss = [zeros(K, V) for _ in 1:T]
    bounds = zeros(T)
    ws = [CTMWorkspace(K) for _ in 1:T]
    trace = Float64[]
    converged = false
    its = 0
    calm = 0
    for it in 1:iters
        its = it
        F = cholesky(Symmetric(Sigma))
        siginv = inv(F)
        ctm_estep!(corpus, beta, lambda, lognu2, mu, siginv, parts, beta_ss, bounds, ws, doc_maxiter)
        push!(trace, sum(bounds) + D * (-0.5 * logdet(F) + K1 / 2))
        verbose && @printf("EM %4d  ELBO = %.4f\n", it, trace[end])

        _merge_ss!(beta_ss, T)
        beta, beta_ss[1] = normalize_rows!(beta_ss[1]), beta    # the old β is the next accumulator
        mu = vec(mean(lambda; dims=1))
        R = lambda .- mu'
        Sigma = (R' * R .+ Diagonal(vec(sum(exp.(lognu2); dims=1)))) ./ D
        Sigma = (Sigma .+ Sigma') ./ 2
        shrinkage > 0 && (Sigma = (1 - shrinkage) .* Sigma .+ shrinkage .* Diagonal(diag(Sigma)))

        calm = _calm_iterations(calm, trace, tol)
        if calm >= 2
            converged = true
            break
        end
    end
    theta = _softmax_rows(lambda)
    return CTM(K, beta, theta, lambda, exp.(lognu2), mu, Sigma, corpus.vocab, trace, its, converged, time() - t0)
end

function _softmax_rows(lambda::Matrix{Float64})
    D, K1 = size(lambda)
    theta = Matrix{Float64}(undef, D, K1 + 1)
    ηfull = zeros(K1 + 1)
    for d in 1:D
        ηfull[1:K1] = view(lambda, d, :)
        theta[d, :] = softmax(ηfull)
    end
    return theta
end

# Per-task buffers of the E-step.
struct CTMWorkspace
    doc::LogisticNormalDoc
    opt::LBFGS
    x::Vector{Float64}
end

CTMWorkspace(K::Int) = CTMWorkspace(LogisticNormalDoc(K), LBFGS(2(K - 1); m=10), zeros(2(K - 1)))

function ctm_estep!(corpus, beta, lambda, lognu2, mu, siginv, parts, beta_ss, bounds, ws, doc_maxiter;
                    collect_ss::Bool=true)
    K = size(beta, 1)
    K1 = K - 1
    sdiag = diag(siginv)
    Threads.@threads for t in 1:length(parts)
        collect_ss && fill!(beta_ss[t], 0.0)
        bounds[t] = 0.0
        doc, opt, x = ws[t].doc, ws[t].opt, ws[t].x
        doc.siginv = siginv
        doc.mu .= mu
        obj = CTMDocObjective(doc, sdiag)
        bound = 0.0
        for d in parts[t]
            load_doc!(doc, corpus.docs[d], beta)
            @inbounds for k in 1:K1
                x[k] = lambda[d, k]; x[K1 + k] = lognu2[d, k]
            end
            f, _, _ = minimize!(obj, x, opt; maxiter=doc_maxiter, gtol=1e-6, ftol=1e-13)
            bound -= f
            @inbounds for k in 1:K1
                lambda[d, k] = x[k]; lognu2[d, k] = x[K1 + k]
            end
            if collect_ss
                _loglik_terms!(doc, x)
                bs = beta_ss[t]
                terms = corpus.docs[d].terms
                @inbounds for i in 1:doc.n
                    w = terms[i]
                    r = doc.counts[i] / doc.s[i]
                    @simd for k in 1:K
                        bs[k, w] += r * doc.a[k] * doc.B[k, i]
                    end
                end
            end
        end
        bounds[t] = bound
    end
    return nothing
end

"""
    transform(model::CTM, corpus) -> D×K

Topic proportions of new documents: variational inference under the fitted (μ, Σ, β).
"""
function transform(m::CTM, c::Corpus; nthreads::Int=Threads.nthreads(), doc_maxiter::Int=500)
    check_vocabulary(c, m.phi)
    D = ndocs(c)
    K1 = m.K - 1
    lambda = repeat(m.mu', D); lognu2 = zeros(D, K1)
    parts = balanced_chunks([length(d.terms) + m.K for d in c.docs], nthreads)
    siginv = inv(cholesky(Symmetric(m.Sigma)))
    ctm_estep!(c, m.phi, lambda, lognu2, m.mu, siginv, parts, Matrix{Float64}[], zeros(length(parts)),
               [CTMWorkspace(m.K) for _ in parts], doc_maxiter; collect_ss=false)
    return _softmax_rows(lambda)
end
