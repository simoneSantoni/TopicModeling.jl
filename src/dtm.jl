# Dynamic Topic Model (Blei & Lafferty 2006, ICML).
#
#     β_{t,k} | β_{t-1,k} ~ N(β_{t-1,k}, σ² I),   φ_{t,k} = softmax(β_{t,k})
#     θ_d ~ Dirichlet(α),  z ~ θ_d,  w ~ φ_{t_d, z}
#
# Inference is the variational Kalman filtering of the paper: for every topic and word the
# chain β_{1:T} gets a Gaussian variational posterior equal to the Kalman-smoothed posterior
# under pseudo-observations β̂_t with variance ν̂². Two facts make this cheap here:
#
#   1. The smoothed covariance S depends only on (σ², ν̂², T) — it is the same for every word
#      and topic — and the smoothed mean is m = (S/ν̂²) β̂, an invertible linear map. So the
#      optimisation over pseudo-observations is an optimisation over the means m directly.
#   2. In terms of m the bound for a topic is concave,
#          Σ_t [ Σ_w n_tw m_tw − n_t log Σ_w exp(m_tw + S_tt/2) ] − ½ Σ_w m_wᵀ P m_w + const,
#      with P the tridiagonal random-walk precision, so a quasi-Newton method over the V×T
#      means replaces per-word conjugate gradients with Kalman passes in the inner loop.
#
# The variational family, and hence the optimum, are those of Blei & Lafferty.

"""
    DTM

Fitted Dynamic Topic Model; obtain with `fit(DTM, corpus, times, K; ...)`.

Fields: `phi` (length-T vector of K×V topic-word matrices), `theta` (D×K), `means` (length-K
vector of V×T variational means of β), `variances` (length-T marginal variances `S_tt`),
`times` (slice index per document), `periods` (the sorted unique time labels, a `Vector{P}`),
`alpha`, `chain_variance`, `obs_variance`, `trace` (ELBO per EM iteration).
"""
struct DTM{P} <: AbstractTopicModel
    K::Int
    phi::Vector{Matrix{Float64}}
    theta::Matrix{Float64}
    means::Vector{Matrix{Float64}}
    variances::Vector{Float64}
    times::Vector{Int}
    periods::Vector{P}
    alpha::Vector{Float64}
    chain_variance::Float64
    obs_variance::Float64
    vocab::Vector{String}
    trace::Vector{Float64}
    iterations::Int
    converged::Bool
    elapsed::Float64
end

"Topic-word matrix at time slice `t`; without `t`, the average over slices."
topicword(m::DTM, t::Integer) = m.phi[t]
topicword(m::DTM) = sum(m.phi) ./ length(m.phi)
doctopic(m::DTM) = m.theta

"""
    topwords(model::DTM, t; n=10)

Top words of every topic at time slice `t`.
"""
topwords(m::DTM, t::Integer; n::Int=10) = topwords(m.phi[t], m.vocab; n)

# Random-walk prior precision P = Dᵀ W D (D = first differences, W = diag(1/v₁, 1/σ², …)) and
# the smoothed covariance S = (P + I/ν̂²)⁻¹ shared by all words and topics.
function dtm_prior(T::Int, chain_variance::Float64, obs_variance::Float64, init_variance::Float64)
    W = fill(1 / chain_variance, T)
    W[1] = 1 / (init_variance + chain_variance)
    P = zeros(T, T)
    for t in 1:T
        P[t, t] = W[t] + (t < T ? W[t + 1] : 0.0)
        t < T && (P[t, t + 1] = P[t + 1, t] = -W[t + 1])
    end
    S = inv(Symmetric(P + I / obs_variance))
    # Per-word constant of the bound: E_q[log p(β)] + H(q) minus the quadratic in the means.
    perword = -0.5 * tr(P * S) + 0.5 * logdet(Symmetric(P)) + 0.5 * logdet(Symmetric(Matrix(S))) + T / 2
    return W, diag(S), perword
end

# Negative bound of one topic as a function of its V×T means (flattened column-major).
struct DTMTopicObjective
    n::Matrix{Float64}      # V×T expected counts
    nt::Vector{Float64}     # column sums
    W::Vector{Float64}
end

function (o::DTMTopicObjective)(g::Vector{Float64}, x::Vector{Float64})
    V, T = size(o.n)
    f = 0.0
    @inbounds for t in 1:T
        off = (t - 1) * V
        mx = -Inf
        for w in 1:V
            mx = max(mx, x[off + w])
        end
        # One `exp` per word: the softmax numerators are parked in `g` until `se` is known.
        se = 0.0
        for w in 1:V
            e = exp(x[off + w] - mx)
            g[off + w] = e
            se += e
        end
        nt = o.nt[t]
        c = nt / se
        lin = 0.0
        @simd for w in 1:V
            lin += o.n[w, t] * x[off + w]
            g[off + w] = c * g[off + w] - o.n[w, t]
        end
        f += nt * (mx + log(se)) - lin
        # Random-walk penalty ½ W_t ‖m_t − m_{t−1}‖² (m_0 ≡ 0).
        Wt = o.W[t]
        q = 0.0
        if t == 1
            @simd for w in 1:V
                δ = x[w]
                q += δ * δ
                g[w] += Wt * δ
            end
        else
            poff = off - V
            @simd for w in 1:V
                δ = x[off + w] - x[poff + w]
                q += δ * δ
                g[off + w] += Wt * δ
                g[poff + w] -= Wt * δ
            end
        end
        f += 0.5 * Wt * q
    end
    return f
end

# The E-step is cut into about this many blocks of documents (at least one per slice). The
# number is fixed, and block statistics are merged in block order, so that the fit does not
# depend on `nthreads`.
const DTM_ESTEP_BLOCKS = 32

# Documents of one time slice whose E-step runs as one unit of work, with the expected
# topic-term counts (K×V) they produce.
struct DTMBlock
    t::Int
    docs::Vector{Int}
    sstats::Matrix{Float64}
end

# Buffers owned by one E-step task: exp(E[log β_t]) of the slice at hand, the counts of the
# block at hand (kept here while they are accumulated, so that the scattered updates hit the
# same cache-resident matrix for every block) and the document workspace.
struct DTMTaskBuffers
    topics::Matrix{Float64}
    sstats::Matrix{Float64}
    doc::DocVB
end

# E-step with exp(E[log β_t]) = softmax(m_t) · exp(−S_tt/2): fills `counts` and the columns of
# `gamma` (K×D), returns the document part of the bound. Block statistics are merged slice by
# slice in block order, and block bounds are added in block order.
function dtm_estep!(counts::Vector{Matrix{Float64}}, gamma::Matrix{Float64}, bounds::Vector{Float64},
                    blocks::Vector{DTMBlock}, order::Vector{Int}, byslice::Vector{UnitRange{Int}},
                    bufs::Vector{DTMTaskBuffers}, corpus::Corpus, means::Vector{Matrix{Float64}},
                    Sdiag::Vector{Float64}, α::Vector{Float64}; maxiter::Int, tol::Float64)
    K, V = length(means), nterms(corpus)
    _foreach_pooled(order, length(bufs)) do b, p
        # Buffers are allocated by the task that uses them: small arrays allocated in a row by
        # one thread share cache lines, and the γ updates of neighbouring tasks then collide.
        isassigned(bufs, p) || (bufs[p] = DTMTaskBuffers(zeros(K, V), zeros(K, V), DocVB(K)))
        blk, buf = blocks[b], bufs[p]
        for k in 1:K
            softmax!(view(buf.topics, k, :), view(means[k], :, blk.t))
        end
        buf.topics .*= exp(-Sdiag[blk.t] / 2)
        fill!(buf.sstats, 0.0)
        bound = 0.0
        for d in blk.docs
            doc = corpus.docs[d]
            infer_doc!(buf.doc, doc, buf.topics, α; maxiter, tol)
            accumulate_sstats!(buf.sstats, buf.doc, doc)
            bound += doc_bound(buf.doc, doc, α)
            gamma[:, d] = buf.doc.gamma
        end
        copyto!(blk.sstats, buf.sstats)
        bounds[b] = bound
    end
    _foreach_pooled(eachindex(byslice), length(bufs)) do t, _
        for k in 1:K
            col = fill!(view(counts[k], :, t), 0.0)
            for b in byslice[t]
                col .+= view(blocks[b].sstats, k, :)
            end
        end
    end
    return sum(bounds)
end

"""
    fit(DTM, corpus, times, K; kwargs...)

`times[d]` is the time label of document `d` (any sortable values; the sorted unique labels
become slices `1:T`). Slices are treated as equally spaced, whatever their labels: with the
labels 1999, 2001 and 2010 the topics take one random-walk step from 1999 to 2001 and one from
2001 to 2010. Map the labels to periods of equal length first if the gaps matter.

Keywords: `alpha=0.01` (scalar or length-K vector), `chain_variance=0.005`, `obs_variance=0.5`,
`init_variance=1000 * chain_variance` (the defaults of Blei & Lafferty's implementation; all
must be positive), `iters=50` EM iterations, `tol=1e-5` relative ELBO change, `init=:lda` (a
short collapsed Gibbs run) or a K×V matrix of topic-word probabilities, `mstep_maxiter=100`
L-BFGS iterations per topic and M-step, `doc_maxiter=100` and `doc_tol=1e-4` (iteration limit of
the per-document update of γ and its stopping threshold, the mean absolute change of γ),
`nthreads`, `rng`, `verbose`.

Given the initial topics, the result does not depend on `nthreads`: the work is cut into
blocks and merged in an order that is independent of it. The default `init=:lda` does depend
on it, because the threaded Gibbs sampler is a different Markov chain for each number of
threads (and the 200 sweeps are a large share of the run time, so they are not run serially):
fix `nthreads` as well as `rng` to reproduce a fit, or pass the topics of your own
initialisation as `init`.
"""
function StatsAPI.fit(::Type{DTM}, corpus::Corpus, times::AbstractVector, K::Integer;
                      alpha=0.01, chain_variance::Real=0.005, obs_variance::Real=0.5,
                      init_variance::Real=1000 * chain_variance, iters::Int=50, tol::Float64=1e-5,
                      init=:lda, mstep_maxiter::Int=100, doc_maxiter::Int=100, doc_tol::Float64=1e-4,
                      nthreads::Int=Threads.nthreads(), rng::AbstractRNG=Random.default_rng(),
                      verbose::Bool=false)
    t0 = time()
    K = Int(K)
    K >= 2 || throw(ArgumentError("need at least two topics"))
    nthreads >= 1 || throw(ArgumentError("nthreads must be at least 1, got $nthreads"))
    for (name, v) in (("chain_variance", chain_variance), ("obs_variance", obs_variance), ("init_variance", init_variance))
        (v > 0 && isfinite(v)) || throw(ArgumentError("$name must be positive and finite, got $v"))
    end
    D, V = ndocs(corpus), nterms(corpus)
    D >= 1 || throw(ArgumentError("the corpus has no documents"))
    length(times) == D || throw(DimensionMismatch("times must have one entry per document"))
    periods = sort!(unique(map(identity, times)))
    T = length(periods)
    slice = Int[searchsortedfirst(periods, t) for t in times]
    α = _alpha_vector(alpha, K)
    all(a -> a > 0 && isfinite(a), α) || throw(ArgumentError("alpha must be positive and finite"))
    W, Sdiag, perword = dtm_prior(T, Float64(chain_variance), Float64(obs_variance), Float64(init_variance))

    phi0 = if init isa AbstractMatrix
        Matrix{Float64}(init)
    elseif init === :lda
        fit(LDA, corpus, K; method=:gibbs, iters=200, alpha=max(0.1, α[1]), eta=0.01,
            optimize_alpha=false, rng, nthreads).phi
    else
        throw(ArgumentError("unknown init $init"))
    end
    size(phi0) == (K, V) || throw(DimensionMismatch("init must be K×V"))
    means = map(1:K) do k
        lp = log.(max.(view(phi0, k, :), 1e-10))
        lp .-= mean(lp)
        repeat(lp, 1, T)
    end

    # Blocks of documents of near-equal work within each slice, largest first for the tasks.
    blocks = DTMBlock[]
    byslice = Vector{UnitRange{Int}}(undef, T)
    work = Int[]
    for t in 1:T
        ids = findall(==(t), slice)
        weights = [length(corpus.docs[d].terms) + 1 for d in ids]
        lo = length(blocks) + 1
        for part in balanced_chunks(weights, cld(DTM_ESTEP_BLOCKS, T))
            push!(blocks, DTMBlock(t, ids[part], zeros(K, V)))
            push!(work, sum(view(weights, part)))
        end
        byslice[t] = lo:length(blocks)
    end
    order = sortperm(work; rev=true)
    bounds = zeros(length(blocks))
    bufs = Vector{DTMTaskBuffers}(undef, min(nthreads, length(blocks)))
    counts = [zeros(V, T) for _ in 1:K]
    gamma = zeros(K, D)
    # One task per topic in the M-step, slowest topic of the previous M-step first. `minimize!`
    # resets the L-BFGS memory, so it does not matter which workspace a topic gets.
    opts = [LBFGS(V * T; m=8) for _ in 1:min(K, nthreads)]
    mstep_order = collect(1:K)
    mstep_iters = zeros(Int, K)

    trace = Float64[]
    converged = false
    its = 0
    for it in 1:iters
        its = it
        docbound = dtm_estep!(counts, gamma, bounds, blocks, order, byslice, bufs, corpus, means, Sdiag, α;
                              maxiter=doc_maxiter, tol=doc_tol)

        # The ELBO is evaluated here, where the document and topic parts are consistent.
        chain = 0.0
        for k in 1:K
            m = means[k]
            for t in 1:T
                δ2 = t == 1 ? sum(abs2, view(m, :, 1)) : sum(abs2, view(m, :, t) .- view(m, :, t - 1))
                chain -= 0.5 * W[t] * δ2
            end
        end
        push!(trace, docbound + chain + K * V * perword)
        verbose && @printf("EM %3d  ELBO = %.4f\n", it, trace[end])

        # M-step: one concave problem per topic, warm-started at the current means.
        _foreach_pooled(mstep_order, length(opts)) do k, p
            obj = DTMTopicObjective(counts[k], vec(sum(counts[k]; dims=1)), W)
            _, n, _ = minimize!(obj, vec(means[k]), opts[p]; maxiter=mstep_maxiter, gtol=1e-4, ftol=1e-12)
            mstep_iters[k] = n
        end
        sortperm!(mstep_order, mstep_iters; rev=true)

        if it > 1 && abs(trace[end] - trace[end - 1]) < tol * abs(trace[end - 1])
            converged = true
            break
        end
    end

    phi = map(1:T) do t
        φ = zeros(K, V)
        for k in 1:K
            softmax!(view(φ, k, :), view(means[k], :, t))
        end
        φ
    end
    theta = permutedims(gamma ./ sum(gamma; dims=1))
    return DTM(K, phi, theta, means, Sdiag, slice, periods, α, Float64(chain_variance),
               Float64(obs_variance), corpus.vocab, trace, its, converged, time() - t0)
end

"""
    transform(model::DTM, corpus, times) -> D×K

Topic proportions of new documents, each inferred with the topics of its own time slice.
`times` must use labels seen in training (`model.periods`).
"""
function transform(m::DTM, c::Corpus, times::AbstractVector; kwargs...)
    length(times) == ndocs(c) || throw(DimensionMismatch("times must have one entry per document"))
    nterms(c) == size(first(m.phi), 2) ||
        throw(DimensionMismatch("the corpus has $(nterms(c)) terms but the model has $(size(first(m.phi), 2))"))
    theta = zeros(ndocs(c), m.K)
    for (t, period) in enumerate(m.periods)
        ids = findall(==(period), times)
        isempty(ids) && continue
        theta[ids, :] = first(infer_theta(c[ids], m.phi[t], m.alpha; kwargs...))
    end
    unknown = setdiff(unique(times), m.periods)
    isempty(unknown) || throw(ArgumentError("time labels not seen in training: $unknown"))
    return theta
end
