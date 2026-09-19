# Synthetic corpora drawn from each model's generative process, for tests and for
# parameter-recovery benchmarks where the ground truth is known.

_doclen(rng, n::Integer) = Int(n)
_doclen(rng, r::AbstractRange) = rand(rng, r)

function _draw_document(rng::AbstractRNG, theta::AbstractVector{Float64}, phi::AbstractMatrix{Float64},
                        n::Int, cdf_theta::Vector{Float64}, cdf_phi::Matrix{Float64})
    K, V = size(phi)
    cumsum!(cdf_theta, theta)
    toks = Vector{Int32}(undef, n)
    for i in 1:n
        k = min(searchsortedfirst(cdf_theta, rand(rng) * cdf_theta[end]), K)
        col = view(cdf_phi, k, :)
        toks[i] = min(searchsortedfirst(col, rand(rng) * col[end]), V)
    end
    return Document(toks)
end

_vocab(V::Int) = ["w$(lpad(i, ndigits(V), '0'))" for i in 1:V]

"""
    bars_topics(side=5) -> (2side)×(side²) matrix

The "bars" topics of Griffiths & Steyvers (2004): the vocabulary is a `side`×`side`
pixel grid and each topic is uniform over one row or one column.
"""
function bars_topics(side::Int=5)
    phi = zeros(2side, side^2)
    for r in 1:side, c in 1:side
        v = (r - 1) * side + c
        phi[r, v] = 1 / side
        phi[side + c, v] = 1 / side
    end
    return phi
end

"""
    simulate_lda(; D=500, K=10, V=1000, doclen=100, alpha=0.1, eta=0.05, topics=nothing, rng)
        -> (corpus, phi, theta)
"""
function simulate_lda(; D::Int=500, K::Int=10, V::Int=1000, doclen=100, alpha=0.1, eta::Real=0.05,
                      topics::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      rng::AbstractRNG=Random.default_rng())
    phi = topics === nothing ? permutedims(reduce(hcat, [randdirichlet(rng, fill(eta, V)) for _ in 1:K])) :
                               Matrix{Float64}(topics)
    K, V = size(phi)
    α = _alpha_vector(alpha, K)
    theta = Matrix{Float64}(undef, D, K)
    cdfφ = cumsum(phi; dims=2); cdfθ = zeros(K)
    docs = Vector{Document}(undef, D)
    for d in 1:D
        θ = randdirichlet(rng, α)
        theta[d, :] = θ
        docs[d] = _draw_document(rng, θ, phi, _doclen(rng, doclen), cdfθ, cdfφ)
    end
    return Corpus(docs, _vocab(V)), phi, theta
end

"""
    simulate_logistic_normal(; D, K, V, mu, Sigma, doclen, eta, topics, rng) -> (corpus, phi, theta)

Documents with logistic-normal topic proportions: `η_d ~ N(μ_d, Σ)` in `K-1`
dimensions, `θ_d = softmax([η_d; 0])`. `mu` is a length `K-1` vector (CTM) or a
D×(K-1) matrix of document-specific means (STM: `X * Γ`).
"""
function simulate_logistic_normal(; D::Int=500, K::Int=5, V::Int=500, mu=zeros(K - 1),
                                  Sigma::AbstractMatrix{<:Real}=Matrix(1.0I, K - 1, K - 1),
                                  doclen=100, eta::Real=0.05,
                                  topics::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                                  rng::AbstractRNG=Random.default_rng())
    phi = topics === nothing ? permutedims(reduce(hcat, [randdirichlet(rng, fill(eta, V)) for _ in 1:K])) :
                               Matrix{Float64}(topics)
    K, V = size(phi)
    L = cholesky(Symmetric(Matrix{Float64}(Sigma))).L
    theta = Matrix{Float64}(undef, D, K)
    cdfφ = cumsum(phi; dims=2); cdfθ = zeros(K)
    docs = Vector{Document}(undef, D)
    ηfull = zeros(K)
    for d in 1:D
        μd = mu isa AbstractMatrix ? view(mu, d, :) : mu
        ηfull[1:(K - 1)] = μd .+ L * randn(rng, K - 1)
        ηfull[K] = 0.0
        θ = softmax(ηfull)
        theta[d, :] = θ
        docs[d] = _draw_document(rng, θ, phi, _doclen(rng, doclen), cdfθ, cdfφ)
    end
    return Corpus(docs, _vocab(V)), phi, theta
end

"""
    simulate_dtm(; T=10, docs_per_slice=100, K=5, V=500, chain_variance=0.01, init_scale=2.0,
                 doclen=100, alpha=0.1, rng) -> (corpus, times, phi, theta)

Dynamic topic model (Blei & Lafferty 2006): the natural parameters of each topic follow
a Gaussian random walk, `β_{t,k} ~ N(β_{t-1,k}, σ² I)`, and `phi[t][k, :] = softmax(β_{t,k})`.
`times[d] ∈ 1:T` is the slice of document `d`.
"""
function simulate_dtm(; T::Int=10, docs_per_slice::Int=100, K::Int=5, V::Int=500,
                      chain_variance::Real=0.01, init_scale::Real=2.0, doclen=100, alpha=0.1,
                      rng::AbstractRNG=Random.default_rng())
    α = _alpha_vector(alpha, K)
    β = init_scale .* randn(rng, K, V)
    phi = Vector{Matrix{Float64}}(undef, T)
    docs = Document[]; times = Int[]
    theta = Matrix{Float64}(undef, T * docs_per_slice, K)
    cdfθ = zeros(K)
    for t in 1:T
        t > 1 && (β .+= sqrt(chain_variance) .* randn(rng, K, V))
        φ = similar(β)
        for k in 1:K
            softmax!(view(φ, k, :), view(β, k, :))
        end
        phi[t] = φ
        cdfφ = cumsum(φ; dims=2)
        for _ in 1:docs_per_slice
            θ = randdirichlet(rng, α)
            push!(docs, _draw_document(rng, θ, φ, _doclen(rng, doclen), cdfθ, cdfφ))
            push!(times, t)
            theta[length(docs), :] = θ
        end
    end
    return Corpus(docs, _vocab(V)), times, phi, theta
end
