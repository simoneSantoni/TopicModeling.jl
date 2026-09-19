# Model-agnostic evaluation: top words, coherence, held-out likelihood, topic recovery.

"""
    topwords(model; n=10) / topwords(phi, vocab; n=10)

The `n` most probable words of each topic, as a vector of string vectors.
"""
function topwords(phi::AbstractMatrix{<:Real}, vocab::AbstractVector{<:AbstractString}; n::Int=10)
    n = min(n, size(phi, 2))
    return [vocab[partialsortperm(view(phi, k, :), 1:n; rev=true)] for k in axes(phi, 1)]
end
topwords(m::AbstractTopicModel; n::Int=10) = topwords(topicword(m), vocabulary(m); n)

function topword_ids(phi::AbstractMatrix{<:Real}; n::Int=10)
    n = min(n, size(phi, 2))
    return [collect(partialsortperm(view(phi, k, :), 1:n; rev=true)) for k in axes(phi, 1)]
end

# Document frequencies and co-document frequencies for a set of term ids.
function _cooccurrence(c::Corpus, ids::Vector{Int})
    pos = Dict(t => i for (i, t) in enumerate(ids))
    n = length(ids)
    df = zeros(Int, n); co = zeros(Int, n, n)
    present = Int[]
    for doc in c.docs
        empty!(present)
        for t in doc.terms
            i = get(pos, Int(t), 0)
            i == 0 || push!(present, i)
        end
        for a in present
            df[a] += 1
            for b in present
                a < b && (co[a, b] += 1; co[b, a] += 1)
            end
        end
    end
    return pos, df, co
end

"""
    coherence(phi, corpus; n=10, measure=:npmi) -> Vector{Float64}

Per-topic coherence of the `n` top words, from document co-occurrence in `corpus`.

- `:npmi`  normalised pointwise mutual information (Bouma 2009; Lau, Newman & Baldwin 2014), in [-1, 1].
- `:umass` Mimno et al. (2011): `Σ_{i<j} log((D(w_i, w_j) + 1) / D(w_j))` with words ordered by probability.

Use the same reference corpus when comparing models or implementations.
"""
function coherence(phi::AbstractMatrix{<:Real}, c::Corpus; n::Int=10, measure::Symbol=:npmi)
    tops = topword_ids(phi; n)
    ids = sort!(unique(reduce(vcat, tops)))
    pos, df, co = _cooccurrence(c, ids)
    D = ndocs(c)
    return map(tops) do top
        s = 0.0; pairs = 0
        for a in 2:length(top), b in 1:(a - 1)
            i, j = pos[top[a]], pos[top[b]]       # j is the higher-ranked word
            if measure === :umass
                df[j] > 0 && (s += log((co[i, j] + 1) / df[j]))
            elseif measure === :npmi
                if co[i, j] == 0
                    s -= 1.0
                else
                    pij = co[i, j] / D
                    pmi = log(pij / ((df[i] / D) * (df[j] / D)))
                    s += pij >= 1 ? 1.0 : pmi / -log(pij)
                end
            else
                throw(ArgumentError("unknown coherence measure $measure"))
            end
            pairs += 1
        end
        pairs == 0 ? 0.0 : s / pairs
    end
end
coherence(m::AbstractTopicModel, c::Corpus; kwargs...) = coherence(topicword(m), c; kwargs...)

"""
    topic_diversity(phi; n=25)

Share of unique words among the top-`n` words of all topics (Dieng, Ruiz & Blei 2020):
1 when no two topics share a top word, `1/K` when all topics have the same ones.

# Examples
```jldoctest
julia> topic_diversity([0.5 0.4 0.1 0.0; 0.0 0.1 0.4 0.5]; n=2)
1.0

julia> topic_diversity([0.5 0.4 0.1 0.0; 0.4 0.5 0.0 0.1]; n=2)
0.5
```
"""
function topic_diversity(phi::AbstractMatrix{<:Real}; n::Int=25)
    tops = topword_ids(phi; n)
    return length(unique(reduce(vcat, tops))) / sum(length, tops)
end

# Number of document blocks of the held-out scoring loop. It is fixed, and the block sums are
# added in block order, so the result does not depend on `nthreads`.
const HELDOUT_BLOCKS = 64

# Σ_d Σ_w c_dw log(θ_dᵀ φ_w) over the documents `ids[j]` of `held`, with `theta[ids[j], :]`.
# Terms whose φ column is zero everywhere (never seen in training by a model that does not
# smooth its topics) cannot be scored and are counted instead.
# Returns (log-likelihood, tokens scored, tokens skipped).
function _heldout_loglik(theta::AbstractMatrix{Float64}, phi::AbstractMatrix{Float64}, held::Corpus,
                         ids::AbstractVector{Int}, nthreads::Int)
    K, V = size(phi)
    dead = [all(iszero, view(phi, :, w)) for w in 1:V]
    parts = balanced_chunks([length(held.docs[d].terms) + 1 for d in ids], HELDOUT_BLOCKS)
    ll = zeros(length(parts)); n = zeros(Int, length(parts)); skipped = zeros(Int, length(parts))
    _foreach_pooled(eachindex(parts), nthreads) do b, _
        l = 0.0; m = 0; s = 0
        for j in parts[b]
            d = ids[j]
            doc = held.docs[d]
            @inbounds for i in eachindex(doc.terms)
                w = doc.terms[i]
                if dead[w]
                    s += doc.counts[i]
                    continue
                end
                p = 0.0
                @simd for k in 1:K
                    p += theta[d, k] * phi[k, w]
                end
                l += doc.counts[i] * log(max(p, 1e-300))
                m += doc.counts[i]
            end
        end
        ll[b] = l; n[b] = m; skipped[b] = s
    end
    return sum(ll), sum(n), sum(skipped)       # `sum` of a vector: fixed (pairwise) order
end

function _perplexity(ll::Float64, n::Int, skipped::Int)
    skipped > 0 && @warn "heldout_perplexity: $skipped held-out tokens belong to terms with zero probability under every topic (terms unseen in training); they were skipped, $n tokens were scored"
    n > 0 || throw(ArgumentError("no held-out token to score: " * (skipped > 0 ?
        "all of them belong to terms with zero probability under every topic" :
        "the test documents have fewer than two tokens each")))
    return exp(-ll / n), ll / n
end

_check_heldout(phi, test::Corpus) = nterms(test) == size(phi, 2) ||
    throw(DimensionMismatch("the test corpus has $(nterms(test)) terms but the topics have $(size(phi, 2)); " *
                            "both must use the training vocabulary"))

"""
    heldout_perplexity(phi, alpha, test; frac=0.5, rng, nthreads) -> (perplexity, loglik_per_token)

Document-completion held-out evaluation (Wallach, Murray, Salakhutdinov & Mimno 2009,
§5.1): every test document's tokens are split at random; topic proportions are
inferred from one part with the topics `phi` fixed, and the other part is scored as
`Σ log(θ̂ᵀ φ_w)`. Only `phi` and a Dirichlet `alpha` are needed, so the same function
scores the output of any implementation on equal terms.

`test` must be indexed by the training vocabulary (`nterms(test) == size(phi, 2)`). Held-out
tokens of terms with zero probability under every topic (terms unseen in training, for models
that do not smooth their topics) cannot be scored: they are skipped, and a warning reports how
many. The result does not depend on `nthreads`.
"""
function heldout_perplexity(phi::AbstractMatrix{<:Real}, alpha, test::Corpus; frac::Real=0.5,
                            rng::AbstractRNG=Random.default_rng(), nthreads::Int=Threads.nthreads())
    _check_heldout(phi, test)
    K = size(phi, 1)
    topics = Matrix{Float64}(phi)
    observed, held = split_documents(test; frac, rng)
    theta, _ = infer_theta(observed, topics, _alpha_vector(alpha, K); nthreads)
    return _perplexity(_heldout_loglik(theta, topics, held, 1:ndocs(held), nthreads)...)
end

"""
    heldout_perplexity(model, test; frac=0.5, rng, kwargs...) -> (perplexity, loglik_per_token)

Document completion using the model's own prior: `transform(model, observed_half)` gives the
topic proportions, which then score the held-out half. This is the comparison of Blei &
Lafferty (2007) between LDA and the CTM. Extra keywords go to `transform`.
"""
function heldout_perplexity(m::AbstractTopicModel, test::Corpus; frac::Real=0.5,
                            rng::AbstractRNG=Random.default_rng(), kwargs...)
    phi = topicword(m)
    _check_heldout(phi, test)
    observed, held = split_documents(test; frac, rng)
    theta = transform(m, observed; kwargs...)
    nthreads = get(kwargs, :nthreads, Threads.nthreads())
    return _perplexity(_heldout_loglik(theta, phi, held, 1:ndocs(held), nthreads)...)
end

"""
    heldout_perplexity(model::DTM, test, times; frac=0.5, rng, kwargs...) -> (perplexity, loglik_per_token)

Document completion for the dynamic topic model: `times[d]` is the time label of test document
`d` (one of `model.periods`), and each document is completed and scored with the topics of its
own time slice. Extra keywords go to [`transform`](@ref).
"""
function heldout_perplexity(m::DTM, test::Corpus, times::AbstractVector; frac::Real=0.5,
                            rng::AbstractRNG=Random.default_rng(), kwargs...)
    _check_heldout(first(m.phi), test)
    observed, held = split_documents(test; frac, rng)
    theta = transform(m, observed, times; kwargs...)
    nthreads = get(kwargs, :nthreads, Threads.nthreads())
    ll = 0.0; n = 0; skipped = 0
    for (t, period) in enumerate(m.periods)                 # fixed slice order
        ids = findall(==(period), times)
        isempty(ids) && continue
        l, c, s = _heldout_loglik(theta, m.phi[t], held, ids, nthreads)
        ll += l; n += c; skipped += s
    end
    return _perplexity(ll, n, skipped)
end

heldout_perplexity(m::DTM, test::Corpus; kwargs...) =
    throw(ArgumentError("the topics of a DTM depend on the time slice: call " *
                        "heldout_perplexity(model, test, times) with one time label per test document"))

"""
    match_topics(phi_true, phi_est) -> (perm, distances)

Optimal one-to-one matching of estimated to true topics (Hungarian algorithm on total
variation distance). `phi_est[perm, :]` is aligned with `phi_true`; `distances[k]` is
the TV distance of the k-th matched pair, in [0, 1].
"""
function match_topics(phi_true::AbstractMatrix{<:Real}, phi_est::AbstractMatrix{<:Real})
    K = size(phi_true, 1)
    size(phi_est) == size(phi_true) || throw(DimensionMismatch("topic matrices must have equal size"))
    cost = [0.5 * sum(abs, view(phi_true, i, :) .- view(phi_est, j, :)) for i in 1:K, j in 1:K]
    perm = hungarian(cost)
    return perm, [cost[i, perm[i]] for i in 1:K]
end
