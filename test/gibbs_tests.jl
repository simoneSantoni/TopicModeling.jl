# Collapsed Gibbs sampler: partitioned threading, posterior averaging, reproducibility.

# Bytes allocated by one warmed call of the serial kernel (behind a function barrier).
function gibbs_sweep_allocations(args...)
    TopicModeling.gibbs_sweep!(args...)
    return @allocated TopicModeling.gibbs_sweep!(args...)
end

@testset "Gibbs sampler" begin
    TM = TopicModeling
    corpus, truth, _ = simulate_lda(; D=600, K=5, V=200, doclen=100, rng=Xoshiro(1))   # 60k tokens: 3 tasks
    N, D, V, K = ntokens(corpus), ndocs(corpus), nterms(corpus), 5
    doclen = [ntokens(d) for d in corpus.docs]

    # The partitioned sweep conserves the count tables (a lost or doubled update of a racing
    # pair of tasks would not): drive it exactly as `lda_gibbs` does.
    T = 3
    rng = Xoshiro(2)
    docptr = cumsum(vcat(1, doclen))
    w = reduce(vcat, [shuffle!(rng, tokens(d)) for d in corpus.docs])
    z = rand(rng, Int32(1):Int32(K), N)
    w0, z0 = copy(w), copy(z)
    block, newid = TM.word_blocks(termfreq(corpus), T)
    @test sort(newid) == 1:V && issorted(block[sortperm(newid)])          # blocks are contiguous column ranges
    mass = [sum(termfreq(corpus)[block .== b]) for b in 1:T]
    @test maximum(mass) - minimum(mass) <= maximum(termfreq(corpus))
    ptr, origin = TM.group_tokens!(w, z, docptr, block, newid, T)
    @test ptr[1, :] == docptr[1:D] && ptr[T + 1, :] == docptr[2:end] && all(diff(ptr; dims=1) .>= 0)
    src = [docptr[d] + origin[i] for d in 1:D for i in docptr[d]:(docptr[d + 1] - 1)]   # where token i came from
    @test sort(src) == 1:N && w == newid[w0[src]] && z == z0[src]
    @test all(issorted(block[w0[src[docptr[d]:(docptr[d + 1] - 1)]]]) for d in 1:D)
    @test all(block[w0[src[i]]] == b for d in 1:D for b in 1:T for i in ptr[b, d]:(ptr[b + 1, d] - 1))
    ndk = zeros(Int32, K, D); nkw = zeros(Int32, K, V); nk = zeros(Int, K)
    for d in 1:D, i in docptr[d]:(docptr[d + 1] - 1)
        ndk[z[i], d] += 1; nkw[z[i], w[i]] += 1; nk[z[i]] += 1
    end
    α = fill(0.1, K); η = 0.01
    parts = TM.balanced_chunks(doclen, T)
    rngs = [Xoshiro(t) for t in 1:T]
    ps = [zeros(K + 16) for _ in 1:T]; adks = [zeros(K + 16) for _ in 1:T]; invs = [zeros(K + 16) for _ in 1:T]
    local_nk = [zeros(Int, K + 16) for _ in 1:T]
    for _ in 1:5
        TM.gibbs_sweep_partitioned!(z, w, ptr, ndk, nkw, nk, local_nk, α, η, parts, rngs, ps, adks, invs)
    end
    @test z != z0[src]
    @test vec(sum(nkw; dims=1))[newid] == termfreq(corpus)
    @test vec(sum(ndk; dims=1)) == doclen
    @test nk == vec(sum(nkw; dims=2)) == vec(sum(ndk; dims=2)) && sum(nk) == N
    @test all(>=(0), nkw) && all(>=(0), ndk)
    tab = zeros(Int32, K, V)
    for i in 1:N
        tab[z[i], w[i]] += 1
    end
    @test tab == nkw                                                      # the tables still describe z

    # The serial kernel does not allocate.
    sptr = permutedims(hcat(docptr[1:D], docptr[2:end]))
    ndk .= 0; nkw .= 0; nk .= 0
    for d in 1:D, i in docptr[d]:(docptr[d + 1] - 1)
        ndk[z0[i], d] += 1; nkw[z0[i], w0[i]] += 1; nk[z0[i]] += 1
    end
    @test gibbs_sweep_allocations(z0, w0, sptr, 1, ndk, nkw, nk, α, η, 1:D, rngs[1], ps[1], adks[1], invs[1]) == 0
    @test vec(sum(nkw; dims=1)) == termfreq(corpus) && sum(nk) == N

    # Minka's fixed point: threaded over k, bitwise equal to the serial run.
    a1 = TM.optimize_alpha_fixedpoint!(fill(0.1, K), ndk, doclen; nthreads=1)
    a4 = TM.optimize_alpha_fixedpoint!(fill(0.1, K), ndk, doclen; nthreads=4)
    @test a1 == a4
    @test TM.gibbs_loglik(ndk, nkw, nk, doclen, a1, η; nthreads=1) == TM.gibbs_loglik(ndk, nkw, nk, doclen, a1, η; nthreads=4)

    # Same (seed, nthreads) ⇒ same model, threaded or not.
    for nt in (1, 3)
        a = fit(LDA, corpus, K; iters=40, burnin=10, eval_every=10, nthreads=nt, rng=Xoshiro(3))
        b = fit(LDA, corpus, K; iters=40, burnin=10, eval_every=10, nthreads=nt, rng=Xoshiro(3))
        @test a.phi == b.phi && a.theta == b.theta && a.trace == b.trace && a.alpha == b.alpha && a.z == b.z
        @test length(a.trace) == 4 && all(isfinite, a.trace)
    end

    # Threaded fit: topics recovered, z consistent with the documents, estimates stochastic.
    m = fit(LDA, corpus, K; iters=150, nthreads=3, rng=Xoshiro(4))
    @test mean(match_topics(truth, m.phi)[2]) < 0.06
    @test length.(m.z) == doclen && all(z -> all(in(1:K), z), m.z)
    @test all(sum(m.phi; dims=2) .≈ 1) && all(sum(m.theta; dims=2) .≈ 1)

    # nsamples = 1 is the estimate from the final state; the default averages ten states.
    # Both chains are the same, so `z` and `alpha` agree, and z determines the last-state φ.
    for nt in (1, 3)
        last = fit(LDA, corpus, K; iters=150, nsamples=1, nthreads=nt, rng=Xoshiro(4))
        avg = fit(LDA, corpus, K; iters=150, nthreads=nt, rng=Xoshiro(4))
        @test last.z == avg.z && last.alpha == avg.alpha && last.trace == avg.trace
        @test all(sum(last.phi; dims=2) .≈ 1) && all(sum(last.theta; dims=2) .≈ 1)
        @test all(sum(avg.phi; dims=2) .≈ 1) && all(sum(avg.theta; dims=2) .≈ 1)
        @test last.phi != avg.phi && maximum(abs, last.phi .- avg.phi) < 0.05
        ndk1 = zeros(Int, D, K)
        for d in 1:D, k in last.z[d]
            ndk1[d, k] += 1
        end
        @test last.theta ≈ (ndk1 .+ last.alpha') ./ (doclen .+ sum(last.alpha))
        r = Xoshiro(4)                                                    # z is in the documented token order
        nkw1 = zeros(Int, K, V)
        for (d, doc) in enumerate(corpus.docs), (k, v) in zip(last.z[d], shuffle!(r, tokens(doc)))
            nkw1[k, v] += 1
        end
        @test last.phi ≈ (nkw1 .+ last.eta) ./ (sum(nkw1; dims=2) .+ V * last.eta)
        @test mean(match_topics(truth, avg.phi)[2]) <= mean(match_topics(truth, last.phi)[2]) + 1e-3
    end
    short = fit(LDA, corpus, K; iters=5, rng=Xoshiro(5))                  # iters <= burnin: last state only
    @test short.phi == fit(LDA, corpus, K; iters=5, nsamples=1, rng=Xoshiro(5)).phi
    @test_throws ArgumentError fit(LDA, corpus, K; iters=5, nsamples=0)
    @test_throws ArgumentError fit(LDA, corpus, K; iters=5, sample_lag=0)
end
