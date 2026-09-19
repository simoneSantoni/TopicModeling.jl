# Input validation of the corpus layer, the file readers and held-out evaluation. The inference
# kernels gather topic columns with `@inbounds`, so a bad term id must never get past
# construction: every check here is on the constructor or reader, never on a kernel.
# Runs on its own or from runtests.jl.
using TopicModeling
using Random, Test

# Write `text` to a temporary file, call `f(path)`, clean up.
function with_file(f, text::AbstractString)
    path, io = mktemp()
    try
        write(io, text); close(io)
        return f(path)
    finally
        rm(path; force=true)
    end
end

@testset "robustness" begin
    vocab = ["w$i" for i in 1:40]

    @testset "Document and Corpus validation" begin
        d = Document([1, 5, 9], [2, 1, 3])
        @test d.terms == [1, 5, 9] && d.counts == [2, 1, 3] && ntokens(d) == 6
        @test_throws ArgumentError Document([5, 1], [1, 1])              # unsorted
        @test_throws ArgumentError Document([1, 1], [1, 1])              # duplicate
        @test_throws ArgumentError Document([1, 2], [1, 0])              # zero count
        @test_throws ArgumentError Document([1, 2], [1, -3])             # negative count
        @test_throws ArgumentError Document([0, 2], [1, 1])              # ids are 1-based
        @test_throws ArgumentError Document([-4], [1])
        @test_throws DimensionMismatch Document([1, 2], [1])
        @test_throws ArgumentError Document([3, 0, 3])                   # token form: bad id
        t = Document([7, 2, 7, 7, 2, 40])                                # token form: any order
        @test t.terms == [2, 7, 40] && t.counts == [2, 3, 1]
        @test isempty(Document(Int[])) && isempty(Document(Int[], Int[]))

        # out-of-vocabulary ids are rejected when the corpus is built (they used to reach the
        # kernels: a segmentation fault, or a plausible θ read from stray memory)
        @test_throws ArgumentError Corpus([Document([1, 2_000_000_000], [2, 3])], vocab)
        @test_throws ArgumentError Corpus([Document([1, 2], [1, 1]), Document([45], [1])], vocab)
        @test_throws ArgumentError Corpus([Document([41], [1])], vocab)
        @test_throws ArgumentError Corpus(Document[Document([1], [1])], String[])
        @test_throws Exception Document([1, 3_000_000_000], [1, 1])      # does not fit an Int32
        ok = Corpus([Document([1, 40], [1, 1]), Document(Int[], Int[])], vocab)
        @test ndocs(ok) == 2 && nterms(ok) == 40 && ntokens(ok) == 2
        @test Corpus(view(ok.docs, 1:2), view(vocab, 1:40)) isa Corpus   # converting method
        @test_throws DimensionMismatch Corpus([1 0; 0 2], ["a"])

        # sub-corpora share the vocabulary and stay valid
        c, _, _ = simulate_lda(; D=30, K=3, V=40, doclen=20, rng=Xoshiro(1))
        sub = c[[3, 1, 3]]
        @test ndocs(sub) == 3 && sub.vocab === c.vocab && sub[1] === c[3]
        @test ndocs(c[2:end]) == 29 && c[end] === c[30] && ndocs(c[Int[]]) == 0
        @test_throws BoundsError c[[31]]
    end

    @testset "splitting" begin
        c, _, _ = simulate_lda(; D=30, K=3, V=40, doclen=20, rng=Xoshiro(1))
        o1, h1 = split_documents(c; rng=Xoshiro(3)); o2, h2 = split_documents(c; rng=Xoshiro(3))
        @test all(o1[i].terms == o2[i].terms && o1[i].counts == o2[i].counts &&
                  h1[i].terms == h2[i].terms && h1[i].counts == h2[i].counts for i in 1:30)
        @test all(ntokens(o1[i]) == 10 && ntokens(h1[i]) == 10 for i in 1:30)
        @test all(issorted(o1[i].terms) && allunique(o1[i].terms) for i in 1:30)
        o, h = split_documents(c; frac=0.0, rng=Xoshiro(3))
        @test all(ntokens(o[i]) == 1 for i in 1:30)                      # never an empty half
        @test_throws ArgumentError split_documents(c; frac=1.5)
        short = Corpus([Document([2], [1]), Document(Int[], Int[])], c.vocab)
        o, h = split_documents(short; rng=Xoshiro(1))
        @test ntokens(o) == 1 && ntokens(h) == 0

        tr, te, itr, ite = train_test_split(c[1:2]; test=0.9, rng=Xoshiro(1))
        @test ndocs(tr) == 1 && ndocs(te) == 1 && sort(vcat(itr, ite)) == 1:2
        tr, te, _, _ = train_test_split(c[1:2]; test=0.0, rng=Xoshiro(1))
        @test ndocs(tr) == 1 && ndocs(te) == 1
        tr, te, itr, ite = train_test_split(c; test=0.2, rng=Xoshiro(1))
        @test ndocs(tr) == 24 && ndocs(te) == 6 && isempty(intersect(itr, ite))
        @test_throws ArgumentError train_test_split(c[1:1])
        @test_throws ArgumentError train_test_split(c[Int[]])
        @test_throws ArgumentError train_test_split(c; test=1.2)
    end

    @testset "heldout_perplexity" begin
        c, phi, _ = simulate_lda(; D=60, K=3, V=40, doclen=30, rng=Xoshiro(1))
        ppl, ll = heldout_perplexity(phi, 0.1, c; rng=Xoshiro(2), nthreads=1)
        @test isfinite(ppl) && ppl < 40 && ll ≈ -log(ppl)
        @test heldout_perplexity(phi, 0.1, c; rng=Xoshiro(2), nthreads=3) == (ppl, ll)   # not a function of nthreads
        # the reference formula, serially
        obs, held = split_documents(c; rng=Xoshiro(2))
        theta, _ = TopicModeling.infer_theta(obs, phi, fill(0.1, 3); nthreads=1)
        ref = sum(n * log(sum(theta[d, :] .* phi[:, w])) for d in 1:60 for (w, n) in zip(held[d].terms, held[d].counts)) / ntokens(held)
        @test ll ≈ ref

        # vocabulary mismatch
        @test_throws DimensionMismatch heldout_perplexity(phi[:, 1:39], 0.1, c)
        @test_throws DimensionMismatch heldout_perplexity(hcat(phi, zeros(3, 1)), 0.1, c)
        lda = fit(LDA, c, 3; method=:vb, iters=3, rng=Xoshiro(1))
        small, _ = prune(c; max_terms=30)
        @test_throws DimensionMismatch heldout_perplexity(lda, small)

        # terms with zero probability under every topic are skipped, counted and reported
        dead = copy(phi); dead[:, [2, 7]] .= 0; dead ./= sum(dead; dims=2)
        r = @test_logs (:warn, r"held-out tokens.*skipped") heldout_perplexity(dead, 0.1, c; rng=Xoshiro(2), nthreads=2)
        @test all(isfinite, r) && r[1] < 40
        # the same through a model: the STM does not smooth its topics, so a term that never
        # occurs in training has probability zero (the perplexity used to be ~1e240)
        wide = Corpus(c.docs, vcat(c.vocab, "never"))
        stm = fit(STM, wide[1:40], 3; iters=3, rng=Xoshiro(1))
        test = Corpus([Document(vcat(d.terms, 41), vcat(d.counts, 6)) for d in c.docs[41:60]], wide.vocab)
        if all(iszero, view(stm.phi, :, 41))
            r = @test_logs (:warn, r"skipped") heldout_perplexity(stm, test; rng=Xoshiro(2))
        else
            r = heldout_perplexity(stm, test; rng=Xoshiro(2))
        end
        @test r[1] < 1e3

        # nothing to score
        onlyfirst = zeros(3, 40); onlyfirst[:, 1] .= 1
        others = Corpus([Document([2, 3], [2, 2]), Document([5], [4])], c.vocab)
        @test_logs (:warn, r"skipped") (@test_throws ArgumentError heldout_perplexity(onlyfirst, 0.1, others; rng=Xoshiro(1)))
        singles = Corpus([Document([2], [1]), Document([5], [1])], c.vocab)
        @test_throws ArgumentError heldout_perplexity(phi, 0.1, singles; rng=Xoshiro(1))
    end

    @testset "read_ldac" begin
        # a blank line is an empty document; trailing blank lines are not documents
        c = with_file(read_ldac, "2 0:1 3:2\n\n1 1:4\n0\n\n\n")
        @test ndocs(c) == 4 && nterms(c) == 4
        @test isempty(c[2]) && isempty(c[4]) && c[3].terms == [2] && c[3].counts == [4]
        # unsorted pairs are sorted, a repeated term is merged, zero counts are dropped
        c = with_file(read_ldac, "4 5:1 2:2 5:3 0:0\n")
        @test c[1].terms == [3, 6] && c[1].counts == [2, 4]
        for (text, needle) in ["3 0:1 1:1\n" => "line 1",                  # declared N ≠ pairs
                               "1 0:1\n1 0:1 2:2\n" => "line 2",
                               "1 0:1\n1 -1:2\n" => "line 2",              # negative id
                               "1 0:-2\n" => "line 1",                     # negative count
                               "1 0:1\n\n1 3\n" => "line 3",               # malformed pairs
                               "1 a:1\n" => "line 1",
                               "1 1:\n" => "line 1",
                               "1 1:2:3\n" => "line 1",
                               "x 0:1\n" => "line 1"]
            err = try with_file(read_ldac, text); nothing catch e; e end
            @test err isa ArgumentError && occursin(needle, err.msg)
        end
        # vocabulary file: a trailing blank line is not a term; too short a file is an error
        with_file("a\nb\nc\n\n") do vpath
            c = with_file(p -> read_ldac(p; vocab=vpath), "1 2:1\n")
            @test c.vocab == ["a", "b", "c"]
            @test_throws ArgumentError with_file(p -> read_ldac(p; vocab=vpath), "1 3:1\n")
        end
        # round trip: exact with the vocabulary file; without it the unused last terms are lost
        orig = Corpus([Document([1, 3], [2, 1]), Document(Int[], Int[]), Document([2], [5])], ["a", "b", "c", "unused"])
        mktempdir() do dir
            path = write_ldac(joinpath(dir, "c.ldac"), orig; vocab=joinpath(dir, "c.vocab"))
            back = read_ldac(path; vocab=joinpath(dir, "c.vocab"))
            @test back.vocab == orig.vocab && ndocs(back) == 3 && isempty(back[2])
            @test all(back[i].terms == orig[i].terms && back[i].counts == orig[i].counts for i in 1:3)
            bare = read_ldac(path)
            @test nterms(bare) == 3 && all(bare[i].terms == orig[i].terms for i in 1:3)
        end
    end

    @testset "read_uci" begin
        c = with_file(read_uci, "3\n4\n5\n1 2 3\n1 1 1\n3 4 2\n1 2 1\n3 1 0\n")
        @test ndocs(c) == 3 && nterms(c) == 4 && isempty(c[2])
        @test c[1].terms == [1, 2] && c[1].counts == [1, 4]                # sorted, repeated triple merged
        @test c[3].terms == [4] && c[3].counts == [2]                      # zero count dropped
        for (text, needle) in ["2\n3\n1\n3 1 1\n" => "docID",              # docID > D
                               "2\n3\n1\n0 1 1\n" => "docID",
                               "2\n3\n1\n1 4 1\n" => "wordID",             # wordID > W
                               "2\n3\n1\n1 0 1\n" => "wordID",
                               "2\n3\n1\n1 1 -1\n" => "count",
                               "2\n3\n1\n1 1\n" => "line 4",               # malformed triple
                               "2\n3\n1\n1 x 1\n" => "line 4",
                               "2\n3\n2\n1 1 1\n" => "NNZ",                # truncated file
                               "2\nthree\n1\n1 1 1\n" => "W",              # malformed header
                               "" => "D"]
            err = try with_file(read_uci, text); nothing catch e; e end
            @test err isa ArgumentError && occursin(needle, err.msg)
        end
        with_file("a\nb\nc\n\n") do vpath
            @test with_file(p -> read_uci(p; vocab=vpath), "1\n3\n1\n1 3 2\n").vocab == ["a", "b", "c"]
            @test_throws ArgumentError with_file(p -> read_uci(p; vocab=vpath), "1\n4\n1\n1 3 2\n")
        end
    end
end

@testset "vocabulary mismatch in transform" begin
    c, _, _ = simulate_lda(D=60, K=3, V=40, doclen=30, rng=Xoshiro(1))
    other = Corpus(c.docs[1:5], vcat(c.vocab, ["extra"]))          # V + 1 terms
    lda = fit(LDA, c, 3; iters=5, rng=Xoshiro(1))
    vb = fit(LDA, c, 3; method=:vb, iters=3, rng=Xoshiro(1))
    stm = fit(STM, c, 3; iters=2)
    ctm = fit(CTM, c, 3; iters=2)
    for m in (lda, vb, stm, ctm)
        @test_throws DimensionMismatch transform(m, other)
        @test size(transform(m, c[1:5])) == (5, 3)
    end
end
