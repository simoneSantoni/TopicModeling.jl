# Fixed 90/10 document splits, written once so every implementation trains and is scored
# on identical documents.
using TopicModeling, Random
data = joinpath(@__DIR__, "data")
for name in ("ap", "poliblog5k")
    c = read_ldac(joinpath(data, "$name.ldac"); vocab=joinpath(data, "$name.vocab"))
    keep = findall(d -> ntokens(d) >= 2, c.docs)
    c = c[keep]
    train, test, tr, te = train_test_split(c; test=0.1, rng=Xoshiro(20260918))
    write_ldac(joinpath(data, "$name.train.ldac"), train)
    write_ldac(joinpath(data, "$name.test.ldac"), test)
    write(joinpath(data, "$name.train.idx"), join(keep[tr], '\n'))
    write(joinpath(data, "$name.test.idx"), join(keep[te], '\n'))
    println(name, ": ", c, " -> train ", ndocs(train), " / test ", ndocs(test))
end
