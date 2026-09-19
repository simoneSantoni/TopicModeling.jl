# Limited-memory BFGS with a backtracking Armijo line search.
#
# The logistic-normal models (STM, CTM) solve one small smooth problem per
# document per EM iteration, and the DTM solves one large one per topic. All
# buffers live in a reusable workspace so the per-document solves allocate
# nothing.

struct LBFGS
    n::Int
    m::Int
    S::Matrix{Float64}
    Y::Matrix{Float64}
    rho::Vector{Float64}
    a::Vector{Float64}
    g::Vector{Float64}
    gnew::Vector{Float64}
    xnew::Vector{Float64}
    d::Vector{Float64}
end

function LBFGS(n::Int; m::Int=8)
    return LBFGS(n, m, zeros(n, m), zeros(n, m), zeros(m), zeros(m),
                 zeros(n), zeros(n), zeros(n), zeros(n))
end

"""
    minimize!(fg!, x, ws; maxiter=200, gtol=1e-6, ftol=1e-12) -> (f, iterations, converged)

Minimise a smooth function in place. `fg!(g, x)` must return the objective at
`x` and overwrite `g` with its gradient.
"""
function minimize!(fg!, x::Vector{Float64}, ws::LBFGS;
                   maxiter::Int=200, gtol::Float64=1e-6, ftol::Float64=1e-12)
    n, m = ws.n, ws.m
    S, Y, rho, a = ws.S, ws.Y, ws.rho, ws.a
    g, gnew, xnew, d = ws.g, ws.gnew, ws.xnew, ws.d
    f = fg!(g, x)
    stored = 0      # number of valid (s, y) pairs
    head = 0        # column of the most recent pair
    converged = false
    iters = 0
    for it in 1:maxiter
        iters = it
        gmax = 0.0
        @inbounds for i in 1:n
            gmax = max(gmax, abs(g[i]))
        end
        if gmax <= gtol
            converged = true
            break
        end

        # Two-loop recursion for d = -H g.
        @inbounds @simd for i in 1:n
            d[i] = -g[i]
        end
        @inbounds for j in 0:(stored - 1)
            c = mod1(head - j, m)
            s = 0.0
            @simd for i in 1:n
                s += S[i, c] * d[i]
            end
            a[c] = rho[c] * s
            ac = a[c]
            @simd for i in 1:n
                d[i] -= ac * Y[i, c]
            end
        end
        if stored > 0
            sy = 0.0
            yy = 0.0
            @inbounds @simd for i in 1:n
                sy += S[i, head] * Y[i, head]
                yy += Y[i, head] * Y[i, head]
            end
            γ = sy / yy
            @inbounds @simd for i in 1:n
                d[i] *= γ
            end
        end
        @inbounds for j in (stored - 1):-1:0
            c = mod1(head - j, m)
            s = 0.0
            @simd for i in 1:n
                s += Y[i, c] * d[i]
            end
            b = a[c] - rho[c] * s
            @simd for i in 1:n
                d[i] += b * S[i, c]
            end
        end

        dg = 0.0
        @inbounds @simd for i in 1:n
            dg += d[i] * g[i]
        end
        if !(dg < 0)            # not a descent direction: restart from steepest descent
            stored = 0
            dg = 0.0
            @inbounds @simd for i in 1:n
                d[i] = -g[i]
                dg -= g[i] * g[i]
            end
        end

        step = 1.0
        if stored == 0
            gn = sqrt(-dg)
            step = min(1.0, 1.0 / gn)
        end
        fnew = f
        ok = false
        for _ in 1:40
            @inbounds @simd for i in 1:n
                xnew[i] = x[i] + step * d[i]
            end
            fnew = fg!(gnew, xnew)
            if isfinite(fnew) && fnew <= f + 1e-4 * step * dg
                ok = true
                break
            end
            step *= 0.5
        end
        if !ok
            if stored > 0       # the curvature model may be stale: retry with steepest descent
                stored = 0
                continue
            end
            break
        end

        # Curvature test first: a rejected pair must not touch the memory, because with a full
        # memory column `c` still holds the oldest live pair (and its rho).
        sy = 0.0
        yy = 0.0
        @inbounds @simd for i in 1:n
            yi = gnew[i] - g[i]
            sy += (xnew[i] - x[i]) * yi
            yy += yi * yi
        end
        if sy > 1e-10 * yy && yy > 0
            c = mod1(head + 1, m)
            @inbounds @simd for i in 1:n
                S[i, c] = xnew[i] - x[i]
                Y[i, c] = gnew[i] - g[i]
            end
            rho[c] = 1.0 / sy
            head = c
            stored = min(stored + 1, m)
        end

        fold = f
        f = fnew
        copyto!(x, xnew)
        copyto!(g, gnew)
        if abs(fold - f) <= ftol * (abs(fold) + abs(f) + 1e-300)
            converged = true
            break
        end
    end
    return f, iters, converged
end
