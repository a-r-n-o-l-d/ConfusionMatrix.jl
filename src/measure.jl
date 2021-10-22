function measure(cm::ConfMat{2, T}) where T
    TP = cm.cmat[1, 1]
    TN = cm.cmat[2, 2]
    FP = cm.cmat[2, 1]
    FN = cm.cmat[1, 2]
    TP, TN, FP, FN
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

# weight ?
function measure(cm::ConfMat)
    TP = Int[]
    TN = Int[]
    FP = Int[]
    FN = Int[]
    for l ∈ cm.labs
        push!(TP, cm.cmat[l, l])
        push!(TN, sum(cm.cmat[Not(l), Not(l)]))
        push!(FP, sum(cm.cmat[Not(l), l]))
        push!(FN, sum(cm.cmat[l, Not(l)]))
    end
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

function measure(cm, measr...)
    bse = measure(cm)
    mm = []
    for m ∈ measr
        push!(mm, m => _measure(bse, m))
    end
    merge(bse, Dict(mm...))
end

nozero(x) = begin
    x = float(x)
    iszero(x) ? eps(x) : x
end

#avoidzero2(x::Int) = (avoidzero2(x) |> float)

# function safediv(n, d)
#     d = float(d)
#     float(n) / (iszero(d) ? eps(d) : d)
# end

# macro safediv(d)
#     :(iszero.($d) ? eps.(float.($d)) : float.($d))
# end

_measure(b, m::Symbol) = _measure(b, Val{m}())

_measure(b, m::Val{:TPR}) = @. b[:TP] / nozero(b[:TP] + b[:FN])

_measure(b, m::Val{:TNR}) = @. b[:TN] / nozero(b[:TN] + b[:FP])

_measure(b, m::Val{:NPV}) = @. b[:TN] / nozero(b[:TN] + b[:FN])

_measure(b, m::Val{:FNR}) = @. b[:FN] / nozero(b[:FN] + b[:TP])

_measure(b, m::Val{:FPR}) = @. b[:FP] / nozero(b[:FP] + b[:TN])

_measure(b, m::Val{:FDR}) = @. b[:FP] / nozero(b[:FP] + b[:TP])

_measure(b, m::Val{:FOR}) = @. b[:FN] / nozero(b[:FN] + b[:TN])

_measure(b, m::Val{:PT}) = @. b[:FN] / nozero(b[:FN] + b[:TN])

_measure(b, m::Val{:ACC}) = begin
    @. (b[:TP] + b[:TN]) / nozero(b[:TP] + b[:TN] + b[:FP] + b[:FN])
end

_measure(b, m::Symbol, p) = _measure(b, Val{m}(), p)

_measure(b, t::Tuple{Symbol, Real}) = _measure(b, t...)

_measure(b, m::Val{:Fscore}, β) = begin
    @. (1 + β^2) * b[:TP] / nozero((1 + β^2) * b[:TP] + β^2 * b[:FN] + b[:FP])
end