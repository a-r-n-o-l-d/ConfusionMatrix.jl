function measure(cm::ConfMat{2, T}) where T
    TP = cm.cmat[1, 1]
    TN = cm.cmat[2, 2]
    FP = cm.cmat[2, 1]
    FN = cm.cmat[1, 2]
    TP, TN, FP, FN
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

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

_measure(b, m::Symbol) = _measure(b, Val{m}())

_measure(b, m::Val{:TPR}) = @. b[:TP] / (b[:TP] + b[:FN])

_measure(b, m::Val{:TNR}) = @. b[:TN] / (b[:TN] + b[:FP])

_measure(b, m::Val{:NPV}) = @. b[:TN] / (b[:TN] + b[:FN])

_measure(b, m::Val{:FNR}) = @. b[:FN] / (b[:FN] + b[:TP])

_measure(b, m::Val{:FPR}) = @. b[:FP] / (b[:FP] + b[:TN])

_measure(b, m::Val{:FDR}) = @. b[:FP] / (b[:FP] + b[:TP])

_measure(b, m::Val{:FOR}) = @. b[:FN] / (b[:FN] + b[:TN])

_measure(b, m::Val{:PT}) = @. b[:FN] / (b[:FN] + b[:TN])

_measure(b, m::Val{:ACC}) = begin
    @. (b[:TP] + b[:TN]) / (b[:TP] + b[:TN] + b[:FP] + b[:FN])
end

_measure(b, m::Symbol, p) = _measure(b, Val{m}(), p)

_measure(b, t::Tuple{Symbol, Real}) = _measure(b, t...)

_measure(b, m::Val{:Fscore}, β) = begin
    @. (1 + β^2) * b[:TP] / ((1 + β^2) * b[:TP] + β^2 * b[:FN] + b[:FP])
end