# Class weight : w_j = n_samples / (n_classes * n_samples_j)

"""
    metrics(cm)
    metrics(cm, met...)
* `:TPR` true positive rate
* `:TNR` true negative rate
* `:NPV` negative predictive value
* `:FNR` false negative rate
* `:FPR` false positive rate
* `:FDR` false discovery rate
* `:FOR` false omission rate
* `:ACC` accuracy
* `(:Fscore, β)` Fβ score
* `:PT` prevalence threshold
"""
function metrics(cm::ConfMat{2, T}) where T
    TP = cm.cmat[1, 1] |> float
    TN = cm.cmat[2, 2] |> float
    FP = cm.cmat[2, 1] |> float
    FN = cm.cmat[1, 2] |> float
    #TP, TN, FP, FN
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

function metrics(cm::ConfMat)
    TP, TN, FP, FN = [], [], [], []
    # TN = []
    # FP = []
    # FN = []
    for l ∈ cm.labs
        push!(TP, cm.cmat[l, l] .|> float)
        push!(TN, sum(cm.cmat[Not(l), Not(l)]) .|> float)
        push!(FP, sum(cm.cmat[Not(l), l]) .|> float)
        push!(FN, sum(cm.cmat[l, Not(l)]) .|> float)
    end
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

function metrics(cm, met...)
    res = metrics(cm)
    for m ∈ met
        if !haskey(res, m)
            res[m] = _metric(res, m)
        end
    end
    res
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

# vérif les formules

_metric(b, m::Symbol) = _metric(b, Val{m}())

_metric(b, m::Val{:TPR}) = @. b[:TP] / nozero(b[:TP] + b[:FN])

_metric(b, m::Val{:TNR}) = @. b[:TN] / nozero(b[:TN] + b[:FP])

_metric(b, m::Val{:NPV}) = @. b[:TN] / nozero(b[:TN] + b[:FN])

_metric(b, m::Val{:FNR}) = @. b[:FN] / nozero(b[:FN] + b[:TP])

_metric(b, m::Val{:FPR}) = @. b[:FP] / nozero(b[:FP] + b[:TN])

_metric(b, m::Val{:FDR}) = @. b[:FP] / nozero(b[:FP] + b[:TP])

_metric(b, m::Val{:FOR}) = @. b[:FN] / nozero(b[:FN] + b[:TN])

_metric(b, m::Val{:ACC}) = begin
    @. (b[:TP] + b[:TN]) / nozero(b[:TP] + b[:TN] + b[:FP] + b[:FN])
end

_metric(b, m::Symbol, p) = _metric(b, Val{m}(), p)

_metric(b, t::Tuple{Symbol, Real}) = _metric(b, t...)

_metric(b, m::Val{:Fscore}, β) = begin
    a = (1 + β^2)
    @. a * b[:TP] / nozero(a * b[:TP] + β^2 * b[:FN] + b[:FP])
end

function _metric(b, m::Val{:PT})
    if !haskey(b, :FPR)
        b[:FPR] = _metric(b, :FPR)
    end
    if !haskey(b, :TPR)
        b[:TPR] = _metric(b, :TPR)
    end
    a = @. sqrt(b[:FPR])
    @. a / nozero(sqrt(b[:TPR]) + a)
end
