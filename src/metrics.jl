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
    TP = cm.counts[1, 1] |> float
    TN = cm.counts[2, 2] |> float
    FP = cm.counts[2, 1] |> float
    FN = cm.counts[1, 2] |> float
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

function metrics(cm::ConfMat)
    TP, TN, FP, FN = [], [], [], []
    for l ∈ cm.labels
        push!(TP, cm.counts[l, l] .|> float)
        push!(TN, sum(cm.counts[Not(l), Not(l)]) .|> float)
        push!(FP, sum(cm.counts[Not(l), l]) .|> float)
        push!(FN, sum(cm.counts[l, Not(l)]) .|> float)
    end
    Dict(:TP => TP, :TN => TN, :FP => FP, :FN => FN)
end

function metrics(cm::ConfMat, met...)
    metrics(cm) |> m -> metrics!(m, met...)
end

tosymbol(x) = x
function tosymbol(x::Tuple{Symbol, Real})
    s, v = x
    st = string(s) * "_" * string(v)
    st = replace(st, "." => "_")
    Symbol(st)
end

function metrics!(mm::Dict, met...)
    for m ∈ met
        if !haskey(mm, m)
            # F-score special case handling : tuple is transformed to a symbol
            k = tosymbol(m)
            mm[k] = metric(mm, m)
        end
    end
    mm
end

@inline nozero(x) = iszero(float(x)) ? eps(float(x)) : x

#avoidzero2(x::Int) = (avoidzero2(x) |> float)

# function safediv(n, d)
#     d = float(d)
#     float(n) / (iszero(d) ? eps(d) : d)
# end

# macro safediv(d)
#     :(iszero.($d) ? eps.(float.($d)) : float.($d))
# end

# vérif les formules

metric(b, m::Symbol) = metric(b, Val{m}())

metric(b, m::Val{:TPR}) = @. b[:TP] / nozero(b[:TP] + b[:FN])

metric(b, m::Val{:TNR}) = @. b[:TN] / nozero(b[:TN] + b[:FP])

metric(b, m::Val{:NPV}) = @. b[:TN] / nozero(b[:TN] + b[:FN])

metric(b, m::Val{:FNR}) = @. b[:FN] / nozero(b[:FN] + b[:TP])

metric(b, m::Val{:FPR}) = @. b[:FP] / nozero(b[:FP] + b[:TN])

metric(b, m::Val{:FDR}) = @. b[:FP] / nozero(b[:FP] + b[:TP])

metric(b, m::Val{:FOR}) = @. b[:FN] / nozero(b[:FN] + b[:TN])

metric(b, m::Val{:ACC}) = 
    @. (b[:TP] + b[:TN]) / nozero(b[:TP] + b[:TN] + b[:FP] + b[:FN])

#_metric(b, t::Int) = println("pouet")

metric(b, m::Symbol, p) = metric(b, Val{m}(), p)

metric(b, t::Tuple{Symbol, Real}) = metric(b, t...)

metric(b, m::Val{:Fscore}, β) = begin
    a = (1 + β^2)
    @. a * b[:TP] / nozero(a * b[:TP] + β^2 * b[:FN] + b[:FP])
end

function metric(b, m::Val{:PT})
    fpr = haskey(b, :FPR) ? b[:FPR] : metric(b, :FPR)
    tpr = haskey(b, :TPR) ? b[:TPR] : metric(b, :TPR)
    a = @. sqrt(fpr)
    @. a / nozero(sqrt(tpr) + a)
end
