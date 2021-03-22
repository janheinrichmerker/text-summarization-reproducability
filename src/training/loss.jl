using Flux:epseltype
using Flux.Losses:crossentropy, kldivergence
using Statistics:mean
using Transformers:logcrossentropy, logkldivergence

"""
Return the loss for a neural machine translation model, 
where ŷ is given as log().
If α ∈ (0,1), return KL-divergence between ŷ and label-smoothed y.
Otherwise, return cross entropy between ŷ and y without label smoothing.
"""
function logtranslationloss(ŷ, y; α=0.0, dims=1, agg=mean, ϵ=epseltype(ŷ))
    return translationloss(exp.(ŷ), y, α=α, dims=dims, agg=agg, ϵ=ϵ)
end

"""
Return the loss for a neural machine translation model.
If α ∈ (0,1), return KL-divergence between ŷ and label-smoothed y.
Otherwise, return cross entropy between ŷ and y without label smoothing.
"""
function translationloss(ŷ, y; α=0.0, dims=1, agg=mean, ϵ=epseltype(ŷ))
    if 0 < α < 1
        y = label_smoothing(y, α, dims=dims)
        return kldivergence(ŷ, y, dims=dims, agg=agg, ϵ=ϵ)
    else
        return crossentropy(y, ŷ, dims=dims, agg=agg, ϵ=ϵ)
    end    
end

"""
Smooth ground proof labels.
Copied from https://github.com/FluxML/Flux.jl/blob/7a9a5eef2532d7029a960f7209b9a2e2fc76c29d/src/losses/functions.jl#L135-L147
which was released under MIT License.
"""
function label_smoothing(y::Union{AbstractArray,Number}, α::Number; dims::Int=1)
    if !(0 < α < 1)
        throw(ArgumentError("α must be between 0 and 1"))
    end
    if dims == 0
        y_smoothed = y .* (1 - α) .+ α * 1 // 2
    elseif dims == 1
        y_smoothed = y .* (1 - α) .+ α * 1 // size(y, 1)
    else
        throw(ArgumentError("`dims` should be either 0 or 1"))
    end
    return y_smoothed
end
