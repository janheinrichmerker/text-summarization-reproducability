using Flux:OneHotMatrix, OneHotVector
using Flux.Losses:kldivergence
using Statistics:mean

function translation_loss(
    ŷ_probs::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector{<:Integer};
    label_smoothing_α::AbstractFloat=0.0,
    agg::Function=sum)
    
    if label_smoothing_α > 0
        return label_smoothing_loss(
            ŷ_probs, y, 
            label_smoothing_α, 
            agg=agg)
    else
        return negative_log_likelihood(ŷ_probs, y, agg=agg)
    end    
end

function label_smoothing_loss(
    ŷ_probs::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector{<:Integer},
    α::AbstractFloat;
    agg::Function=mean)
    
    vocab_size = size(ŷ_probs, 1)
    y_probs = OneHotMatrix(
        length(y),
        map(index -> OneHotVector(index, vocab_size), y)
    )
    y_probs = label_smoothing(y_probs, α)
    kldivergence(ŷ_probs, y_probs, agg=agg)
end

function negative_log_likelihood(
    ŷ_probs::AbstractMatrix{<:AbstractFloat},
    y::AbstractVector{<:Integer};
    agg::Function=mean)

    @assert size(ŷ_probs, 2) == size(y)
    N = size(y) # Batch size.
    L = [
        -ŷ_probs[y[i],i]
        for i ∈ 1:N
    ]
    return agg(L)
end

# Copied from https://github.com/FluxML/Flux.jl/blob/7a9a5eef2532d7029a960f7209b9a2e2fc76c29d/src/losses/functions.jl#L135, licensed under MIT License.
function label_smoothing(y::Union{AbstractArray,AbstractFloat}, α::AbstractFloat; dims::Integer=1)
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
