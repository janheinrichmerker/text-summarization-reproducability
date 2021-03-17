using Transformers.Basic
using Transformers.Basic:AbstractEmbed
using Flux:@functor

struct Encoder
    embed::AbstractEmbed
    layers::Array{<:Transformer} 
end

@functor Encoder

function (encoder::Encoder)(indices)
    if typeof(encoder.embed) <: CompositeEmbedding
        indices = (
            tok = indices,
            segment = fill(1, length(indices))
        )
    end
    
    embedding = encoder.embed(indices)
    for transformer ∈ encoder.layers
        embedding = transformer(embedding)
    end
    return embedding
end

function Encoder(embed::AbstractEmbed, size::Int, head::Int, hs::Int, ps::Int, layer::Int; act=relu, pdrop=0.1) 
    return Encoder(
        embed,
        [
            Transformer(size, head, hs, ps, act=act, pdrop=pdrop)
            for i ∈ 1:layer
        ]
    )
end
