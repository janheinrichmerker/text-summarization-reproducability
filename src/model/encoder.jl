using Transformers.Basic
using Flux:@functor

struct Encoder
    layers::Array{<:Transformer} 
end

@functor Encoder

function (encoder::Encoder)(embedding)
    for transformer ∈ encoder.layers
        embedding = transformer(embedding)
    end
    return embedding
end

function Encoder(
    size::Integer, 
    head::Integer, 
    hs::Integer, 
    ps::Integer, 
    layer::Integer; 
    act=relu, 
    pdrop=0.1
)::Encoder
    return Encoder([
        Transformer(size, head, hs, ps, act=act, pdrop=pdrop)
        for i ∈ 1:layer
    ])
end
