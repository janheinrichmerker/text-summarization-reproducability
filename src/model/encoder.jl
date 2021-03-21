using Flux
using Transformers.Basic

struct Encoder
    layers::Array{<:Transformer} 
end

Flux.@functor Encoder
Flux.trainable(encoder::Encoder) = (encoder.layers,)

function (encoder::Encoder)(embedding)
    return foldl(
        (embedding, transformer) -> transformer(embedding),
        encoder.layers,
        init=embedding
    )
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
