using Flux
using Transformers.Basic

struct Generator
    chain::Positionwise
end

Flux.@functor Generator
Flux.trainable(generator::Generator) = (generator.chain,)

function (generator::Generator)(embedding)
    return generator.chain(embedding)
end

function Generator(size::Integer, vocab_size::Integer)
    return Generator(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end