using Transformers.Basic
using Flux
using Flux:@functor

struct Generator
    chain::Positionwise
end

@functor Generator

function (generator::Generator)(embedding)::AbstractMatrix{Number}
    return generator.chain(embedding)
end

function Generator(size::Int, vocab_size::Int)
    return Generator(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end
