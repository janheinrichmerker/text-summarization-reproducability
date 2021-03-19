using Transformers.Basic
using Flux
using Flux:@functor

struct Generator
    chain::Positionwise
end

@functor Generator

function (generator::Generator)(embedding)
    return generator.chain(embedding)[:,end]
end

function Generator(size::Int, vocab_size::Int)
    return Generator(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end
