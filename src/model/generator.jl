using Transformers.Basic
using Flux
using Flux:@functor

struct Generator
    generator::Positionwise
end

@functor Generator

function (generator::Generator)(embedding)
    return generator.generator(embedding)
end

function Generator(vocab_size::Int, size::Int)
    return Generator(
        Positionwise(
            Dense(size, vocab_size),
            logsoftmax
        )
    )
end
