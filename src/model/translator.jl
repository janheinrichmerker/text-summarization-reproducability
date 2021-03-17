using Transformers
using Transformers.Basic
using Flux
using Flux:@functor, onecold

import("encoder.jl")
import("decoder.jl")
import("classifier.jl")

struct Translator
    encoder::Union{Encoder,Bert}
    decoder::Decoder
    classifier::Classifier
    vocabulary::Vocabulary
end

@functor Translator

function (translator::Translator)(tokens, start_token, end_token)
    indices = tokens |> vocabulary
    sequence = [start_token]
    encoded = translator.encoder(indices)
    while last(sequence) != end_token
        target = sequence |> vocabulary
        decoded = translator.decoder(target, encoded) |> translator.classifier
        next_tokens = onecold(decoded, translator.vocabulary.list)
        # TODO Trigram blocking here?
        @show next_token = next_tokens[end]
        push!(sequence, next_token)
    end
    return sequence[2:end - 1]
end

function Translator(embed::AbstractEmbed, vocabulary::Vocabulary, size::Int, head::Int, hs::Int, ps::Int, layer::Int; act=relu,pdrop=0.1)
    return Translator(
        Encoder(embed, size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Decoder(embed, size, head, hs, ps, layer; act=act,pdrop=pdrop),
        Classifier(length(vocabulary), size),
        vocabulary
    )
end
