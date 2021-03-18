using Transformers
using Transformers.Basic
using Flux
using Flux:@functor, onecold

include("decoder.jl")
include("classifier.jl")
include("translator.jl")

# TransformerAbs model from the "Text Summarization with Pretrained Encoders" paper by Liu et al. (2019) as described on pages 6-7.
function TransformerAbs(
    embed::AbstractEmbed,
    vocab_size::Int,
    max_length::Int,
)::Translator
    return Translator(
        embed, vocab_size,
        5, max_length,
        768, 8, 96, 2048, 6, pdrop=0.1
    )
end

# BertAbs model from the "Text Summarization with Pretrained Encoders" paper by Liu et al. (2019) as described on page 6.
function BertAbs(
    bert_model::TransformerModel{<:AbstractEmbed,<:Bert,<:Any},
    vocab_size::Int,
    max_length::Int,
)::Translator
    return Translator(
        bert_model.embed,
        bert_model.transformers,
        Decoder(768, 8, 96, 2048, 6, pdrop=0.1),
        Classifier(vocab_size, 768),
        BeamSearch(5, max_length)
    )
end
