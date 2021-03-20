using Transformers
using Transformers.Basic
using Flux
using Flux:@functor, onecold

include("decoder.jl")
include("generator.jl")
include("transformers.jl")
include("translator.jl")

# TransformerAbs transformer model for abstractive summarization
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on pages 6-7.
function TransformerAbs(vocab_size::Integer)::Translator
    return Translator(
        TransformersModel(
            Embed(768, vocab_size),
            768, 8, 96, 2048, 6, 
            vocab_size, 
            pdrop=0.1,
        ),
        AbsSearch()
    )
end

# BertAbs transformer model for abstractive summarization 
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on page 6.
function BertAbs(
    bert_model::TransformerModel{<:AbstractEmbed,<:Bert,<:Any},
    vocab_size::Integer
)::Translator
    return Translator(
        TransformersModel(
            # It's not clear how the random embedding is "added to"
            # BERT embeddings.
            Embed(768, vocab_size), # bert_model.embed,
            bert_model.transformers,
            Decoder(768, 8, 96, 2048, 6, pdrop=0.1),
            Generator(768, vocab_size)
        ),
        AbsSearch()
    )
    return 
end

# Beam search used in the abstractive summarization models 
# from the "Text Summarization with Pretrained Encoders" paper 
# by Liu et al. (2019) as described on page 6.
function AbsSearch()
    return BeamSearch(
        5,
        # This parameter is not described in the paper.
        32,
        # The length normalization parameter α should be 
        # tuned on the development set from 0.6 to 1.0.
        length_normalization=0.8
    )
end