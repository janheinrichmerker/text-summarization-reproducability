@info "Loading Flux."
using Flux
@info "Loading CUDA."
using CUDA
@info "Loading Transformers."
using Transformers
using Transformers.Basic
using Transformers.Pretrain

# Enable GPU for Transformers.
enable_gpu(true)

@info "Loading pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece)
@info "Pretrained BERT model loaded successfully."

sample_text = "Peter Piper picked a peck of pickled peppers"
@show sample = sample_text |> tokenizer |> wordpiece |> t -> ["[CLS]", t..., "[SEP]"]

include("abstractive.jl")

@show model = BertAbs(
    bert_model,
    length(vocabulary),
    # This parameter is not described in the paper.
    # Though it seems reasonable that for training 
    # it should roughly match the target sequence length.
    length(sample),
    length_normalization=0.8,
)

# The translated output sequence is nonsense, 
# because the decoder has not been trained yet.
@show translated = model(sample, vocabulary)
