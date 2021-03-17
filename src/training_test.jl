@info "Loading Flux."
using Flux
using Flux: onecold
@info "Loading CUDA."
using CUDA
@info "Loading Transformers."
using Transformers
using Transformers.Basic
using Transformers.Pretrain


# Enable GPU for Transformers.
enable_gpu(true)
gpu = todevice


@info "Loading pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece)
@info "Pretrained BERT model loaded successfully."

sample = "Peter Piper picked a peck of pickled peppers" |> tokenizer |> wordpiece |> tokens -> ["[CLS]"; tokens; "[SEP]"]
sample_indices = sample |> vocabulary


# # Combine embedding layers.
# function embed(token_indices)
#     segment_indices = fill(1, length(token_indices))
#     (tok = token_indices, segment = segment_indices)
#     bert_model.embed(data)
# end

tokens = (tok = sample_indices, segment = fill(1, length(sample_indices)))
embeddings = bert_model.embed(tokens)

# include("src/model/abstractive.jl")
include("model/abstractive.jl")

model = BertAbs(bert_model, vocabulary) #|> gpu

encoded = model.encoder(embeddings)
decoded = model.decoder(sample_indices, encoded)
next_tokens = onecold(decoded, model.vocabulary.list)

sample
translated = model(sample)
