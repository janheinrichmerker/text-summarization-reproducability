@info "Loading Flux."
using Flux
using Flux:onehot
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

preprocess(text::String) = ["[CLS]"; wordpiece(tokenizer(text)); "[SEP]"]
sample = "Peter Piper picked a peck of pickled peppers" |> preprocess
@show sample
target = "Peter picked pickled peppers" |> preprocess
@show target

include("abstractive.jl")
model = BertAbs(bert_model, length(vocabulary))
@show model

# Predict 4th token in target (given the first 3 tokens)
prediction = model.transformers(vocabulary, sample, target[1:3])
@show size(prediction)
ground_truth = onehot(target[4], vocabulary.list)
@show size(ground_truth)

include("loss.jl")
# How good is the prediction?
@show loss = Flux.Losses.logitcrossentropy(prediction, ground_truth)

# Generate a new sequence from the sample.
# The output sequence is likely to be nonsense, 
# because the decoder has not been trained yet.
@show translated = model(sample, vocabulary)
