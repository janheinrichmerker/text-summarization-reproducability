@info "Loading Flux."
using Flux
using Flux:onehotbatch
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
inputs = "Peter Piper picked a peck of pickled peppers." |> preprocess
@show inputs
outputs = "Peter picked pickled peppers." |> preprocess
@show outputs

include("abstractive.jl")
model = BertAbs(bert_model, length(vocabulary))
@show model

# Predict new word probabilities for target.
prediction = model.transformers(vocabulary, inputs, outputs)
@show size(prediction)
# Use original one-hot word probabilities from target for comparison.
ground_truth = onehotbatch(outputs, vocabulary.list)
@show size(ground_truth)

include("loss.jl")
# How good is the prediction?
@show loss = Flux.Losses.logitcrossentropy(prediction, ground_truth)

# Generate a new sequence from the sample.
# The output sequence is likely to be nonsense, 
# because the decoder has not been trained yet.
@show translated = model(sample, vocabulary)
