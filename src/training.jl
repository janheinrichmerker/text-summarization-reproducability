@info "Load packages."
using CUDA
using GPUArrays:allowscalar
using Flux
using Flux:update!,reset!,onehot
using Flux.Losses:logitcrossentropy
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using BSON: @save


if CUDA.functional(true)
    # CUDA.device!(1)
    allowscalar(false)
    Flux.use_cuda[] = true
    enable_gpu(true)
else
    @warn "You're training the model without GPU support."
    Flux.use_cuda[] = false
    enable_gpu(false)
end



@info "Load preprocessed data (CNN / Daily Mail)."
include("data/model.jl")
include("data/datasets.jl")

cnndm_train = cnndm_loader(train_type)
# cnndm_test = cnndm_loader("test")
# cnndm_valid = cnndm_loader("valid")


@info "Load pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
# vocabulary = Vocabulary(wordpiece) |> gpu
vocabulary = Vocabulary(wordpiece.vocab[1:100], wordpiece.vocab[100]) |> gpu


@info "Create summarization model from BERT model."
include("model/abstractive.jl")
# model = BertAbs(bert_model, length(vocabulary)) |> gpu
model = TransformerAbsTiny(length(vocabulary)) |> gpu
@show model


function preprocess(text::String)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
    max_length = min(4096, length(tokens)) # Truncate to 4096 tokens
    tokens = tokens[1:max_length]
    return ["[CLS]"; tokens; "[SEP]"]
end

include("training/loss.jl")
label_smoothing_α = 0.0 # Label smoothing doesn't work yet.
function loss(
    inputs::AbstractVector{String},
    outputs::AbstractVector{String},
    ground_truth::AbstractMatrix{<:Number}
)::AbstractFloat
    prediction = model.transformers(vocabulary, inputs, outputs)
    loss = logtranslationloss(prediction, ground_truth, α=label_smoothing_α)
    return loss
end


include("training/optimizers.jl")
optimizer_encoder = WarmupADAM(2ℯ^(-3), 20_000, (0.9, 0.99)) |> gpu
optimizer_decoder = WarmupADAM(0.1, 10_000, (0.9, 0.99)) |> gpu


parameters_encoder = params(model.transformers.encoder)
parameters_decoder = params(
    model.transformers.embed, 
    model.transformers.decoder, 
    model.transformers.generator
)
include("model/utils.jl")
@info "Found $(params_count(parameters_encoder)) trainable parameters for encoder and $(params_count(parameters_decoder)) parameters for decoder, embeddings, and generator."


reset!(model)
max_steps = 200_000
# snapshot_steps = 2500
snapshot_steps = 1
losses_encoder = []
losses_decoder = []
start_time = now()
include("data/utils.jl")
for (step, summary) ∈ zip(1:max_steps, cnndm_train)
    @info "Training step $step/$max_steps."

    inputs = summary.source |> preprocess |> gpu
    outputs = summary.target |> preprocess |> gpu
    ground_truth = onehot(vocabulary, outputs) |> gpu


    @info "Train encoder."
    local loss_encoder
    @timed gradients_encoder = gradient(parameters_encoder) do
        loss_encoder = loss(inputs, outputs, ground_truth)
        return loss_encoder
    end
    push!(losses_encoder, loss_encoder)
    @info "Updating encoder parameters." loss_encoder
    @timed update!(optimizer_encoder, parameters_encoder, gradients_encoder)

    @info "Train decoder, embeddings, and generator."
    local loss_decoder
    @timed gradients_decoder = gradient(parameters_decoder) do
        loss_decoder = loss(inputs, outputs, ground_truth)
        return loss_decoder
    end
    push!(losses_decoder, loss_decoder)
    @info "Updating decoder, embeddings, and generator parameters." loss_decoder
    @timed update!(optimizer_decoder, parameters_decoder, gradients_decoder)


    if step % snapshot_steps == 0
        @info "Save model snapshot."
        @save snapshot_file(start_time, step, "model.bson") model |> cpu
        @save snapshot_file(start_time, step, "optimizer-encoder.bson") optimizer_encoder |> cpu
        @save snapshot_file(start_time, step, "optimizer-decoder.bson") optimizer_decoder |> cpu
        @save snapshot_file(start_time, step, "losses-encoder.bson") losses_encoder
        @save snapshot_file(start_time, step, "losses-decoder.bson") losses_decoder
        # Evaluate on validation set.
    end
end
