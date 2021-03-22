@info "Load packages."
using DataDeps
using CUDA
using GPUArrays:allowscalar
using Flux
using Flux:update!,reset!,onehot
using Flux.Losses:logitcrossentropy
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using BSON: @save


if !CUDA.functional(true)
    @warn "You're training the model without GPU support."
end
enable_gpu(true)
allowscalar(false)


@info "Load preprocessed data (CNN / Daily Mail)."
include("data/datasets.jl")
include("data/loader.jl")
include("data/model.jl")
function cnndm_loader(corpus_type::String)::Channel{SummaryPair}
    data_dir = joinpath(
        datadep"CNN-Daily-Mail-Preprocessed-BERT",
        "bert_data_cnndm_final"
    )
    return data_loader(data_dir, corpus_type)
end

cnndm_train = cnndm_loader("train")
# cnndm_test = cnndm_loader("test")
# cnndm_valid = cnndm_loader("valid")


@info "Load pretrained BERT model."
bert_model, wordpiece, tokenizer = pretrain"bert-uncased_L-12_H-768_A-12"
vocabulary = Vocabulary(wordpiece) |> todevice
# vocabulary = Vocabulary(wordpiece.vocab[1:100], wordpiece.vocab[100]) |> todevice


@info "Create summarization model from BERT model."
include("model/abstractive.jl")
# model = TransformerAbsTiny(length(vocabulary)) |> todevice
model = BertAbs(bert_model, length(vocabulary)) |> todevice
@show model


function preprocess(text::String)::AbstractVector{String}
    tokens = text |> tokenizer |> wordpiece
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
optimizer_encoder = WarmupADAM(2ℯ^(-3), 20_000, (0.9, 0.99))
optimizer_decoder = WarmupADAM(0.1, 10_000, (0.9, 0.99))


parameters_encoder = params(model.transformers.encoder)
parameters_decoder = params(
    model.transformers.embed, 
    model.transformers.decoder, 
    model.transformers.generator
)
include("model/utils.jl")
@info "Found $(params_count(parameters_encoder)) trainable parameters for encoder and $(params_count(parameters_decoder)) parameters for decoder, embeddings, and generator."
out_path = mkpath(normpath(joinpath(@__FILE__, "..", "out"))) # Create path for storing model snapshots.


reset!(model)
max_steps = 200_000
snapshot_steps = 10 # 2500
for (step, summary) ∈ zip(1:max_steps, cnndm_train)
    @info "Training step $step/$max_steps."

    inputs = summary.source |> preprocess |> todevice
    outputs = summary.target |> preprocess |> todevice
    ground_truth = onehot(vocabulary, outputs) |> todevice


    @info "Train encoder."
    local loss_encoder
    @timed gradients_encoder = gradient(parameters_encoder) do
        loss_encoder = loss(inputs, outputs, ground_truth)
        return loss_encoder
    end
    @show loss_encoder
    @timed update!(optimizer_encoder, parameters_encoder, gradients_encoder)

    @info "Train decoder, embeddings, and generator."
    local loss_decoder
    @timed gradients_decoder = gradient(parameters_decoder) do
        loss_decoder = loss(inputs, outputs, ground_truth)
        return loss_decoder
    end
    @show loss_decoder
    @timed update!(optimizer_decoder, parameters_decoder, gradients_decoder)


    if step % snapshot_steps == 0
        @info "Save model snapshot."
        @save joinpath(out_path, "bert-abs-step-$step-time-$(now())-model.bson") model
        @save joinpath(out_path, "bert-abs-step-$step-time-$(now())-optimizer-encoder.bson") optimizer_encoder
        @save joinpath(out_path, "bert-abs-step-$step-time-$(now())-optimizer-decoder.bson") optimizer_decoder
        # Evaluate on validation set.
    end
end
