using BertAbs
using CUDA
using GPUArrays:allowscalar
using Flux
using Transformers
using Transformers.Basic
using Transformers.Pretrain
using BSON: @load, @save

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


snapshots = readdir(out_dir(), join=true)

model_snapshots = filter(
    file -> isfile(file) && occursin("model.bson", file), 
    snapshots
)

weights_snapshots = filter(
    file -> isfile(file) && occursin("weights.bson", file), 
    weights_snapshots
)

new_weights_snapshots = filter(
    file -> !(file ∈ weights_snapshots),
    map(
        file -> replace(file, "model.bson" => "weights.bson"),
        model_snapshots
    )
)

for weights_snapshot ∈ reverse(new_weights_snapshots)
    model_snapshot = replace(weights_snapshot, "weights.bson" => "model.bson")
    @assert isfile(model_snapshot)
    @load model_snapshot model
    @show model
    model = model |> cpu
    @show model
    @show params(model)
end
