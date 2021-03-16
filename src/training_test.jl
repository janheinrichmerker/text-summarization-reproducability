@info "Loading Flux."
using Flux
using Flux: onecold
@info "Loading CUDA."
using CUDA
@info "Loading Transformers."
using Transformers
@info "Loading Transformers.Basic."
using Transformers.Basic

# Enable GPU for Transformers.
enable_gpu(true)
gpu = todevice

# Define vocabulary.
labels = collect(1:10)
start_symbol = 10 + 1
end_symbol = 10 + 2
unknown_symbol = 0
labels = [unknown_symbol, start_symbol, end_symbol, labels...]
vocabulary = Vocabulary(labels, unknown_symbol)

# Function for adding start & end symbol.
preprocess(data::Array{Int})::Array{Int} = [start_symbol, data..., end_symbol]

# Function for generating training data.
function sample_data()::Tuple{Array{Int}, Array{Int}}
    data = rand(1:10, 15)
    return data, data
end

# Generate a training sample.
@show preprocessed_data = preprocess.(sample_data())
# Encode sample using vocabulary.
@show sample = vocabulary(preprocessed_data[1])


# Define word embedding layer to turn word indexes to word vectors.
embed_word = Embed(512, length(vocabulary)) #|> gpu
# Define position embedding layer.
embed_position = PositionEmbedding(512) #|> gpu
# Combine embedding layers.
function embed(data::Array{Int})
    scale = inv(sqrt(512))
    word_embedding = embed_word(data, scale)
    position_embedding = embed_position(word_embedding)
    return word_embedding + position_embedding
end

@show embed(sample)

# Define two transformer encoder layers.
encoder_layers = [
    Transformer(512, 8, 64, 2048) #|> gpu
    for i ∈ 1:2
]

# Define two transformer encoder layers.
decoder_layers = [
    TransformerDecoder(512, 8, 64, 2048) #|> gpu
    for i ∈ 1:2
]

# Define final classification layer.
linear = Positionwise(
    Dense(512, length(vocabulary)),
    logsoftmax
) #|> gpu

# Define encoder stack.
function encoder(data::Array{Int})
    embedding = embed(data)
    for transformer ∈ encoder_layers
        embedding = transformer(embedding)
    end
    return embedding
end

# Define decoder stack.
function decoder(data::Array{Int}, encoded)
    embedding = embed(data)
    for transformer ∈ decoder_layers
        embedding = transformer(embedding, encoded)
    end
    probabilities = linear(embedding)
    probabilities
end

encoded = encoder(sample)
@show length(enc)
probs = decoder(sample, encoded)
@show length(probs)


function translate(input::Array{Int})
    indices = input |> preprocess |> vocabulary #|> gpu
    sequence::Array{Int} = [start_symbol]
    encoded = encoder(indices)
    max_iterations = 2 * length(indices)
    for i = 1:max_iterations
        @info "Iteration $i of $max_iterations."
        target = sequence |> vocabulary #|> gpu
        decoded = decoder(target, encoded)
        next_tokens = onecold(decoded, labels)
        next_token = next_tokens[end]
        push!(sequence, next_token)
        if next_token == end_symbol
            break
        end
    end
    # Strip start and end token.
    sequence[2:end-1]
end

@show sample
@show translated = translate(sample)
