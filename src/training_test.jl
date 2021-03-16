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
@info "Pretrained BERT model loaded successfully."


vocabulary = Vocabulary(wordpiece)

# Function for adding start & end symbol.
function preprocess(text)
    tokens = text |> tokenizer |> wordpiece
    ["[CLS]"; tokens; "[SEP]"]
end

# Function for generating training data.
function sample_data()
    text = "Peter Piper picked a peck of pickled peppers"
    return text, text
end

# Generate a training sample.
@show preprocessed_data = preprocess.(sample_data())
# Encode sample using vocabulary.

function to_bert_input(preprocessed)
    token_indices = vocabulary(preprocessed)
    segment_indices = fill(1, length(preprocessed))
    (tok = token_indices, segment = segment_indices)
end

@show sample = to_bert_input(preprocessed_data[1])

# Combine embedding layers.
function embed(data)
    bert_model.embed(data)
end

@show embed(sample)



# Define 6 [paper] transformer encoder layers.
decoder_layers = [
    # Each decoder has 768 hidden units (like Liu et al.), 
    # with a hidden size for feed-forward layers of 2048 (like Liu et al.). 
    # We set the dropout probability to 0.1 (like Liu et al.).
    # We use 8 heads with a head size of 768/8 = 96 for the 
    # multi-head attention (like Vaswani et al., as mentioned in Liu et. al).
    TransformerDecoder(768, 8, 96, 2048, pdrop=0.1) #|> gpu
    for i ∈ 1:6
]

# Define final classification layer.
linear = Positionwise(
    Dense(768, length(vocabulary)),
    logsoftmax
) #|> gpu

# Define encoder stack.
function encoder(data)
    data |> embed |> bert_model.transformers
end

# Define decoder stack.
function decoder(data, encoded)
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


function translate(input::String)
    indices = input |> preprocess |> to_bert_input #|> gpu
    sequence = ["[CLS]"]
    encoded = encoder(indices)
    max_iterations = 2 * length(indices.tok)
    for i = 1:max_iterations
        @info "Iteration $i of $max_iterations."
        target = sequence |> to_bert_input #|> gpu
        decoded = decoder(target, encoded)
        next_tokens = onecold(decoded, vocabulary.list)
        @show next_token = next_tokens[end]
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
