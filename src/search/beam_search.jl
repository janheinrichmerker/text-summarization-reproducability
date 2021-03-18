struct Path{T}
    sequence::Array{T}
    log_probability::AbstractFloat
end

is_probability(x) = x >= 0 && x <= 1
is_log_probability(x) = is_probability(exp(x))

function expand(
    path::Path{T},
    predict::Function,
    vocabulary::AbstractVector{T}
)::AbstractVector{Path{T}} where T
    # Compute probabilities for next token.
    log_word_probabilities = predict(path.sequence)
    @assert all(is_log_probability, log_word_probabilities)
    @assert length(log_word_probabilities) == length(vocabulary)

    # Combine with own log probability.
    # Addition because log(p1 * p2) = log(p1) + log(p2).
    log_probabilities = log_word_probabilities .+ path.log_probability
    @assert all(is_log_probability, log_probabilities)


    # Return expanded paths.
    return [
        Path(
            [path.sequence..., vocabulary[i]],
            log_probabilities[i],
        )
        for i ∈ 1:length(vocabulary)
    ]
end

function expand(
    path::Path{T},
    predict::Function,
    vocabulary::AbstractVector{T},
    expandable::Function
)::AbstractVector{Path{T}} where T
    if expandable(path.sequence)
        # Expand next token.
        expand(path, predict, vocabulary)
    else
        # Otherwise, return as is.
        [path]
    end
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function;
    initial_sequence::AbstractVector{T}=[],
    length_normalization::Number=0.0,
)::AbstractVector{T} where T
    @assert width <= length(vocabulary)

    # Score used for selecting best paths in each iteration.
    function score(path::Path{T})
        # Compute length penalty for normalization.
        α = length_normalization
        length_penalty = ((5 + length(path.sequence))^α) / ((5 + 1)^α)

        return path.log_probability / length_penalty
    end

    # Start with one initial path, the empty sequence.
    paths::AbstractVector{Path{T}} = [Path(initial_sequence, 1.0)]

    # Expand iteratively until no expandable path is left.
    i = 1
    while length(paths) > 0 && any(path -> expandable(path.sequence), paths)
        @info "Predicting next token in iteration $i."
        
        # Calculate paths (hypotheses and probabilities).
        next_paths::AbstractVector{Path{T}} = vcat(
            [
                expand(path, predict, vocabulary, expandable)
                for path ∈ paths 
            ]...
        )
        @assert all(x -> is_log_probability(x.log_probability), next_paths)

        # Sort paths by descending score.
        sort!(next_paths, by=score, rev=true)

        # Select best paths.
        paths = next_paths[1:min(length(next_paths), width)]

        @info "Scores in iteration $i." map(score, paths)
        i += 1
    end

    return paths[1].sequence
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[],
    length_normalization::Number=0.0,
)::AbstractVector{T} where T
    beam_search(
        width,
        vocabulary,
        predict,
        sequence -> length(sequence) <= length(initial_sequence) + steps,
        initial_sequence=initial_sequence,
        length_normalization=length_normalization,
    )
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[],
    length_normalization::Number=0.0,
)::AbstractVector{T} where T
    beam_search(
        width,
        vocabulary,
        predict,
        sequence -> expandable(sequence) && (length(sequence) <= length(initial_sequence) + steps),
        initial_sequence=initial_sequence,
        length_normalization=length_normalization,
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function;
    initial_sequence::AbstractVector{T}=[],
)::AbstractVector{T} where T
    beam_search(
        1, 
        vocabulary, 
        predict, 
        expandable, 
        initial_sequence=initial_sequence,
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{T} where T
    beam_search(
        1, 
        vocabulary, 
        predict, 
        steps, 
        initial_sequence=initial_sequence,
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of log probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{T} where T
    beam_search(
        1, 
        vocabulary, 
        predict,
        expandable, 
        steps, 
        initial_sequence=initial_sequence,
    )
end
