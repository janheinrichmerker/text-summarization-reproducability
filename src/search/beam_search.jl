struct Path{T}
    sequence::Array{T}
    probability::AbstractFloat
end

function expand(
    path::Path{T},
    predict_next::Function,
    vocabulary::AbstractVector{T}
)::AbstractVector{Path{T}} where T
    # Compute probabilities for next token.
    word_probabilities = predict_next(path.sequence)

    # Combine with own probability.
    probabilities = word_probabilities .* path.probability

    @assert length(probabilities) == length(vocabulary)

    # Return expanded paths.
    return [
        Path(
            [path.sequence..., vocabulary[i]],
            probabilities[i],
        )
        for i ∈ 1:length(vocabulary)
    ]
end

function expand(
    path::Path{T},
    predict_next::Function,
    vocabulary::AbstractVector{T},
    expandable::Function,
)::AbstractVector{Path{T}} where T
    if expandable(path.sequence)
        # Expand next token.
        expand(path, predict_next, vocabulary)
    else
        # Otherwise, return as is.
        [path]
    end
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{AbstractVector{T}} where T
    @assert width <= length(vocabulary)

    # Start with one initial path, the empty sequence.
    local paths::AbstractVector{Path{T}} = [Path(initial_sequence, 1.0)]

    # Expand iteratively until no expandable path is left.
    i = 0
    while length(paths) > 0 && any(path -> expandable(path.sequence), paths)
        @show i += 1
        

        # Calculate paths (hypotheses and probabilities).
        next_paths::AbstractVector{Path{T}} = vcat(
            [
                expand(path, predict, vocabulary, expandable)
                for path ∈ paths 
            ]...
        )

        # Sort paths by descending probability.
        sort!(next_paths, by=path -> path.probability, rev=true)

        # Select best paths.
        paths = next_paths[1:min(length(next_paths), width)]

        @show map(path -> path.sequence, paths)
    end
    
    return map(path -> path.sequence, paths)
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{AbstractVector{T}} where T
    beam_search(
        width,
        vocabulary,
        predict,
        sequence -> length(sequence) <= length(initial_sequence) + steps,
        initial_sequence=initial_sequence
    )
end

# Beam search for generating sequences.
function beam_search(
    width::Int, 
    vocabulary::AbstractVector{T}, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function,
    # Max steps to expand.
    steps::Int;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{AbstractVector{T}} where T
    beam_search(
        width,
        vocabulary,
        predict,
        sequence -> expandable(sequence) && (length(sequence) <= length(initial_sequence) + steps),
        initial_sequence=initial_sequence
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict::Function,
    # Predicate to check whether a sequence is expandable.
    # One might for example check for an end token.
    expandable::Function;
    initial_sequence::AbstractVector{T}=[]
)::AbstractVector{T} where T
    beam_search(
        1, 
        vocabulary, 
        predict, 
        expandable, 
        initial_sequence=initial_sequence
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of probabilities for each word 
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
        initial_sequence=initial_sequence
    )
end

# Like beam search, but only the single locally best path is considered.
function greedy_search(
    vocabulary::AbstractVector{T},
    # Function returning a column vector of probabilities for each word 
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
        initial_sequence=initial_sequence
    )
end
