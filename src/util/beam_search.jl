# Beam search for generating sequences.
function beam_search(
    width::Int, 
    steps::Int, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict_next::Function,
    vocabulary::AbstractVector;
)::AbstractVector{T} where T
    @assert size <= length(vocabulary)

    struct Path{T}
        sequence::Array{T}
        probability::AbstractFloat
    end

    # Start with one initial path, the empty sequence.
    local paths = [Path([], 1.0)]
    for i ∈ 1:steps
        # Probability for each word in the vocabulary, 
        # to appear next with each currently considered sequence.
        probabilities_matrix::Array{<:AbstractFloat,2} = hcat(
            predict_next(path.sequence) .* path.probability
            for path ∈ paths
        )
        @assert size(probabilities_matrix, 1) == length(vocabulary)
        @assert size(probabilities_matrix, 2) <= width

        # Helper function to get a potential path's probability.
        function probability(word_index::Int, path_index::Int)
            probabilities_matrix[word_index, path_index]
        end

        # Array of pairs of row (vocabulary) and column (sequence) indices. 
        indices = [
            (word_index, path_index)
            for word_index ∈ axes(probabilities_matrix, 1),
                path_index ∈ axes(probabilities_matrix, 2)
        ]

        # Sort indices by descending probability.
        sort!(indices, by=probability, rev=true)

        # Select best indices.
        most_likely_indices = first(indices, width)

        # Build next paths.
        paths = [
            Path(
                push!(paths[path_index].sequence, vocabulary[word_index]),
                probability(word_index, path_index)
            )
            for (word_index, path_index) ∈ most_likely_indices
        ]
    end

    sort!(paths, by=path -> path.probability, rev=true)
    return map(path -> path.sequence, paths)
end

# Like beam search, but only the single locally best path is considered.
greedy_search(
    steps::Int, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict_next::Function,
    vocabulary::AbstractVector,
) = beam_search(1, steps, predict_next, vocabulary)