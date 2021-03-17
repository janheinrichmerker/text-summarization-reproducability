# Beam search for generating sequences.
function beam_search(
    width::Int, 
    steps::Int, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence.
    predict_next::Function,
    vocabulary::AbstractVector,
)::AbstractVector{T} where T
    @assert size <= length(vocabulary)

    # Start with only one option, the empty sequence.
    sequences::Array{NamedTuple{(:sequence, :probability),Tuple{Array{T},<:AbstractFloat}}} = [
        (sequence = [], probability = 1.0)
    ]
    for i ∈ 1:steps
        # Probability for each word in the vocabulary, 
        # to appear next with each currently considered sequence.
        probabilities_matrix::Array{<:AbstractFloat,2} = hcat(
            predict_next(sequence.sequence) .* sequence.probability
            for sequence ∈ sequences
        )
        @assert size(probabilities_matrix, 1) == length(vocabulary)
        @assert size(probabilities_matrix, 2) <= width

        # Array of pairs of row (vocabulary) and column (sequence) indices. 
        indices = [
            (v, s)
            for v ∈ axes(probabilities_matrix, 1),
                s ∈ axes(probabilities_matrix, 2)
        ]

        # Sort indices by descending probability.
        sort!(
            indices,
            by=(v, s) -> probabilities_matrix[v, s],
            rev=true
        )

        # Select best indices.
        most_likely_indices = first(indices, width)

        # Build next sequences.
        sequences = [
            (
                sequence = push!(sequences[s].sequence, vocabulary[v]),
                probability = probabilities_matrix[v, s]
            )
            for (v, s) ∈ most_likely_indices
        ]
    end
    return sequence[end]
end

greedy_search(
    steps::Int, 
    # Function returning a column vector of probabilities for each word 
    # in the vocabulary, given the current sequence and input.
    predict_next::Function,
    vocabulary::AbstractVector,
) = beam_search(1, steps, predict_next, vocabulary)