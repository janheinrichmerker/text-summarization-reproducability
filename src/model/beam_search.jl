using Transformers.Basic

include("../search/beam_search.jl")

struct BeamSearch
    width::Integer
    steps::Integer
    length_normalization::AbstractFloat
end

function (search::BeamSearch)(
    vocabulary::Vocabulary{T},
    predict::Function;
    start_token::T="[CLS]",
    end_token::T="[SEP]"
)::AbstractVector{T} where T
    return beam_search(
        search.width,
        vocabulary.list,
        predict,
        sequence -> last(sequence) != end_token,
        search.steps,
        initial_sequence=[start_token],
        length_normalization=search.length_normalization
    )
end

function BeamSearch(
    width::Integer,
    steps::Integer;
    length_normalization::AbstractFloat=0.0,
)
    return BeamSearch(width, steps, length_normalization)
end