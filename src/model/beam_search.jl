using Transformers.Basic
using Flux:@functor

include("../search/beam_search.jl")

struct BeamSearch
    width::Int
    steps::Int
end

@functor BeamSearch

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
        initial_sequence=[start_token]
    )[1]
end

function BeamSearch(
    width::Int,
    steps::Int
)
    return BeamSearch(width, steps)
end
