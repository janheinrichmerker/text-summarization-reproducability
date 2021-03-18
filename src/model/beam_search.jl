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
    search = beam_search(
        search.width,
        vocabulary.list,
        predict,
        sequence -> last(sequence) != end_token,
        search.steps,
        initial_sequence=[start_token]
    )
    @show search
    return search[1]
end
