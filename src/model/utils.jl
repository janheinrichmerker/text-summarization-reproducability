using Flux:Params

params_count(params::Params) = sum(map(length, collect(params.params)))
