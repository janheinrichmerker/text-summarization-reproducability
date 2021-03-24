using Flux
using ParameterSchedulers
using ParameterSchedulers:AbstractSchedule

struct Warmup <: AbstractSchedule
    eta::Float32
    warmup::Int
end

function (schedule::Warmup)(t::Int)
    schedule.eta * min(t^(-0.5), t * schedule.warmup^(-1.5))
end

Base.getindex(schedule::Warmup, t::Int) = schedule(t)
Base.iterate(schedule::Warmup, t::Int=1) = schedule(t), t + 1

Warmup(;η=0.001, w=10_000) = Warmup(η, w)

struct WarmupADAM
    adam::ADAM
    warmup::ScheduleIterator{Warmup,<:Any}
end

WarmupADAM(η, w, β) = WarmupADAM(
    ADAM(0.0, β), 
    ScheduleIterator(Warmup(η, w))
)
WarmupADAM(;η=0.001, w=10_000, β=(0.9, 0.999)) = WarmupADAM(η, w, β)


function Flux.Optimise.apply!(opt::WarmupADAM, x, Δ)
    opt.adam.eta = next!(opt.warmup)
    return Flux.Optimise.apply!(opt.adam, x, Δ)
end
