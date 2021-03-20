using Flux
using ParameterSchedulers
using ParameterSchedulers:AbstractSchedule

struct Warmup <: AbstractSchedule
    learning_rate::AbstractFloat
    warmup::AbstractFloat
end

function (schedule::Warmup)(t)
    schedule.learning_rate * min(t^(-0.5), t * schedule.warmup^(-1.5))
end

Base.iterate(schedule::Warmup, t=1) = schedule(t), t + 1

Warmup(;η=0.001, w=10_000) = Warmup(η, w)

struct WarmupADAM
    adam::ADAM
    warmup::ScheduleIterator{Warmup,S}
end

WarmupADAM(η=0.001, w=10_000, β=(0.9, 0.999)) = WarmupADAM(
    ADAM(0.0, β), 
    ScheduleIterator(Warmup(η, w))
)

WarmupADAM(;η=0.001, w=10_000, β=(0.9, 0.999)) = WarmupADAM(η, w, β)

function Flux.Optimise.apply!(opt::WarmupADAM, x, Δ)
    opt.adam.eta = next!(opt.warmup)
    return Flux.Optimise.apply!(opt.adam, x, Δ)
end
