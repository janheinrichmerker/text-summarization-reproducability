[![GitHub Actions](https://img.shields.io/github/workflow/status/heinrichreimer/text-summarization-reproducability/CI?style=flat-square)](https://github.com/heinrichreimer/text-summarization-reproducability/actions?query=workflow%3A"CI")

# 📝 text-summarization-reproducability

Reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
In contrast to [the original implementation](https://github.com/nlpyang/PreSumm), we use the [Julia Language](https://julialang.org/) with [Flux.jl](https://fluxml.ai/) and [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) for building the model. 

_The study is conducted as part of the [Data Mining](https://www.informatik.uni-halle.de/arbeitsgruppen/dbs/lehre/2757674_2757760/) lecture at [Martin Luther University Halle-Wittenberg](https://uni-halle.de)._

## Usage

1. Start [Pluto]:

    ```shell script
    julia --project=. --eval="using Pluto; Pluto.run()"
    ```

## License

This project is [MIT licensed](LICENSE), so you can use the code for whatever you want as long as you mention this repository.
