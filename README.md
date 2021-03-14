[![CI](https://img.shields.io/github/workflow/status/heinrichreimer/text-summarization-reproducability/CI?style=flat-square)](https://github.com/heinrichreimer/text-summarization-reproducability/actions?query=workflow%3A"CI")
[![Code coverage](https://img.shields.io/codecov/c/github/heinrichreimer/text-summarization-reproducability?style=flat-square)](https://codecov.io/github/heinrichreimer/text-summarization-reproducability/)
[![Issues](https://img.shields.io/github/issues/heinrichreimer/text-summarization-reproducability?style=flat-square)](https://github.com/heinrichreimer/text-summarization-reproducability/issues)
[![Commit activity](https://img.shields.io/github/commit-activity/m/heinrichreimer/text-summarization-reproducability?style=flat-square)](https://github.com/heinrichreimer/text-summarization-reproducability/commits)
[![License](https://img.shields.io/github/license/heinrichreimer/text-summarization-reproducability?style=flat-square)](LICENSE)

# 📝 text-summarization-reproducability

Reproducability study on the ["Text Summarization with Pretrained Encoders"](https://doi.org/10.18653/v1/D19-1387) paper by Yang Liu and Mirella Lapata.
In contrast to [the original implementation](https://github.com/nlpyang/PreSumm), we use the [Julia Language](https://julialang.org/) with [Flux.jl](https://fluxml.ai/) and [Transformers.jl](https://github.com/chengchingwen/Transformers.jl) for building the model. 

_The study is conducted as part of the [Data Mining](https://www.informatik.uni-halle.de/arbeitsgruppen/dbs/lehre/2757674_2757760/) lecture at [Martin Luther University Halle-Wittenberg](https://uni-halle.de)._

## Usage

### Local machine

1. Install [Julia](https://julialang.org/downloads/) and open Julia REPL.

    ```shell script
    julia
    ```

1. Activate project and install dependencies.

    ```julia
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    ```

1. Start [Pluto](https://github.com/fonsp/Pluto.jl) notebook.

    ```julia
    using Pluto
    Pluto.run(notebook="./src/notebook.jl")
    ```

### Docker container

1. Install [Docker](https://docs.docker.com/get-docker/).
1. Build a Docker container with this project.

    ```shell script
    docker build -t text-summarization-reproducability .
    ```

1. Start [Pluto](https://github.com/fonsp/Pluto.jl) notebook.

    ```shell script
    docker run -p 1234:1234 -it text-summarization-reproducability
    ```

    Note that Julia runs rather slow inside Docker.

## License

This project is [MIT licensed](LICENSE), so you can use the code for whatever you want as long as you mention this repository.
