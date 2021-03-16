FROM julia:1.5.3

WORKDIR /app

# Install dependencies.
COPY Project.toml Manifest.toml ./
RUN julia --eval " \
    using Pkg; \
    Pkg.activate(\".\"); \
    Pkg.instantiate()"

# Setup PyCall to use a Conda environment.
COPY src/setup_python.jl ./src/
RUN julia --project=./ ./src/setup_python.jl

# Pre-compile Pluto.
RUN julia --eval " \
    using Pkg; \
    Pkg.activate(\".\"); \
    using Pluto"

COPY . ./

EXPOSE 1234

# Start notebook.
CMD julia --eval=" \
    using Pkg; \
    Pkg.activate(\".\"); \
    using Pluto; \
    Pluto.run(host=\"0.0.0.0\", notebook=\"src/notebook.jl\")" |\
    sed "s/0.0.0.0/localhost/"
