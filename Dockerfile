FROM julia:1.5.3

WORKDIR /app

# Install dependencies.
COPY Project.toml Manifest.toml ./
RUN julia --eval " \
    using Pkg; \
    Pkg.activate(\".\"); \
    Pkg.instantiate()"

# Setup PyCall to use a Conda environment.
RUN julia --eval " \
    ENV[\"PYTHON\"] = \"\" \
    using Pkg; \
    Pkg.build(\"PyCall\")"

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
