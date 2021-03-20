#!/bin/bash

set -e

VERSION="1.5.3"

if [ -z `which julia` ]; then
  echo "Installing Julia ..."
  URL="https://julialang-s3.julialang.org/bin/linux/x64/$(cut -d '.' -f -2 <<< "$VERSION")/julia-$VERSION-linux-x86_64.tar.gz"
  mkdir -p ./tmp
  wget -nv $URL -O ./tmp/julia.tar.gz
  mkdir -p "./julia-$VERSION"
  tar -x -f ./tmp/julia.tar.gz -C "./julia-$VERSION" --strip-components 1
  rm ./tmp/julia.tar.gz
  ln -s "./julia-$VERSION/bin/julia" ./julia
  echo "Finished installing Julia."
fi