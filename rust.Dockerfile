# syntax=docker/dockerfile:1

ARG RUST_VERSION=1.81.0
ARG APP_NAME=rem_math_sandbox
ARG APP_WORKDIR=/rem_math_sandbox

FROM rust:${RUST_VERSION}-slim-bullseye AS build
ARG APP_NAME
ARG APP_WORKDIR

WORKDIR ${APP_WORKDIR}

# ssl lib のインストール
RUN apt-get update -y && \
  apt-get install -y pkg-config make g++ libssl-dev && \
  rustup target add x86_64-unknown-linux-gnu

# OpenCLのライブラリインストールをします
RUN apt update
RUN apt install ocl-icd-opencl-dev -y

RUN --mount=type=bind,source=src,target=src \
    --mount=type=bind,source=Cargo.toml,target=Cargo.toml \
    --mount=type=bind,source=Cargo.lock,target=Cargo.lock \
    --mount=type=cache,target=/$APP_WORKDIR/target/ \
    --mount=type=cache,target=/usr/local/cargo/registry/ \
    RUSTFLAGS="-Z threads=8" cargo +nightly build --release --locked

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser
USER appuser

ENTRYPOINT ["/bin/bash", "-c"]
