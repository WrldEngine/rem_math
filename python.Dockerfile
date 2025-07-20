# syntax=docker/dockerfile:1

FROM python:3.12

WORKDIR /sandbox

# OpenCLのライブラリインストールをします
RUN apt update
RUN apt install ocl-icd-opencl-dev -y

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install rem-math

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
