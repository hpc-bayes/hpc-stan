FROM python:3.11-slim-bookworm

LABEL authors="Brian Parbhu"
LABEL description="Development and test image for hpc-stan"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml setup.py README.md ./
COPY hpc_stan ./hpc_stan
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

# Set INSTALL_CMDSTAN=true at build time to include a local CmdStan toolchain.
# Example: docker build --build-arg INSTALL_CMDSTAN=true -t hpc-stan .
ARG INSTALL_CMDSTAN=false
RUN if [ "$INSTALL_CMDSTAN" = "true" ]; then python -m cmdstanpy.install_cmdstan --cores 2; fi

CMD ["python", "-m", "pytest", "-q"]
