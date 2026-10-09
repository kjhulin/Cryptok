# Production image for `cryptok serve` (see docs/DEPLOY-AWS.md).
#   docker build -t cryptok .
#   docker run --rm -p 8077:8077 -e CRYPTOK_AUTH='user:a-long-random-password' \
#       --read-only --cap-drop ALL --memory 512m cryptok --allowed-host cryptok.example.com
#
# Memory. Running: the default (--lean) uses about 180 MB idle and peaked at 300 MB in a stress
# battery, so 512 MB is comfortable. Building: training the language model needs about 420 MB
# (845 MB before it was made leaner) on top of the Rust compiler, so use a build host with
# 2 GB+ (or add swap), or train a smaller model:
#   docker build --build-arg MODEL_ORDER=5 -t cryptok .     # ~170 MB to train, 23 MB model, ~2.6 points less accurate
# or skip training here: build the model elsewhere (`cryptok train`, or the release artifact)
# and use  --build-arg MODEL_FILE=cryptok.cklm  with that file in the build context.
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates crates
COPY corpus corpus
COPY data data
ARG MODEL_ORDER=6
ARG MODEL_FILE=
RUN cargo build --release --locked
# Use a pre-trained model if one is named, otherwise train one now.
COPY . /ctx
RUN if [ -n "$MODEL_FILE" ]; then cp "/ctx/$MODEL_FILE" /src/cryptok.cklm; \
    else ./target/release/cryptok train --corpus corpus --order "$MODEL_ORDER" --out /src/cryptok.cklm; fi

FROM debian:bookworm-slim
# Tesseract does the OCR server-side, so browsers never need to reach a CDN (and the page's
# content-security policy can stay strict).
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin cryptok
WORKDIR /app
COPY --from=build /src/target/release/cryptok /src/cryptok.cklm ./
# Public-domain texts only. Do NOT copy bench/private (copyrighted keys) into this image:
# the known-text search would reveal their contents to users.
COPY corpus corpus
COPY data data
USER 10001:10001
# Memory: glibc's per-thread arenas and lazy trimming otherwise keep freed search buffers
# resident long after a job ends. Two arenas, and large blocks (search buffers) go straight to
# mmap so they are returned to the OS when freed.
ENV MALLOC_ARENA_MAX=2 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_TRIM_THRESHOLD_=131072
EXPOSE 8077
# Authentication is mandatory off-localhost: pass CRYPTOK_AUTH=user:password at run time
# (from AWS Secrets Manager / SSM Parameter Store, never baked into the image).
ENTRYPOINT ["/app/cryptok", "serve", "--host", "0.0.0.0", "--port", "8077", "--no-open", "--lean", "--model", "/app/cryptok.cklm", "--sources", "/app/corpus"]
