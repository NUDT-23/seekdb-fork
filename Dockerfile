# MiniVectorDB - Linux x86_64, no network service, stdin/stdout driver.
FROM rust:1.85-slim-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked

FROM debian:bookworm-slim
RUN mkdir -p /data && chmod 0777 /data
COPY --from=builder /build/target/release/minivectordb /usr/local/bin/minivectordb
ENV MINIVEC_DATA_DIR=/data
# The evaluator connects pipes to stdin/stdout; do not use -d or -t.
ENTRYPOINT ["/usr/local/bin/minivectordb", "--data-dir", "/data"]
