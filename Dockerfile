FROM rust:1.85-slim-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN RUSTFLAGS='-C target-feature=+crt-static' cargo build --release --locked
RUN mkdir /runtime-data && chmod 0777 /runtime-data

FROM scratch
COPY --from=builder /runtime-data /data
COPY --from=builder /build/target/release/minivectordb /usr/local/bin/minivectordb
ENV MINIVEC_DATA_DIR=/data
ENTRYPOINT ["/usr/local/bin/minivectordb", "--data-dir", "/data"]
