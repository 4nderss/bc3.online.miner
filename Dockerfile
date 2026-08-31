# One image for every backend.
#
# The miner opens libcuda and libOpenCL with dlopen rather than linking against
# them, so the same binary mines on NVIDIA, on AMD/Intel, or on CPU alone,
# depending only on what the host exposes. That is why there is one image here
# and not three.
#
# No CUDA toolkit is installed: the kernel is precompiled to PTX at build time
# and the driver JITs it. The build stage needs nothing but Rust.

FROM rust:1-trixie AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked

FROM debian:trixie-slim
# ocl-icd is the OpenCL loader, not a driver: it finds whatever ICD the host
# mounts in. Without it AMD and Intel users would be stuck on CPU inside the
# container, which is exactly the group the OpenCL backend was written for.
# Pinned by tag rather than digest on purpose - the base is rebuilt for
# security fixes, and every release rebuilds this image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ocl-icd-libopencl1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/target/release/bc3-miner /usr/local/bin/bc3-miner

# Read by the NVIDIA container runtime. They make GPUs visible when the daemon
# already uses that runtime; with the stock runtime you still need `--gpus all`
# on the run command. Nothing inside an image can request a GPU by itself.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Mining needs no privileges and writes nothing. Run as nobody.
USER 65534:65534

# No shell in between: signals reach the miner directly, so `docker stop`
# actually stops it instead of waiting out the timeout.
ENTRYPOINT ["/usr/local/bin/bc3-miner"]
