FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir numpy scipy sympy pytest pyyaml "jax[cpu]>=0.4.20"
ENV JAX_PLATFORMS=cpu
ENV XLA_FLAGS="--xla_force_host_platform_device_count=4"
CMD ["pytest", "tests/", "-v", "--tb=short"]
