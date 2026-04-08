FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir numpy scipy sympy pytest pyyaml "jax[cpu]"
CMD ["pytest", "tests/", "-v", "--tb=short"]
