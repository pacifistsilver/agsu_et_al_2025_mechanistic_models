FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLBACKEND=Agg

# Create non-root user for HPC environments that don't use Singularity
# For HPCs using Singularity/Apptainer, the host user automatically overrides this, which is fine.
RUN useradd -m appuser && \
    mkdir -p /app /tmp/numba_cache && \
    chown -R appuser:appuser /app /tmp/numba_cache

WORKDIR /app

# Install dependencies first for layer caching
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip3 install --prefer-binary --no-cache-dir --upgrade '.[inference,data]'

# Copy the rest of the project
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

# The previous entrypoint invoked `snakemake --snakefile multiparam_set.smk`,
# but no Snakefile exists anywhere in the repository, so `docker run` failed
# immediately. Default to a shell; run the scripts explicitly, e.g.
#   docker run --rm -v $(pwd):/app stochtf python figures/fig01_burst_parameters.py
CMD ["/bin/bash"]
