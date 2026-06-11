FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

# Create non-root user for HPC environments that don't use Singularity
# For HPCs using Singularity/Apptainer, the host user automatically overrides this, which is fine.
RUN useradd -m appuser && \
    mkdir -p /app /tmp/numba_cache && \
    chown -R appuser:appuser /app /tmp/numba_cache

WORKDIR /app

# Install requirements first for layer caching
COPY requirements.txt .
RUN pip3 install --prefer-binary --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY . .
RUN chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["snakemake"]
CMD ["--cores", "1", "--snakefile", "multiparam_set.smk"]