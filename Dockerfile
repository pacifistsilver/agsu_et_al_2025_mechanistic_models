FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    PYTENSOR_FLAGS=base_compiledir=/tmp/pytensor \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser \
 && mkdir -p /app /tmp/numba_cache /tmp/pytensor \
 && chown -R appuser:appuser /app /tmp/numba_cache /tmp/pytensor

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/stochtf && touch src/stochtf/__init__.py \
 && pip install --prefer-binary '.[inference,data]' \
 && pip uninstall -y stochtf

COPY --chown=appuser:appuser . .

RUN pip install --no-deps -e . && chown -R appuser:appuser /app/src

USER appuser

RUN python -c "import stochtf"

CMD ["/bin/bash"]