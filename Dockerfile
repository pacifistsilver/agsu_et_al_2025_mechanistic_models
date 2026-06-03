FROM python:3.13
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --prefer-binary --no-cache-dir --upgrade -r requirements.txt
COPY . .
ENTRYPOINT ["snakemake"]
CMD ["--cores", "1", "--snakefile", "Snakefile.smk"]