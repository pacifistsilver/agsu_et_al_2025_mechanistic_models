# Stochastic modelling of NANOG/SOX2 dynamics in gene expression noise

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## repo structure
```text
├── notebooks/             
├── scripts/               
├── src/  
├── Dockerfile             
├── Snakefile.smk          
├── environment.yaml       
├── requirements.txt       
├── submission_script.sh   
└── README.md          
```

## installation and running

### Option 1: Build local Docker container
1. Clone the repository:
```bash
   git clone [https://github.com/pacifistsilver/1d-spatial-gillespie.git](https://github.com/pacifistsilver/1d-spatial-gillespie.git)
   cd 1d-spatial-gillespie
```
2. Build with Docker
```
docker build -t srinjan_modelling:latest .
```
3. Run the container
```
docker run -it -v $(pwd):/app srinjan_modelling:latest /bin/bash
```
### Option 2: Pull Docker image and run
1. Pull container
```
docker pull pacifistsilver/stochresidence:0.1
```

2. Run the container
```
docker run -it -v $(pwd):/app pacifistsilver/stochresidence:0.1 
```


## Publication
G. G. Agsu et al., “Protein-protein interactions drive differences in the spatiotemporal dynamics of transcription factors NANOG and SOX2 in naïve pluripotent cells,” Dec. 2025, doi: https://doi.org/10.64898/2025.12.03.691924.