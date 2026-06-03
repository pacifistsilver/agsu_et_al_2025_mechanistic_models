# Protein-protein interactions drive differences in the spatiotemporal dynamics of transcription factors NANOG and SOX2 in naïve pluripotent cells

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Supported-orange.svg)
![Snakemake](https://img.shields.io/badge/Snakemake-Enabled-brightgreen.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)

## repo structure
```text
├── notebooks/             # Jupyter notebooks for data exploration
├── scripts/               # Snakemake helper scripts 
├── src/
│   └── expression_model/  # modules for the stochastic model
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
   git clone [https://github.com/pacifistsilver/Srinjan_Modelling.git](https://github.com/pacifistsilver/Srinjan_Modelling.git)
   cd Srinjan_Modelling
```
2. Build w/ Docker
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
docker run -it -v $(pwd):/app pacifistsilver/stochresidence:0.1 /bin/bash
```



Developed for LIFE70040 / SB-RPC Joint Lab Project
