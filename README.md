# Protein Identification Reproduction

This repository provides the environment and files required to reproduce the protein-identification result.

## Environment

Python version:

```bash
Python 3.8.20
```

Dependencies: requirements.txt


## Data Preparation

Download the Davis dataset from Google Drive:

[Download Davis dataset](https://drive.google.com/drive/folders/1SwnO4kGNgtrys8nQ-eEgjJ_P2LgNBtuB?usp=drive_link)

Download the pretrained model:

[Download model_300dim.pkl](https://drive.google.com/file/d/1ZcK8C5f01WctelBpHYLq9BzmAT62zYHU/view?usp=sharing)

Move the Davis dataset to:

```bash
./data
```

Move the pretrained model to the project root directory:

```bash
./model_300dim.pkl
```

The expected directory structure is:

```text
.
├── data/
│   └── Davis/
├── model_300dim.pkl
├── train_src_upload.py
└── requirements.txt
```

## Reproduce Protein-Identification Result

Run the following command:

```bash
python train_src_upload.py
```
