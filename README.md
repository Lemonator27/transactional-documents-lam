# Transactional Documents
This repository contains code and data related to the paper titled "Neurosymbolic Information Extraction from Transactional Documents".

## Project Structure

```
├── src/
│   ├── label_studio/
│   │   ├── import_label_studio.py    # Script to import and validate Label Studio annotations
│   │   ├── label_studio_model.py     # Label Studio data models and entity types, for convenience
│   │   └── label_studio_interface.xml # Label Studio labeling interface configuration
│   ├── model/
│   │   ├── constraints.py     # Constraint definitions
│   │   ├── dataset.py         # Dataset loading utilities
│   │   ├── document.py        # Document model definition
│   │   └── invoice.py         # Invoice models and constraint definitions
│   └── utils/
│       ├── config.py          # Configuration management
│       ├── json_evaluator.py  # JSON comparison and evaluation metrics
│       ├── metrics.py         # Evaluation metrics computation
│       ├── parallel.py        # Parallel processing utilities
│       ├── ocr.py             # OCR result handling utilities
│       └── utils.py           # Utility functions
├── tests/                     # Test files
└── README.md
```

## Datasets
The relabeled CORD and SROIE datasets are available from the links below:

- [CORD]()
- [SROIE]()

Each dataset is organized as follows:

```
dataset_name/
├── labeled.json               # Label Studio format annotations
├── images/                    # Document images
│   └── *.jpg
├── ocr/                       # OCR results
│   └── *.jpg.json
├── train-documents.jsonl      # Generated training split
├── validation-documents.jsonl # Generated validation split
└── test-documents.jsonl       # Generated test split
```

## Data Processing
While the source annotations have already been converted to the `Document` format, you can use the `src/label_studio/import_label_studio.py` script to convert the annotations and debug or evaluate the constraints. When a document fails validation, it will print:
- The document ID
- Which constraints failed

Example output for a constraint violation:
```
Invalid constraints for document validation-21-0:
net_price=net_unit_pricequantity
gross_price=net_price+tax_amount
```

## Environment Setup
Create and activate a conda environment using the provided environment.yml:
```bash
conda env create -f environment.yml
conda activate td
```