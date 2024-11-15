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
This repository contains redistributions of:

1. SROIE ([Google Drive](https://drive.google.com/file/d/114d5XjQr0RzU9QhDL8IYHK8_pl8XYsTc/view?usp=drive_link))
- Original source: https://github.com/zzzDavid/ICDAR-2019-SROIE
- Copyright (c) 2019 Niansong Zhang, Songyi Yang, Shengjie Xiu
- Licensed under MIT License

2. CORD ([Google Drive](https://drive.google.com/file/d/1-rMlC10AiYvDnAyMoF0JIYCcxuZm5r2_/view?usp=drive_link))
- Original source: https://github.com/clovaai/cord
- Licensed under CC BY 4.0
- Please cite the following papers when using this dataset:
```
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
```

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

## Licensing
The source code is licensed under the [Apache 2.0 License](LICENSE). The datasets are licensed under their respective licenses listed above.