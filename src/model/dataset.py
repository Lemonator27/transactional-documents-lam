from pathlib import Path
from typing import List, TypeVar

from pydantic import BaseModel

from model.document import Document
from utils.config import data_dir

T = TypeVar('T', bound=BaseModel)

def load_dataset(name: str, split: str) -> List[Document[T]]:
    from model.invoice import Invoice
    split_path = Path(data_dir) / name / f'{split}-documents.jsonl'

    # Load the documents from the JSONL file
    docs = []
    with open(split_path, 'r', encoding='utf-8') as f:
        for line in f:
            docs.append(Document[Invoice].model_validate_json(line))
    
    return docs

if __name__ == '__main__':
    docs = load_dataset('cord', 'validation')
    print(docs[0].model_dump_json(indent=2))