import os

from dotenv import load_dotenv

load_dotenv()
data_dir = os.environ.get('SCRATCH', None)
if data_dir is None:
    data_dir = os.environ['DATA_DIR']