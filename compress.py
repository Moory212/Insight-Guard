import gzip 
import shutil 
with open('data/raw/unified_dataset.csv', 'rb') as f_in: 
    with gzip.open('data/raw/unified_dataset.csv.gz', 'wb') as f_out: 
        shutil.copyfileobj(f_in, f_out) 
