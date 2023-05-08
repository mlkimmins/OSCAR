
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

dataset = load_dataset("scientific_papers", "pubmed", split="train", cache_dir = '/content/drive/MyDrive/109b project/dataset_cache')
num_proc = 8
def concatenate_text(example):
    # Concatenate the article, abstract, and section names using some delimiter, e.g., '\n'
    text = f"ABSTRACT STARTS HERE:\n{example['abstract']}\nSECTIONS START HERE:\n{example['section_names']}\nARTICLE STARTS HERE:{example['article']}\n THIS IS THE END OF THE ARTICLE"
    return {'text': text}


dataset = dataset.map(concatenate_text, 
            remove_columns = ['abstract','section_names','article'],
            desc = 'one column')
num_examples = len(dataset)
dataset = dataset.select(range(0,num_examples - 60))

split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

enc = tiktoken.get_encoding("gpt2")
def process(example):
    ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
    ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
    # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
    out = {'ids': ids, 'len': len(ids)}
    return out

# tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
)

# concatenate all the ids in each dataset into one large file we can use for training
for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'])
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()