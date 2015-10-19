# DAQUAR

Process the DAQUAR dataset.
```lua
DAQUAR = requre 'dataset/DAQUAR'
trainset, testset, vocab = DAQUAR.process_and_check()
```
The output also save to `dataset/data/DAQUAR/DAQUAR-ALL/done/`.

# lmdb2npy.py

An python script which can convert the feature in lmdb produced by caffe into npy format, then we can use `npy4th` to load the data in lua.

# process\_emb.py

An python script which convert the word embeddings from `txt` to `npy` format. The output save in `word_embedding/`. `*_words.txt` is word list,  `*_embs.npy` is the embedding list matching the `*_word.txt`.

# process\_emb.lua

Use `torch.save()` to save the output of `process_emb.py`. Output to `word_embedding/*.t7`. The `*.t7` file has two entries, a table `word_to_index` and a tensor `index_to_emb`.
