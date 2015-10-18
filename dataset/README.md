# DAQUAR

Process the DAQUAR dataset.
```lua
DAQUAR = requre 'dataset/DAQUAR'
trainset, testset, vocab = DAQUAR.process_and_check()
```
The output also save to `dataset/data/DAQUAR/DAQUAR-ALL/done/`.

# lmdb2npy.py
An python script which can convert the feature in lmdb produced by caffe into npy format, then we can use `npy4th` to load the data in lua.
