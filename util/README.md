# SGDTrainer

A simple trainer which call `accGradParameters()` and `updateParameters()` seperately to avoid clash in customed module.

```lua
require 'util/SGDTrainer'
trainer = SGDTrainer(model, criterion)
trainer:train(trainset, testset)
```

Before `trainer:train(trainset, testset)` you can modify some variables of SGDTrainer to adjust the behaviour of it. 

* SGDTrainer.lr - Learning Rate.   
* SGDTrainer.maxEpoch  
* SGDTrainer.displayIter - The interval of Iteration to display loss and accuray.  
* SGDTrainer.snapshotIter - The interval of Iteration to save the model.  
* SGDTrainer.snapshotPrefix - The model save as `[Prefix}_[Iteration].t7`.  

The second argument of SGDTrainer:train() is optional, if given, the trainer will display the accuray in testset after each epoch.

# util

Some useful tools.

## flatten(tt)

Flatten the given table `tt`.

## extract\_vocab(items, ...)

Generate a set from items which can be used as a vocabulary. return `item_to_index` and `index_to_item`. The extra argument will be add at the head of the set. This is useful for special word that items is not contain, like `*unk*` for unkown word.

## split\_line(str, skip)

Split the str using `\n` as the seperator. If the `skip` is true, the null line will be skiped.

## split\_word(str)
Split the str using the character from `%S` as the seperator.

## start\_with(str, head)
Check if the `str` starts with `head`. If so, return `true`.

## accuracy(net, dataset)
Feed the dataset to the trained net and return the accurcy the net reaches.

## narrow(dataset, size)
Chop the dataset to the given `size`. This is useful when check the model.

## lookup(indices, table)
Lookup Table. `indices` is a list of index, `lookup` replace the i-th index `indices[i]` with `table[indices[i]]`.
