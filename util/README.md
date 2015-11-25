# train.lua

Tool to train a model.

```lua
require 'util/train'

train(opt, model, criterion, trainset, testset)
```

`opt` is a table containing all the parameters the `train()` need.

`trainset` and `testset` is a table containing the fields as follow, `images`, `questions`, `answers`, `nsample`, `nimage`, `nvocab`, `nanswer`.

If `testset` is not given, then `train()` will not test the model during training.

# Plotter.lua

A plot tool like `optim.Logger`. 

# util.lua

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

## eval(net, criterion, dataset)
Feed the dataset to the trained net and return the accurcy and loss the net reaches.

## narrow(dataset, size)
Chop the dataset to the given `size`. This is useful when check the model.

## lookup(indices, table)
Lookup Table. `indices` is a list of index, `lookup` replace the i-th index `indices[i]` with `table[indices[i]]`.
