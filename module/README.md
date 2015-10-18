# sentenceCNN

Use the Convolution Neural Network to encode the sentence.

```lua
require 'module/sentenceCNN'

model = sentenceCNN(50, {{3,1,200,2,2},
                         {3,1,300,2,2},
                         {3,1,300,2,2}})
question = torch.rand(30,50)
output = model:forward(question)
```

Format:  
sentenceCNN(emb\_size, {{kwidth, kstrid, kcount, pwidth, pstrid},...})
