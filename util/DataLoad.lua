function loadData(args)
  print('loading '.. args.dataset ..' datasets')
  local trainset, testset, vocab
  if args.dataset == 'DAQUAR' then
    trainset, testset, vocab = DAQUAR.process_to_table()
  elseif args.dataset == 'COCOQA' then
    if args.caption then
      trainset, testset, vocab = COCOQA.load_data{format='table', add_pad_word=false, add_unk_word=true, add_unk_answer=false, load_caption=args.capopt}
    else
      trainset, testset, vocab = COCOQA.load_data{format='table', add_pad_word=false, add_unk_word=true, add_unk_answer=false}
    end
    trainset.answers = torch.Tensor(trainset.answers)
    testset.answers = torch.Tensor(testset.answers)
  else
    error('Unknown dataset')
  end

  if args.caption and args.textonly then
    for i=1,trainset.size do
      local captions = trainset.captions[i]
      assert(captions~=nil,'caption nil in: '..i)
      local newques = {}
      if not args.caponly then -- QUES
        local ques = trainset.questions[i]
        for j=1,#ques do
          table.insert(newques, ques[j]) --newques[#newques+1] = ques[j]
        end
      end
      for j=1,1 do --1,#captions -- CAP
        local cap = captions[j]
        for k=1,#cap do
          table.insert(newques, cap[k]) --newques[#newques+1] = cap[k]
        end
      end
      trainset.questions[i] = newques
    end
    trainset.captions = nil
    collectgarbage()
    for i=1,testset.size do
      local captions = testset.captions[i]
      assert(captions~=nil,'caption nil in: '..i)
      local newques = {}
      if not args.caponly then -- QUES
        local ques = testset.questions[i]
        for j=1,#ques do
          table.insert(newques, ques[j]) --newques[#newques+1] = ques[j]
        end
      end
      for j=1,1 do --1,#captions -- CAP
        local cap = captions[j]
        for k=1,#cap do
          table.insert(newques, cap[k]) --newques[#newques+1] = cap[k]
        end
      end
      testset.questions[i] = newques
    end
    testset.captions = nil
    collectgarbage()
  
    print('Append captions with question done.')
  end
  
  -- Remove determiner
  if args.rmdeter then
    -- build determiner set
    local determiner = {}
    addToSet(determiner,vocab.word_to_index['a'])
    addToSet(determiner,vocab.word_to_index['an'])
    addToSet(determiner,vocab.word_to_index['the'])
    addToSet(determiner,vocab.word_to_index['this'])
    addToSet(determiner,vocab.word_to_index['that'])
    addToSet(determiner,vocab.word_to_index['these'])
    addToSet(determiner,vocab.word_to_index['those'])
    --addToSet(determiner,vocab.word_to_index['such'])
    --addToSet(determiner,vocab.word_to_index['my'])
    --addToSet(determiner,vocab.word_to_index['your'])
    --addToSet(determiner,vocab.word_to_index['his'])
    --addToSet(determiner,vocab.word_to_index['her'])
    --addToSet(determiner,vocab.word_to_index['our'])
    --addToSet(determiner,vocab.word_to_index['their'])
    --addToSet(determiner,vocab.word_to_index['its'])
    --addToSet(determiner,vocab.word_to_index['some'])
    --addToSet(determiner,vocab.word_to_index['any'])
    --addToSet(determiner,vocab.word_to_index['each'])
    --addToSet(determiner,vocab.word_to_index['every'])
    --addToSet(determiner,vocab.word_to_index['no'])
    --addToSet(determiner,vocab.word_to_index['either'])
    --addToSet(determiner,vocab.word_to_index['neither'])
    --addToSet(determiner,vocab.word_to_index['enough'])
    --addToSet(determiner,vocab.word_to_index['all'])
    --addToSet(determiner,vocab.word_to_index['both'])
    --addToSet(determiner,vocab.word_to_index['several'])
    --addToSet(determiner,vocab.word_to_index['many'])
    --addToSet(determiner,vocab.word_to_index['much'])
    --addToSet(determiner,vocab.word_to_index['few'])
    --addToSet(determiner,vocab.word_to_index['little'])
    --addToSet(determiner,vocab.word_to_index['other'])
    --addToSet(determiner,vocab.word_to_index['another'])
  
    for i=1,trainset.size do
      local ques = trainset.questions[i]
      for j=#ques,1,-1 do
        if setContains(determiner, ques[j]) then
          table.remove(ques,j)
        end
      end
    end
    for i=1,testset.size do
      local ques = testset.questions[i]
      for j=#ques,1,-1 do
        if setContains(determiner, ques[j]) then
          table.remove(ques,j)
        end
      end
    end
  
    print('Remove determiner done.')
  end
  
  -- convert table to Tensor
  for i=1,trainset.size do
    trainset.questions[i] = torch.Tensor(trainset.questions[i])
  end
  for i=1,testset.size do
    testset.questions[i] = torch.Tensor(testset.questions[i])
  end
  
  -- convert to cuda
  if args.cuda then
    for i=1,trainset.size do
      trainset.questions[i] = trainset.questions[i]:float():cuda()
    end
    for i=1,testset.size do
      testset.questions[i] = testset.questions[i]:float():cuda()
    end
  end

  ---------- load features ----------
  if not args.textonly then
    local feas = loadFea(args)
    trainset.imagefeas = feas
    testset.imagefeas = feas
  end

  return trainset, testset, vocablo
end

function loadFea(args)
  print('loading features')
  local feas
  if args.dataset == 'DAQUAR' then
    feas = npy4th.loadnpy('./feature/DAQUAR-ALL/GoogLeNet-1000-softmax/im_fea.npy')
  elseif args.dataset == 'COCOQA' then
    if args.im_fea_dim==1000 then
      feas = npy4th.loadnpy('./feature/COCO-QA/GoogLeNet-1000.npy') -- GoogLeNet-1000-softmax.npy GoogLeNet-1000.npy VGG19-1000-softmax.npy VGG19-1000.npy
    elseif args.im_fea_dim==1024 then
      feas = npy4th.loadnpy('./feature/COCO-QA/GoogLeNet-1024.npy')
    elseif args.im_fea_dim==4096 then
      feas = npy4th.loadnpy('./feature/COCO-QA/VGG19-4096-relu.npy') -- VGG19-4096-relu.npy VGG19-4096.npy
    end
  end
  if args.cuda then
    feas = feas:float():cuda()
  end

  return feas
end