# sample 5 questions from each type

import json
import random

def sample():
    dataset_path = 'dataset/data/Visual7W/visual7w-toolkit/datasets/visual7w-telling/dataset.json'
    with open(dataset_path) as f:
        dataset = json.load(f)
    assert(dataset)
    qa_pairs = [' '.join([qa['question'], qa['answer']]) for image in dataset['images'] for qa in image['qa_pairs']]

    types = {}
    for qa in qa_pairs:
        k = qa.split()[0]
        if k not in types:
            types[k] = []
        types[k].append(qa)

    for k,v in types.items():
        if len(v) > 5:
            indice = range(len(v))
            random.shuffle(indice)
            for i in range(5):
                print v[indice[i]]
        else:
            for q in v:
                print q


if __name__ == '__main__':
    sample()
