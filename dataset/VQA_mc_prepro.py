# Preprocess the raw VQA multiple choice question data.
# Note that lowering and stemming will be applied to words.

import sys
import itertools
import json
import random

import nltk
import numpy as np

WORK_PATH = 'dataset/data/VQA/'
QUES_PATH_PATTERN = WORK_PATH + 'MultipleChoice_mscoco_%s_questions.json'
ANNS_PATH_PATTERN = WORK_PATH + 'mscoco_%s_annotations.json'
SAVE_PATH_PATTERN = WORK_PATH + 'MC_seq_qtype_%s_%d.npy'

EMBEDDINGS_PATH = 'word_embedding/CBOW_50_embs.npy'
WORD_LIST_PATH = 'word_embedding/CBOW_50_words.txt'


# return a list of samples.
# sample format: `question || choices`
# `question` is a list of embeddings of the words in question.
# `choices` is a list of choices. each choice is represented as the sum of the 
# word embeddings.
def prepare_data(split_name, ratio=None):

    print 'split name: ', split_name
    
    # get the raw data
    with open(QUES_PATH_PATTERN % split_name) as f:
        ques_raw = json.load(f)
    with open(ANNS_PATH_PATTERN % split_name) as f:
        anns_raw = json.load(f)

    # sampling
    total = len(ques_raw['questions'])
    idx = xrange(total)
    if ratio is not None:
        total = int(total*ratio)
        idx = random.sample(idx, total)
    else:
        ratio = 1
    print 'question count after sampling: ', total

    # index embeddings
    embs = np.load(EMBEDDINGS_PATH)
    with open(WORD_LIST_PATH) as f:
        words = f.readlines()
    words = [ w.strip() for w in words ]
    word_to_emb = {}
    for i,w in enumerate(words):
        word_to_emb[w] = embs[i]
    del(words)
    del(embs)

    # the function to get embedding from word
    def to_emb(word):
        if word in word_to_emb:
            return word_to_emb[word]
        else:
            return word_to_emb['*unk*']

    samples = []
    stemmer = nltk.PorterStemmer()
    for count, i in enumerate(idx):
        q = nltk.word_tokenize(ques_raw['questions'][i]['question'].lower())
        q = [ to_emb(stemmer.stem(t)) for t in q ]
        a = anns_raw['annotations'][i]['multiple_choice_answer'].lower()
        choices = []
        for choice in ques_raw['questions'][i]['multiple_choices']:
            tokens = nltk.word_tokenize(choice.lower())
            emb = np.sum([to_emb(stemmer.stem(t)) for t in tokens], axis=0)
            if choice != a:
                choices.append(emb)
            else:
                choices.insert(0, emb)
        samples.append([q, choices])
        sys.stdout.write('\rprepare data: %.2f%%' % ((count+1.)/total*100))
    sys.stdout.write('\n')

    return samples, ratio


def save_data(save_path, samples):
    with open(save_path, 'w') as f:
        np.save(f, np.array(samples))


# distrubte the qa pairs sequentially
# sample format: `label || question || answer`
# the valid value of `label` is `1` or `-1`, `1` means positive and `-1` means 
# negative; `question` is represent as the embeddings of the first two words
# of the question; `answer` is the sum of all the embeddings responding to the 
# words in answer.
def distribute_seq(split_name, ratio=None):

    questions, ratio = prepare_data(split_name, ratio)
    total = len(questions)
    samples = []
    for count, question in enumerate(questions):
        # concatenate the first two word
        q = np.append(question[0][0], question[0][1])
        for choice in question[1]:
            samples.append(np.append(-1, np.append(q, choice)))
        samples[-18][0] = 1
        sys.stdout.write('\rdistributing sequentially: %.2f%%' % 
                            ((count+1.)/total*100))
    sys.stdout.write('\n')
    assert(len(samples) == total*18)   

    save_data(SAVE_PATH_PATTERN % (split_name,ratio*100), samples)


# distribute as question+choice group
# each sample is a matrix which has 18 rows. The first row corresponds to the 
# true choice.
# row format: `question || answer`
# `question` is represent as the embeddings of the for two words in it; 
# `answer` is the sum of all the words in it.
def distribute_group(split_name, ratio=None):
    
    questions, ratio = prepare_data(split_name, ratio)
    total = len(questions)
    q_fea_len = len(questions[0][0][0])*2
    c_fea_len = len(questions[0][1][0])
    samples = np.zeros((len(questions), 18, q_fea_len+c_fea_len))
    for i, question in enumerate(questions):
        q = np.append(question[0][0], question[0][1])
        for j, c in enumerate(question[1]):
            samples[i][j][:q_fea_len] = q
            samples[i][j][q_fea_len:] = c
        sys.stdout.write('\rdistributing by group: %.2f%%' % ((i+1.)/total*100))
    sys.stdout.write('\n')
    
    save_data(SAVE_PATH_PATTERN % (split_name,ratio*100), samples)

if __name__ == '__main__':
    
    distribute_seq('train2014', 0.25)
    distribute_seq('val2014', 0.25)
    #distribute_seq('train2014', 0.01)
    #distribute_seq('val2014', 0.01)
    #distribute_group('train2014', 0.01)
    #distribute_group('val2014', 0.01)
    #distribute_group('train2014', 0.25)
    #distribute_group('val2014', 0.25)
