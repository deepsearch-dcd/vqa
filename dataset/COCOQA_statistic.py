import sys

def statistic(split, frac, k):
    with open('dataset/data/COCO-QA/' + split + '/questions.txt') as f:
        questions = f.readlines()
    words = [word for word_list in [q.strip().split() for q in questions] 
                for word in word_list]
    word_and_count = {word:0 for word in set(words)}
    for word in words:
        word_and_count[word] += 1
    word_by_count = [[word, count] for word, count in word_and_count.items()]
    word_by_count = sorted(word_by_count, key=lambda t: -t[1])
    for i in range(1, len(word_by_count)):
        word_by_count[i][1] += word_by_count[i-1][1]
    for i in range(len(word_by_count)):
        word_by_count[i][1] = 1.0 * word_by_count[i][1] / word_by_count[-1][1]
    word_and_frac = word_by_count
    del(word_by_count)

    if frac is not None:
        for ith, item in enumerate(word_and_frac):
            if item[1] >= frac:
                print('{:.4f} of words before the {}-th word.'
                    .format(frac, ith))
                break

    if k is not None:
        print('the fraction of the first {} words is {:4f}'
            .format(k, word_and_frac[k][1]))

if __name__ == '__main__':
    k = None
    if len(sys.argv) == 3: 
        split = sys.argv[1]
        frac = float(sys.argv[2])
    elif len(sys.argv) == 4:
        split = sys.argv[1]
        frac = float(sys.argv[2])
        k = int(sys.argv[3])
    else:
        print('argument error!\nUsage: {} split frac [k]'
            .format(sys.argv[0]))
        sys.exit()
    statistic(split, frac, k)

