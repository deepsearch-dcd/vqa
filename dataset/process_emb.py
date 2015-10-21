import numpy as np

def load_emb(src):
	with open(src) as f:
		lines = f.readlines()
	total, size = lines[0].strip().split(' ')
	total = int(total); size = int(size)
	print('total: %d, embedding size: %d' % (total, size))
	del(lines[0])

	words = []; embs = []
	for l in lines:
		items = [ item for item in l.strip().split(' ') if item ]
		assert(len(items) == size + 1)
		words.append(items[0])
		embs.append(np.array([float(item) for item in items[1:]]))
	if '*unk*' not in words:
		mean = np.zeros(size)
		for e in embs:
			mean += e
		mean = mean / len(embs)
		words.append('*unk*')
		embs.append(mean)
	if '*pad*' not in words:
		words.append('*pad*')
		embs.append(np.zeros(size))
	assert(len(words) == len(embs))
	return words, np.array(embs)

def convert(src, dst_dir):
	words, embs = load_emb(src)
	identity = src.rsplit('/', 1)[-1]
	with open('%s/%s_words.txt' % (dst_dir, identity), 'w') as f:
		f.write('\n'.join(words)+'\n')
	np.save('%s/%s_embs.npy' % (dst_dir, identity), embs)

def process():
	todo = ['CBOW_50', 'CBOW_100', 'CBOW_200', 'CBOW_300', 'CBOW_500',
		'SG_50', 'SG_100', 'SG_200', 'SG_300', 'SG_500', 'HLBL_50']
	for f in todo:
		convert('word_embedding/origin/'+f, 'word_embedding/')

if __name__ == '__main__':
	process()
