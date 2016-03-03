import getopt
import sys
import lmdb
import caffe
import numpy as np

def mul(opts):
	base = 1
	for o in opts:
		base *= o
	return base


def lmdb2npy(src, dst):
	env = lmdb.open(src)
	data = []
	count = 0
	with env.begin() as txn:
		with txn.cursor() as cur:
			while cur.next():
				datum = caffe.proto.caffe_pb2.Datum()
				datum.ParseFromString(cur.value())
				array = caffe.io.datum_to_array(datum)
				data.append(array)
				count += 1
				if count % 1000 == 0:
					print "process %d files" % count
	if count % 1000:
		print "process %d files" % count
	data = np.squeeze(np.array(data))
	assert(len(data) == env.stat()['entries'])
	np.save(dst, data)

if __name__ == '__main__':
	if len(sys.argv) != 3:
		print 'Arguments error!'
		print 'lmdb2npy.py src dst'
		sys.exit(1)
	lmdb2npy(sys.argv[1], sys.argv[2])
