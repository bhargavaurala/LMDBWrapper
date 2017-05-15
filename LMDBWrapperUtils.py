import numpy as np
from LMDBWrapper import LMDBWrapper

def coin_toss(bias = 0.5):	
	return True if np.random.random() < bias else False
	
def split(trainval, train, valid, val_fraction = 0.2):
	"""
	Pass trainval, train and val LMDBWrapper objects
	"""
	print 'splitting', trainval.lmdb_name, 'with validation fraction', val_fraction
	count = 0	
	for idx in np.random.permutation(trainval.env.stat()['entries']):
		value = trainval.read(idx = idx)
		if value is None:
			continue
		count += 1	
		selected = valid if coin_toss(bias = val_fraction) else train
		selected.write(value = value)
		if count % 10000 == 0:
			print '{} out of {} entries done. train count: {} val count: {}'.format(count, trainval.env.stat()['entries'], \
				train.env.stat()['entries'], valid.env.stat()['entries'])
	print 'trainval size {}, train size {} val size {}'.format(trainval.env.stat()['entries'], train.env.stat()['entries'], valid.env.stat()['entries'])