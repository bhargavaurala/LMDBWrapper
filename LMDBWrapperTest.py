import os
import numpy as np

from LMDBWrapper import LMDBWrapper
from LMDBWrapperUtils import split

def create_lmdb(lmdb_name):
	db = LMDBWrapper(lmdb_name = lmdb_name)
	data = {k : v for k, v in enumerate(range(20))}
	for k, v in data.iteritems():
		db.write(value = 'item_' + str(v))
	return db

if __name__ == '__main__':
	trainval_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperTest-trainval'
	train_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperTest-train'
	val_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperTest-val'
	trainval_lmdb = create_lmdb(trainval_lmdb_name)
	train_lmdb = LMDBWrapper(lmdb_name = train_lmdb_name)
	val_lmdb = LMDBWrapper(lmdb_name = val_lmdb_name)
	split(trainval_lmdb, train_lmdb, val_lmdb, val_fraction = 1. / 5)