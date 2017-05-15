import os
import cv2
import numpy as np

from LMDBWrapperCaffe import LMDBWrapperCaffe
from LMDBWrapperUtils import split
from LMDBWrapperCaffeUtils import get_image_stats, preprocess, view

def create_lmdb(src_folder, lmdb_name):
	db = LMDBWrapperCaffe(lmdb_name = lmdb_name, channels_first = False)
	_, dirs, _ = os.walk(src_folder).next()
	for dir_ in dirs:
		label = int(dir_)
		_, _, files = os.walk(os.path.join(src_folder, dir_)).next()
		for f in files:
			im = cv2.imread(os.path.join(src_folder, dir_, f))			
			if 0 in im.shape:
				print dir_, f
			db.write_img(im, label)
	return db

if __name__ == '__main__':
	trainval_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperCaffeTest-trainval'
	train_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperCaffeTest-train'
	val_lmdb_name = '/media/buralako/Data1/CMU-MultiPie-LMDBWrapperCaffeTest-val'
	trainval_lmdb = create_lmdb('/media/buralako/Data1/CMU-MultiPie-LMDBWrapperTestDB', trainval_lmdb_name)
	# view(trainval_lmdb)
	trainval_lmdb = LMDBWrapperCaffe(lmdb_name = trainval_lmdb_name)
	train_lmdb = LMDBWrapperCaffe(lmdb_name = train_lmdb_name)
	val_lmdb = LMDBWrapperCaffe(lmdb_name = val_lmdb_name)
	split(trainval_lmdb, train_lmdb, val_lmdb, val_fraction = 1. / 3)
	# view(train_lmdb)
	# view(val_lmdb)
	train_mean, train_std = get_image_stats(train_lmdb)
	preprocess(train_lmdb, train_mean, train_std)
	preprocess(val_lmdb, train_mean, train_std)
	view(train_lmdb)
	view(val_lmdb)







