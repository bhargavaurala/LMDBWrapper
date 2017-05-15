import numpy as np
from PIL import Image

from LMDBWrapperCaffe import LMDBWrapperCaffe

def normalize_image(im):
	min_ = np.min(im)
	max_ = np.max(im)
	im1 = (im - min_) / float(max_ - min_)
	im1 *= 255
	return im1.astype('uint8')

def get_image_stats(db):
	"""
	Pass a LMDBWrapperCaffe object only
	"""
	if not isinstance(db, LMDBWrapperCaffe):
		print 'get_image_stats works only for LMDBWrapperCaffe objects'
		return None
	print 'getting mean and std deviation for lmdb:', db.lmdb_name
	count = 0
	with db.env.begin() as txn:
		print 'computing mean'
		for idx in xrange(db.env.stat()['entries']):			
			im, _ = db.read_img(idx)
			if im is None:
				continue			
			if count == 0:				
				mean = np.zeros_like(im, dtype = 'float32')
				stddev = np.zeros_like(im, dtype = 'float32')			
			mean += im
			count += 1
		mean /= count
		print 'computing standard deviation'
		for idx in xrange(db.env.stat()['entries']):			
			im, _ = db.read_img(idx)
			if im is None:
				continue			
			stddev += (im - mean) ** 2
		stddev /= count
		stddev **= 0.5	
	return mean, stddev

def preprocess(db, mean, stddev, dtype = 'uint8'):
	"""
	Pass a LMDBWrapperCaffe object, mean image and standard deviation image
	"""
	if not isinstance(db, LMDBWrapperCaffe):
		print 'preprocess works only for LMDBWrapperCaffe objects'
		return None
	print 'subtracting mean and dividing by standard deviation for lmdb:', db.lmdb_name
	for idx in xrange(db.env.stat()['entries']):
		im, label = db.read_img(idx)
		im = im.astype('float32')
		im -= mean
		im /= stddev
		im = im.transpose(2, 0, 1) if not db.channels_first else im # convert to C, H, W if not already
		im = np.concatenate([normalize_image(im_)[np.newaxis, ...] for im_ in im], axis = 0) # compute channelwise normalized image		
		if not db.channels_first: #convert back to H, W, C if necessary
			im = im.transpose(1, 2, 0)
		db.write_img(im, label, idx = idx)

def view(db, view_label = None):
	"""
	Pass a LMDBWrapperCaffe object
	"""
	if not isinstance(db, LMDBWrapperCaffe):
		print 'view_lmdb works only for LMDBWrapperCaffe objects'
		return None
	print 'viewing', db.lmdb_name
	with db.env.begin() as txn:
		while True:
			idx = np.random.randint(0, db.env.stat()['entries'])
			im, label = db.read_img(idx)
			print im.shape
			im = im.transpose(1, 2, 0) if db.channels_first else im
			vlabel = label if view_label is None else view_label
			if label == vlabel:
				Image.fromarray(im[:, :, ::-1]).show()
				q = raw_input('label {}. Hit q to quit '.format(label))								
				if q == 'q' or q == 'Q':
					break