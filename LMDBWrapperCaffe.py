import caffe
import numpy as np

from LMDBWrapper import LMDBWrapper

class LMDBWrapperCaffe(LMDBWrapper):
	def __init__(self, lmdb_name, init_sample_estimate = 250 * 400, img_size = (224, 224, 3), channels_first = False):		
		LMDBWrapper.__init__(self, lmdb_name, init_sample_estimate = init_sample_estimate, sample_size = np.prod(img_size))
		self.img_size = (224, 224, 3)
		self.channels_first = channels_first		

	def read_img(self, idx):
		value = self.read(idx = idx)
		if value is None:
			return None, None
		datum = caffe.proto.caffe_pb2.Datum()
		datum.ParseFromString(value)
		im = np.fromstring(datum.data, dtype = 'uint8')		
		im = im.reshape(datum.channels, datum.height, datum.width)
		if not self.channels_first:
			im = im.transpose(1, 2, 0)
		return im, datum.label

	def write_img(self, img, label, idx = None):
		if not self.channels_first:
			img = img.transpose(2, 0, 1)
		datum = caffe.proto.caffe_pb2.Datum()
		datum.channels = img.shape[0]		
		datum.height = img.shape[1]
		datum.width = img.shape[2]
		datum.data = img.tobytes()
		datum.label = label
		key = self.keygen(self.idx) if idx is None else self.keygen(idx)
		self.idx = self.idx + 1 if idx is None else self.idx
		self.write(key = key, value = datum.SerializeToString())
