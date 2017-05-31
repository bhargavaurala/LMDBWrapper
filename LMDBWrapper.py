import lmdb

class LMDBWrapperBase(object):
	def __init__(self, lmdb_name, init_sample_estimate = 1000, sample_size = 64):				
		self.lmdb_name = lmdb_name
		self.init_sample_estimate = init_sample_estimate
		self.sample_size = sample_size
		self.env = lmdb.open(lmdb_name, map_size = init_sample_estimate * sample_size)
		self.idx = self.env.stat()['entries'] - 1

	def keygen(self, idx):
		raise NotImplementedError("Implement a keygen function to map an index to a unique key. Check LMDBWrapper class for default keygen")
		pass

	def read(self, key = None, idx = None):
		if key is None and idx is None:
			return None
		key = self.keygen(idx) if key is None else key		
		with self.env.begin() as txn:		
			value = txn.get(key)
		return value

	def write(self, key = None, value = ''):		
		self.idx += 1 if key is None and idx is None else 0
		key = self.keygen(self.idx) if key is None else key		
		# key = self.keygen(idx) if idx is not None else key
		try:	
			with self.env.begin(write = True) as txn:
				txn.put(key, value)								
		except lmdb.MapFullError:
			txn.abort()
			current_map_size = self.env.info()['map_size']
			print 'increasing size of lmdb {} from {} MB to {} MB'.format(self.env.path(), current_map_size / 1e6, 2 * current_map_size / 1e6)
			self.env.set_mapsize(current_map_size * 2)
			self.write(key, value)

	def __del__(self):
		self.env.close()

class LMDBWrapper(LMDBWrapperBase):
	def __init__(self, lmdb_name, init_sample_estimate = 1000, sample_size = 64):
		LMDBWrapperBase.__init__(self, lmdb_name = lmdb_name, init_sample_estimate = init_sample_estimate, sample_size = sample_size)

	def keygen(self, idx):
		return '{:08}'.format(idx).encode('ascii')
