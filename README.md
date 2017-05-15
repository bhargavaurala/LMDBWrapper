LMDBWrapper allows you read and write key, value pairs into an [LMDB](https://symas.com/lightning-memory-mapped-database/) persistent storage. This project was written to provide a simple handle to write into a LMDB by working around the fixed map size during initialization. It also allows you to access and write using an array-like index as well as traditional key, value.
The LMDBWrapperCaffe allows you to access and write by directly passing an image as numpy array as the value.

Refer LMDBWrapperCaffeTest.py and LMDBWrapperTest.py for usage examples. Please refer utils for utility functions like splitting a database into train and validation sets, standard preprocessing and viewing an existing database.

Dependencies:
1) lmdb (pip install lmdb)
2) numpy (pip install numpy)

If using LMDBWrapperCaffe
1) [caffe](http://caffe.berkeleyvision.org/installation.html)

