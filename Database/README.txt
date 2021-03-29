This folder stores the database of melody fragments or pitch vectors.

There are stored two files per database:
	1) <filename>_DB.npy :- a 2-D numpy matrix with rows as pitch vectors
	2) <filename>_index,data :- This file contains index to the pitch vectors. It contains the name of the song the pitch vector belongs to and the onset time.
