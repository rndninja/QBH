import os
import numpy as np


from QBH_utils import *
from path_functions import *
from falconn import LSHIndex, Queryable, LSHConstructionParameters, DistanceFunction, LSHFamily, StorageHashTable



if __name__ == '__main__':

	vector_length = 300
	stride = 3

#################################################################################################################################################################
#Generate Dirlist for MIDI_database and store all the pitch countours in "/Database/pitch_estimates"
# Pitch estimates need to be stored just once
#################################################################################################################################################################

	# dirlist = []

	# for track_name in os.listdir(get_path_to_dataset_audio()):
	# 	dirlist.append(track_name[:track_name.rfind('.')])
	# 	print(track_name)

	# StoreDB_pitch_estimates(dirlist)
	# print("Pitch estimates stored successfully!\n\n\n\n")

#################################################################################################################################################################
#################################################################################################################################################################

	

################################################################################################################################################################
# Retrieve pitch contours stored earlier and generate a database of pitch vectors which are 3 second long.
# The database is stored as a 2D Numpy Matrix and the Index is stored as a list
################################################################################################################################################################


	dirlist = []

	for track_name in os.listdir(get_path_to_pitch_estimates()):
		dirlist.append(track_name[:track_name.rfind('.')])


	pitch_vectorDB = np.zeros((1,vector_length))
	pitchvector_index = [["New",0]]

	np.save('{0}/Database_Name_DB'.format(get_path_to_database()), pitch_vectorDB)

	with open('{0}/Database_Name_index.data'.format(get_path_to_database()),'wb') as filehandle:
		pickle.dump(pitchvector_index, filehandle)

	for track_name in dirlist:
		pitch_vectorDB, pitchvector_index = GenerateDB_vectorwise_freq_normalisation(track_name,vector_length)
		Store_Periodic_DB(pitch_vectorDB,pitchvector_index,"Database_Name",track_name)

#################################################################################################################################################################
#################################################################################################################################################################


#################################################################################################################################################################
# Retrieve the pitch vector database and index stored earlier and Initialise LSH objects for querying
#################################################################################################################################################################
	
	#define an object of LSHConstructionParameters : params
	# params = LSH_initialization(vector_length,3,1,3, LSHFamily.Hyperplane, DistanceFunction.EuclideanSquared, StorageHashTable.LinearProbingHashTable)


	# #define an object of LSHIndex : a
	# a = LSHIndex(params)

	# # Setup LSH Database for further search
	# Database, Index = Retrieve_DB("vaishnav_song_Database")
	# print(len(Database),len(Index))
	# a.setup(Database)

	# #returns a "queryable"-type instance
	# res = a.construct_query_object()

# #################################################################################################################################################################
# #################################################################################################################################################################


# #################################################################################################################################################################
# # Generate a dirlist for query from "/QBH/Query" folder and one by one query it.
# # This section uses LSH for fuzzy search and Linear Scaling and Note Based Recursive Alignment for accurate search
# #################################################################################################################################################################

	# query_dirlist = []

	# for track_name in os.listdir(get_path_to_query()):
	# 	query_dirlist.append(track_name.split('_')[0])


	# for query in query_dirlist:
	# 	print(query,":")
	# 	candidate_segment_list = Query_1(query,res,a,Index,model,vector_length,stride)
	# 	print(Final_Ranking_1(query, query_pitch_estimates,candidate_segment_list,dirlist))

# #################################################################################################################################################################
# #################################################################################################################################################################


# #################################################################################################################################################################
# # This section uses LSH for fuzzy search and Earth Mover's Distance and DTW for accurate search
# #################################################################################################################################################################

# 	query_dirlist = []

# 	for track_name in os.listdir(get_path_to_query()):
# 		query_dirlist.append(track_name.split('_')[0])


# 	for query in query_dirlist:
# 		print(query,":")
# 		candidate_segment_list = Query_2(query,res,a,Index,model,vector_length,stride)
# 		print(Final_Ranking_2(query, query_pitch_estimates,candidate_segment_list,dirlist))

#################################################################################################################################################################
#################################################################################################################################################################


