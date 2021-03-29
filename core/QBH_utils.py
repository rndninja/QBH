import matplotlib.pyplot as plt
import path_functions
import statistics
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import librosa
import os
import pickle
import librosa
import pretty_midi

from path_functions import *
from statistics import mode
from falconn import LSHIndex, Queryable, LSHConstructionParameters, DistanceFunction, LSHFamily, StorageHashTable
from scipy.signal import lfilter, savgol_filter, medfilt

# Parameters of the model are globally defined for the trained network
number_of_patches = 20
patch_size = 25
segment_length = 500  # number_of_patches x patch_size
feature_size = 301
number_of_classes = 62
step_notes = 5
SR = 22050
hop_size = 256

def Generate_query_pitch_estimates(track_name):

	audio_fpath = '{0}/{1}.mid'.format(get_path_to_dataset_audio(),track_name)
	midi_data = pretty_midi.PrettyMIDI(audio_fpath)
	pitch_estimates = []

	i=0
	for instrument in midi_data.instruments:
		note_pitches = []
		onset = []
		offset = []

		if not instrument.is_drum:
			i=i+1

			for note in instrument.notes:

				note_pitches.append(note.pitch)
				onset.append(time_to_index(note.start))
				offset.append(time_to_index(note.end))

			pitch_estimates = convert_note_to_pitch_estimates(note_pitches,onset,offset)


	return pitch_estimates[200:1000]

def StoreDB_pitch_estimates(dirlist):

	for track_name in dirlist:
		audio_fpath = '{0}/{1}.mid'.format(get_path_to_dataset_audio(),track_name)
		midi_data = pretty_midi.PrettyMIDI(audio_fpath)

		i=0
		for instrument in midi_data.instruments:
			track_name_new = track_name + str(i)
			note_pitches = []
			onset = []
			offset = []

			if not instrument.is_drum:
				i=i+1

				for note in instrument.notes:

					note_pitches.append(note.pitch)
					onset.append(time_to_index(note.start))
					offset.append(time_to_index(note.end))

				note = [note_pitches,onset,offset]
				note = np.asarray(note)
				np.save('{0}/pitch_estimates_DB/{1}'.format(get_path_to_database(),track_name_new), note)
		
	return


def GenerateDB(track_name, pitch_estimates, pitch_vectorDB, pitchvector_index, vector_length, stride):

	num_of_pitch_vectors = int((pitch_estimates.size - vector_length)/stride)
	pitch_vectorDB = np.zeros((1,vector_length))

	for i in range(num_of_pitch_vectors):
		pitch_vectorDB = np.concatenate((pitch_vectorDB,np.matrix(pitch_estimates[stride*i:stride*i+vector_length])),axis=0)
		pitchvector_index.append([track_name,time(i)])

	pitch_vectorDB = np.asmatrix(pitch_vectorDB)
	Store_DB(pitch_vectorDB,pitchvector_index,"Vector_Wise_Normalised")
	print("{0} added to dataset, current shape:".format(track_name),pitch_vectorDB.shape)

	return pitch_vectorDB, pitchvector_index

def GenerateDB_freq_normalisation(track_name, pitch_estimates ,pitch_vectorDB, pitchvector_index, vector_length, stride):

	num_of_pitch_vectors = int((len(pitch_estimates) - vector_length)/stride)

	for i in range(num_of_pitch_vectors):
		pitch_vector = (freq_normalisation(pitch_estimates[stride*i:stride*i+vector_length]))
		pitch_vectorDB = np.concatenate((pitch_vectorDB,np.matrix(pitch_vector)),axis=0)
		pitchvector_index.append([track_name,time(i)])

	pitch_vectorDB = np.asmatrix(pitch_vectorDB)

	return pitch_vectorDB, pitchvector_index

def GenerateDB_vectorwise_freq_normalisation(track_name, vector_length=20, stride=3):

	pitch_vectorDB = np.zeros((1,vector_length))
	pitchvector_index = [["Null",0]]

	[note_pitches,onset,offset] = np.load('{0}/pitch_estimates_DB/{1}.npy'.format(get_path_to_database(),track_name))
	pitch_estimates = convert_note_to_pitch_estimates(note_pitches,onset,offset)

	pitch_vectorDB, pitchvector_index = GenerateDB_freq_normalisation(track_name,pitch_estimates,pitch_vectorDB,pitchvector_index,vector_length,stride)

	return pitch_vectorDB, pitchvector_index


def GenerateDB_normal(dirlist, model, vector_length=20, stride=3):

	pitch_vectorDB = np.zeros((1,vector_length))
	pitchvector_index = [["Null",0]]

	for track_name in dirlist:

		pitch_estimates = np.load('{0}/pitch_estimates_DB/{1}.npy'.format(get_path_to_database(),track_name))

		pitch_vectorDB, pitchvector_index = GenerateDB(track_name,pitch_estimates,pitch_vectorDB,pitchvector_index,vector_length,stride)

	return pitch_vectorDB, pitchvector_index


def Store_DB(pitch_vectorDB, pitchvector_index, file_name):

	np.save('{0}/{1}_DB'.format(get_path_to_database(),file_name), pitch_vectorDB)

	with open('{0}/{1}_index.data'.format(get_path_to_database(),file_name), 'wb') as filehandle:

		pickle.dump(pitchvector_index, filehandle)

	return

def Store_Periodic_DB(pitch_vectorDB, pitchvector_index, file_name, track_name):

	pitch_vectorDB1, pitchvector_index1 = Retrieve_DB(file_name)

	if(pitchvector_index1[0][0] == 'New'):
		pitch_vectorDB = pitch_vectorDB
		pitchvector_index = pitchvector_index
	else:
		pitch_vectorDB = np.concatenate((pitch_vectorDB1,pitch_vectorDB),axis=0)
		pitchvector_index = pitchvector_index1 + pitchvector_index

	np.save('{0}/{1}_DB'.format(get_path_to_database(),file_name), pitch_vectorDB)

	with open('{0}/{1}_index.data'.format(get_path_to_database(),file_name), 'wb') as filehandle:

		pickle.dump(pitchvector_index, filehandle)

	print("{0} added to dataset, current shape:".format(track_name),pitch_vectorDB.shape)

	return

def Retrieve_DB(file_name):

	pitch_vectorDB = np.load('{0}/{1}_DB.npy'.format(get_path_to_database(),file_name))

	with open('{0}/{1}_index.data'.format(get_path_to_database(),file_name), 'rb') as filehandle:

		pitchvector_index = pickle.load(filehandle)

	return pitch_vectorDB, pitchvector_index


def Query_1(query_track_name, query_obj, Database, pitchvector_index, model, vector_length=300, stride=3):

	pitch_estimates = Generate_query_pitch_estimates(query_track_name)
	pitch_estimates = smoothen_Waveform(pitch_estimates)
	scaled_pitch_estimate_matrix = get_list_of_elongated_pitch_estimates(pitch_estimates)

	candidate_segment_list = []

	for [pitch_estimates,scale] in scaled_pitch_estimate_matrix:

		# print("\n\n\n\n\n\n",len(pitch_estimates))

		query_pitch_matrix = np.zeros((1,vector_length))
		num_of_pitch_vectors = int((len(pitch_estimates)-vector_length)/stride)
		query_times = []

		for i in range(num_of_pitch_vectors):
			query_pitch_matrix = np.concatenate((query_pitch_matrix,np.matrix(pitch_estimates[stride*i:stride*i+vector_length])),axis=0)
			query_times.append(time(i*stride))

		# print("Query from track '{0}'".format(query_track_name))

		# print(num_of_pitch_vectors, scale)

		for i in range(num_of_pitch_vectors):
			query = query_pitch_matrix[i]
			query = np.asarray(query,dtype=np.float64)
			query = (freq_normalisation(query))
			index_list = query_obj.find_k_nearest_neighbors(query[0],1)
			result_list = []
			for j in index_list:
				result_list.append([pitchvector_index[j][0],pitchvector_index[j][1],query_times[i],scale])
				bc = pitchvector_index[j][1]
				candidate_segment_list.append([scale,[pitchvector_index[j][0],(bc-query_times[i]/scale),(bc+(query_times[-1]-query_times[i])/scale)]])

			# print(result_list[0][0],result_list[1][0])

	return candidate_segment_list

def Query_2(query_track_name, query_obj, Database, pitchvector_index, vector_length, stride):

	query_pitch_estimates = Generate_query_pitch_estimates(query_track_name)
	scaled_pitch_estimate_matrix = get_list_of_elongated_pitch_estimates(query_pitch_estimates)
	candidate_segment_list = []

	print("Employing Fuzzy search using LSH")

	for [pitch_estimates,scale] in scaled_pitch_estimate_matrix:

		query_pitch_matrix = np.zeros((1,vector_length))
		num_of_pitch_vectors = int((len(pitch_estimates)-vector_length)/stride)
		query_times = []

		for i in range(num_of_pitch_vectors):
			query_pitch_matrix = np.concatenate((query_pitch_matrix,np.matrix(pitch_estimates[stride*i:stride*i+vector_length])),axis=0)
			query_times.append(time(i*stride))

		for i in range(num_of_pitch_vectors):
			query = query_pitch_matrix[i]
			query = np.asarray(query,dtype=np.float64)
			query = (freq_normalisation(query))
			index_list = query_obj.find_k_nearest_neighbors(query[0],3)
			result_list = []
			for j in index_list:
				result_list.append([pitchvector_index[j][0],pitchvector_index[j][1],query_times[i],scale])
				bc = pitchvector_index[j][1]
				candidate_segment_list.append([scale,[pitchvector_index[j][0],(bc-query_times[i]/scale),(bc+(query_times[-1]-query_times[i])/scale)]])
	print("Candidate segment list generated")

	return candidate_segment_list


def Final_Ranking_1(query_track_name, query, candidate_segment_list, dirlist):

	final_result = []
# 	Result_Dictionary = {}
	# dirlist = ['xml','soo','10-little-indians','2','Train','Kabira_Test','Kabira_Test_High']

# 	for track_name in dirlist:
# 		Result_Dictionary[track_name] = 9999999

	for candidate_segment in candidate_segment_list:
		track_name = candidate_segment[1][0]
		onset = time_to_index(candidate_segment[1][1])
		offset = time_to_index(candidate_segment[1][2])

		if(onset<0):
			onset=0

		if(track_name != "Null"):
			[note_pitches,onsets,offsets] = np.load('{0}/pitch_estimates_DB/{1}.npy'.format(get_path_to_database(),track_name))
			
			pitch_estimates = convert_note_to_pitch_estimates(note_pitches,onsets,offsets)

			if(offset>len(pitch_estimates)):
				offset=len(pitch_estimates)
			
			candidate_melody = pitch_estimates[onset:offset]

			# print("Pitch_Estimates: ",len(pitch_estimates))
			# print("Onset, offset: ",onset,offset)
			# print("candidate_melody: ",len(candidate_melody))
			candidate_melody = freq_normalisation(smoothen_Waveform(candidate_melody))
			query = freq_normalisation(smoothen_Waveform(query))

			score = Recursive_Alignment(query,candidate_melody,[0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],4)
			# final_result.append([track_name,score,candidate_segment[1][1],candidate_segment[1][2]])
			if(-1*score < Result_Dictionary[track_name]):
				Result_Dictionary[track_name] = -1*score

# 	Result_Dictionary = sorted(Result_Dictionary.items(), key = lambda kv:(kv[1], kv[0]))

# 	Hit_Number = 1

# 	for kv_pair in Result_Dictionary:
# 		if(kv_pair[0] == query_track_name):
# 			return Hit_Number

# 		Hit_Number=Hit_Number+1
	final_result.sort(key = lambda final_result: final_result[1])
	
	for i in range(100):
		print(final_result[i][0], "-", final_result[i][1])

	return

def Final_Ranking_2(query_track_name, query_pitch_estimates, candidate_segment_list, dirlist):

	final_result = []
	Result_Dictionary = {}
	EMD_score_Dictionary = {}
	EMD_score_list = []
	# query_pitch_estimates = convert_to_transitions(query_pitch_estimates)

	for track_name in dirlist:
		Result_Dictionary[track_name] = 0

	print("Calculating EMD scores for {0}".format(query_track_name))

	for candidate_segment in candidate_segment_list:
		track_name = candidate_segment[1][0]
		onset = time_to_index(candidate_segment[1][1])
		offset = time_to_index(candidate_segment[1][2])

		if(onset<0):
			onset=0

		if(track_name != "Null"):
			[note_pitches,onsets,offsets] = np.load('{0}/pitch_estimates_DB/{1}.npy'.format(get_path_to_database(),track_name))
			pitch_estimates = convert_note_to_pitch_estimates(note_pitches,onsets,offsets)

			# pitch_estimates = convert_to_transitions(pitch_estimates)

			if(offset>len(pitch_estimates)):
				offset=len(pitch_estimates)
			
			candidate_melody = pitch_estimates[onset:offset]
			candidate_melody = freq_normalisation(smoothen_Waveform(candidate_melody))
			query_pitch_estimates = freq_normalisation(smoothen_Waveform(query_pitch_estimates))

			EMD_score = EMD(query_pitch_estimates,pitch_estimates)

			EMD_score_list.append([EMD_score,track_name,onset,offset])

	EMD_score_list.sort(key = lambda x: x[0])
	shortlisted_EMD_score_list = EMD_score_list[:500]
	euclidean_norm = lambda x, y: np.abs(x - y)

	print("Calculating Final scores for {0}".format(query_track_name))

	for i in range(100):
		track_name = EMD_score_list[i][1]
		onset = EMD_score_list[i][2]
		offset = EMD_score_list[i][3]

		pitch_estimates = np.load('{0}/pitch_estimates_DB/{1}.npy'.format(get_path_to_database(),track_name))
		pitch_estimates = pitch_estimates[onset:offset]
		# pitch_estimates = convert_to_transitions(pitch_estimates)
		dtw_score,x,y,z = dtw(query_pitch_estimates,pitch_estimates,dist=euclidean_norm)
		final_score = 0.6*dtw_score + 0.4*EMD_score_list[i][0]

		final_result.append([final_score,track_name])

	final_result.sort(key = lambda x: x[0])

	return final_result[:20]


def note_sequence(MIDI_sequence, epsilon=0.1): #Pitch_Estimates to Note_Sequence #Done
	onset = []
	offset = []
	note_pitch = []
	note_pitch.append(MIDI_sequence[0])


	onset.append(0)
	prev_pitch = MIDI_sequence[0]

	for pitch_index in range(1,len(MIDI_sequence)):
		pitch = MIDI_sequence[pitch_index]
		
		if abs(pitch - prev_pitch) > epsilon:
			onset.append(pitch_index)
			offset.append(pitch_index)
			note_pitch.append(MIDI_sequence[pitch_index])

		prev_pitch = pitch

	offset.append(len(MIDI_sequence))

	return onset, offset, note_pitch


def convert_note_to_pitch_estimates(note_pitch, onset, offset): #Done
	
	MIDI_sequence = []

	for i in range(len(note_pitch)):
		# print(int(onset[i]),int(offset[i]))

		MIDI_sequence[onset[i]:offset[i]] = [note_pitch[i]]*(offset[i]-onset[i])

	return MIDI_sequence


def get_list_of_elongated_pitch_estimates(pitch_estimates, size_n=15, epsilon=0.1): #Done

	onset, offset, note_pitch = note_sequence(pitch_estimates, epsilon)
	onset = np.asarray(onset)
	offset = np.asarray(offset)

	list_of_elongated_MIDI_sequence = []

	for i in range(size_n+1):

		w = 0.5 + 0.1*i

		onset_scaled = onset*w
		offset_scaled = offset*w

		onset_scaled = np.asarray(onset_scaled,dtype=int)
		offset_scaled = np.asarray(offset_scaled,dtype=int) 

		new_pitch_estimates = convert_note_to_pitch_estimates(note_pitch,onset_scaled,offset_scaled)
		list_of_elongated_MIDI_sequence.append([new_pitch_estimates,w])

	return list_of_elongated_MIDI_sequence

def freq_normalisation(pitch_estimates): #Done

	onset, offset, note_pitch = note_sequence(pitch_estimates)
	
	onset_times = [time(i) for i in onset]
	offset_times = [time(i) for i in offset]

	onset_times = np.asarray(onset_times)
	offset_times = np.asarray(offset_times)

	weights = offset_times - onset_times

	Average = np.sum(np.multiply(note_pitch,weights))/offset_times[-1]

	note_pitch = note_pitch - Average

	pitch_estimates = convert_note_to_pitch_estimates(note_pitch,onset,offset)

	return pitch_estimates

def time_scale(pitch_estimates, scale, end_time):

	onset, offset, note_pitch = note_sequence(pitch_estimates)
	onset = np.asarray(onset)
	offset = np.asarray(offset)

	onset_scaled = onset*scale
	offset_scaled = offset*scale

	onset_scaled = np.asarray(onset_scaled,dtype=int)
	offset_scaled = np.asarray(offset_scaled,dtype=int)
	offset_scaled[-1] = end_time

	pitch_estimates = convert_note_to_pitch_estimates(note_pitch,onset_scaled,offset_scaled)

	return pitch_estimates

def max_freq_normalisation(pitch_estimates):

	pitch_estimates = np.asarray(pitch_estimates)
	pitch_estimates = pitch_estimates/np.max(pitch_estimates)

	return pitch_estimates

def time(pitch_index):

	return (pitch_index*np.float(hop_size)/SR)

def time_to_index(time):

	return int(time*np.float(SR)/np.float(hop_size))

def Linear_Scaling(query_pitch_estimates,candidate_pitch_estimates):

	if(len(candidate_pitch_estimates)==0):
		return -9999999

	scale = len(query_pitch_estimates)/len(candidate_pitch_estimates)

	candidate_pitch_estimates = time_scale(candidate_pitch_estimates,scale,len(query_pitch_estimates))

	query_pitch_estimates = np.asarray(query_pitch_estimates)
	candidate_pitch_estimates = np.asarray(candidate_pitch_estimates)

	diff = query_pitch_estimates - candidate_pitch_estimates
	Dist = diff.dot(diff.T)

	return -Dist

def Recursive_Alignment(query, candidate_melody, scale_pairs, D):

	onset, offset, note_pitch = note_sequence(candidate_melody)
	durations = np.asarray(offset)-np.asarray(onset)

	i=0
	j=int(durations.size/2)
	maxScore = -99999999

	sc = np.sum(durations[:j])/np.sum(durations)

	N1 = note_pitch[:j]
	N2 = note_pitch[j:]

	for scale in scale_pairs:

		k = int(scale*sc*len(query))
		Q1 = query[:k]
		Q2 = query[k:]

		score = Linear_Scaling(Q1,N1) + Linear_Scaling(Q2,N2)

		if(score > maxScore):
			maxScore = score
			i = k
	if(D):
		return maxScore
	else:
		Q1 = query[:i]
		Q2 = query[i:]
		return Recursive_Alignment(Q1,N1,scale_pairs,D-1) + Recursive_Alignment(Q2,N2,scale_pairs,D-1)


def plot_Waveform(model, track_name, vector_length=20, stride=3, isfirsttime = False):

	pitch_estimates = Generate_pitch_vector(track_name,model,isfirsttime)
	num_of_pitch_vectors = int((len(pitch_estimates) - vector_length)/stride)

	times = np.arange(len(pitch_estimates)) * np.float(hop_size)/SR
	plt.plot(times, pitch_estimates)
	plt.show()

	return

def plot_Waveform_Database(model, dirlist, vector_length=20, stride=3, isfirsttime = False):

	for track_name in dirlist:

		plot_Waveform(model, track_name, vector_length, stride, isfirsttime)

	return

def smoothen_Waveform(pitch_estimates, n=51):

	Noise_Free_Waveform = medfilt(pitch_estimates,n)

	times = np.arange(len(pitch_estimates)) * np.float(hop_size)/SR
	# plt.plot(times,Noise_Free_Waveform)
	# plt.show()

	return Noise_Free_Waveform


def LSH_initialization(Feature_Dimension, k, l, threads, lsh_family, distance_function, storage_hash_table):

	params = LSHConstructionParameters()
	params.dimension = Feature_Dimension
	params.k = k
	params.l = l
	params.num_setup_threads = threads
	params.lsh_family = lsh_family
	params.distance_function = distance_function
	params.storage_hash_table = storage_hash_table

	return params
