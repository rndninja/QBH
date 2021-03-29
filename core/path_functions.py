import os


# Parameters of the model are globally defined for the trained network
number_of_patches = 20
patch_size = 25
segment_length = 500  # number_of_patches x patch_size
feature_size = 301
number_of_classes = 62
step_notes = 5
SR = 22050
hop_size = 256
RNN = 'GRU'

#########################################################
## GET PATH FUNCTIONS: Functions to return paths
def get_path():
    '''
    Gets the path of the main folder
    :return: path (string)
    '''
    path = os.getcwd()
    path = path[:path.rfind('/')]

    return path

def get_path_to_database():

    database_path = '{0}/Database'.format(get_path())

    return database_path


def get_path_to_dataset_audio():
    audio_path = '{0}/MIDI_database'.format(get_path())
    return audio_path

def get_path_to_pitch_estimates():
    path = '{0}/Database/pitch_estimates_DB'.format(get_path())
    return path

def get_path_to_query():

    query_path = '{0}/Query'.format(get_path())
    return query_path


def get_path_to_pitch_estimations():
    # Wrapper function
    results_path = get_model_output_save_path()

    return results_path

#######################################################