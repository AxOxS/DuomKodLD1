import os

# Text files configuration
TEXTS = {
    "Lithuanian Literary": "texts/lt_grozinis.txt",
    "Lithuanian Scientific": "texts/lt_mokslinis.txt", 
    "Spanish Literary": "texts/es_grozinis.txt",
    "Spanish Scientific": "texts/es_mokslinis.txt"
}

# Results folder
RESULT_FOLDER = "results"

# Optimal length analysis settings
OPTIMAL_LENGTH_SETTINGS = {
    'min_length': 500,
    'max_length': 35000,
    'step': 500
}

# Markov analysis settings
MARKOV_SETTINGS = {
    'most_frequent_symbols_count': 10
}


def make_result_folder():
    """Creates results folder if it doesn't exist"""
    os.makedirs(RESULT_FOLDER, exist_ok=True)


def get_work_folder():
    """Sets working directory to project root folder"""
    project_folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_folder)
    return project_folder
