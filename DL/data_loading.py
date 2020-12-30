import pickle

"""
2018, Smart_seq scRNAseqBuilding Data. extract_smart_seq_data_from_pickle.
2020, droplet scRNAseqBuilding Data. extract_droplet_data_from_pickle.
"""

def extract_smart_seq_data_from_pickle(pickle_path):
    """
    Retrieves 2018 smart_seq data from PC located in PICKLE_PATH.
    :return: cells_form, gene_names, patients_information
    """
    cells_form, gene_names, patients_information = pickle.load(open(pickle_path, "rb"))
    return cells_form, gene_names, patients_information


