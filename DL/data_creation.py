import pandas
import operator
import math
from DL.supervised_classification import *

CD45_CELLS_INFORMATION_PATH = r'..\Data\source data\GSE120575_patient_ID_single_cells.txt'
GENERAL_11_CELL_CLUSTERS_PATH = r'C:\Users\itay\Desktop\Technion studies\Keren Laboratory\research\articles\Tables and files article5\Table S1 - A Summary of Data Related to All Single-Cells Analysis, Related to Figure 1.xlsx'
TCELLS_2_CLUSTERS_PATH = r'C:\Users\itay\Desktop\Technion studies\Keren Laboratory\research\articles\Tables and files article5\additional files\cluster_annot_cd8_cells_2_clusters.xlsx'
TCELLS_6_CLUSTERS_PATH = r'C:\Users\itay\Desktop\Technion studies\Keren Laboratory\research\articles\Tables and files article5\additional files\cluster_annot_cd8_cells_6_clusters.xlsx'
ROW_CD45_CELLS_DATA_PATH = r'..\Data\source data\GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt'
PROTEIN_CODING_FILE = r'..\Data\gene_ens_map.xlsx'
PICKLE_PATH = r'..\DATA\18.10.20_cells_all_protein_conding_genes(withoutFilterByVariance).p'
CELLS_RANGE = [1, 16291]  # should be an integer (indicates k first cells) or a range of cells (1-16291)


"""
Smart_seq scRNAseqBuilding 2018 Data. PKL files contain all information need to further analysis.
"""


def extract_data_from_pc(cells_range, reduced_amount_of_cells=True):
    cells_range_idx = range(cells_range[0], cells_range[1])
    cells_range_shift_1_idx = [0]+list(range(cells_range[0]+1, cells_range[1] + 1))

    # Extract data files
    with open(CD45_CELLS_INFORMATION_PATH, 'r') as file:
        CD45_cells_information = file.readlines()
    with open(ROW_CD45_CELLS_DATA_PATH, 'r') as file:
        row_CD45_cells_data = file.readlines()

    # Txt type to list type. Drop unimportant columns of patients details:
    patients_information = [patient.split('\t')[:7] for patient in CD45_cells_information[20:-38]]
    num_of_cells = len(cells_range_idx)

    if not reduced_amount_of_cells:
        # Format after rearrangement: (Gene name, list of cell-value)
        cd45_cells = [gene.split('\t')[:-1] for gene in row_CD45_cells_data[2:]]
    else:
        patients_information = operator.itemgetter(*cells_range_idx)(patients_information)
        cd45_cells = [operator.itemgetter(*cells_range_shift_1_idx)(gene.split('\t')) for gene in
                      row_CD45_cells_data[2:]]

    # Reformat and filter by genes that responsible for protein coding.
    cd45_cells = [(gene[0], list(map(float, gene[1:]))) for gene in cd45_cells]
    gene_names = [g[0] for g in cd45_cells]
    # Cells form --> index 1: [(cell 1, gen0) ... (cell 1, genk)]; index 2: [(cell 2, gen0) ... (cell 2, genk)]
    cells_form = np.array([gene[1] for gene in cd45_cells]).T

    # prints
    avg_num_of_activated_gene = np.sum(cells_form != 0) / num_of_cells
    print(f"Number of genes: {len(cells_form[0])}")
    print(f"Number of cells: {num_of_cells}")
    print(f"Average number of activated genes in cell: {avg_num_of_activated_gene}")

    return cells_form, gene_names, patients_information


def save_to_pickle(cells_form, gene_names, patients_information):
    pickle.dump((cells_form, gene_names, patients_information), open(PICKLE_PATH, "wb"))


def save_smaller_file(path, size):
    """
    NOT IN USE.
    :param path:
    :param size:
    :return:
    """
    with open(ROW_CD45_CELLS_DATA_PATH, 'r') as file:
        row_CD45_cells_data = file.readlines()

    # Extracts and rearranges the Data:
    smaller_lines = [' '.join(gene.split('\t')[:size]) for gene in row_CD45_cells_data]

    with open(path, "w+") as file:
        file.writelines(smaller_lines)


def keeps_protein_coding_only(cd45_cells, gene_names):
    df = pandas.read_excel(PROTEIN_CODING_FILE)
    kept_genes_names = [gene[0] for gene in df[df.lincRNA == 'protein_coding'][['MIR1302-11']].values]
    indices_of_protein_coding = [i for i in range(len(gene_names)) if gene_names[i] in kept_genes_names]
    cells = cd45_cells[:, indices_of_protein_coding]
    gene_names = operator.itemgetter(*indices_of_protein_coding)(gene_names)
    return cells, gene_names


def filter_genes_by_variance(cells, gene_names, required_variance=6):
    big_variance_genes = np.var(cells, axis=0) > required_variance
    filtered_cells = cells[:, big_variance_genes]
    filtered_genes = [gene_names[i] for i in range(len(gene_names)) if big_variance_genes[i]]
    return filtered_cells, filtered_genes


def extract_11_general_clusters(patients_information, cells_range):
    xls = pd.ExcelFile(GENERAL_11_CELL_CLUSTERS_PATH)
    df = pd.read_excel(xls, 'Cluster annotation-Fig1B-C')
    match = df.values
    general_11_clusters = operator.itemgetter(*range(cells_range[0], cells_range[1]))(match.tolist())
    for idx, patient in enumerate(patients_information):
        patient['general 11 cluster'] = general_11_clusters[idx][1]

    return patients_information


def extract_Tcell_2_clusters(patients_information):
    xls = pd.ExcelFile(TCELLS_2_CLUSTERS_PATH)
    df = pd.read_excel(xls)
    match = df.values
    match_dict = {k[0]: k[1] for k in match}
    for patient in patients_information:
        val = match_dict.get(patient['cell id'], None)
        if val:
            patient['T-cell 2 cluster'] = val
        else:
            patient['T-cell 2 cluster'] = None

    return patients_information


def extract_Tcell_6_clusters(patients_information):
    xls = pd.ExcelFile(TCELLS_6_CLUSTERS_PATH)
    df = pd.read_excel(xls)
    match = df.values
    match_dict = {k[0]: k[1] for k in match}
    for patient in patients_information:
        val = match_dict.get(patient['cell id'], None)
        if val:
            patient['T-cell 6 cluster'] = val
        else:
            patient['T-cell 6 cluster'] = None

    return patients_information


def divide_to_ranges():
    max_segment_length = 2000
    if type(CELLS_RANGE) is int:
        length = CELLS_RANGE
        cells_ranges = [[0, CELLS_RANGE]]
        start_point = 0
    else:
        length = CELLS_RANGE[1] - CELLS_RANGE[0] + 1
        cells_ranges = [[CELLS_RANGE[0] - 1, CELLS_RANGE[1]]]
        start_point = CELLS_RANGE[0]-1

    if length > max_segment_length:
        cells_ranges = [[max_segment_length * i + start_point, min(max_segment_length * (i + 1) - 1, length - 1) + start_point+1]
                        for i in range(math.ceil(length / max_segment_length))]
    return cells_ranges


def keep_relevant_genes(cells, relevant_genes, gene_names):
    indices = [i for i in range(len(gene_names)) if gene_names[i] in relevant_genes]
    return cells[:, indices]


if __name__ == '__main__':
    # TODO: dropping variance smaller than 6 based on only 2k first cells. should be fixed.
    cells_ranges = divide_to_ranges()
    all_cells = None
    all_patients_informations = []
    print(f'ranges {cells_ranges}')
    for idx, cells_range in enumerate(cells_ranges):
        print(f'current range {cells_range}')
        cells, gene_names, patients_information = extract_data_from_pc(cells_range)
        if idx == 0:
            cells, relevant_genes = keeps_protein_coding_only(cells, gene_names)
            # cells, relevant_genes = filter_genes_by_variance(cells, relevant_genes) # TODO: variance not in use.
        else:
            cells = keep_relevant_genes(cells, relevant_genes, gene_names)

        patients_information = [{'sample index': p[0],
                                 'cell id': p[1],
                                 'patient details': p[4],
                                 'response': p[5],
                                 'treatment': p[6],
                                 'response label': (1 if p[5] == "Responder" else 0)}
                                for p in patients_information]
        extract_11_general_clusters(patients_information, cells_range)

        all_cells = np.concatenate((all_cells, cells)) if idx else cells
        all_patients_informations += patients_information

    extract_Tcell_2_clusters(all_patients_informations)
    extract_Tcell_6_clusters(all_patients_informations)
    add_supervised_cell_types_to_patients(all_patients_informations)
    save_to_pickle(all_cells, relevant_genes, all_patients_informations)
