"""
Python structures for smart-seq data of 2018
"""

from utilities.smart_seq_dataset import *
import numpy as np
import random
import operator


class RNAseq_Dataset:
    def __init__(self, cells, cells_information_list, gene_names):
        self.cells = cells
        if isinstance(cells_information_list, Cells_inforamation):
            self.cells_information_list = cells_information_list
        else:
            self.cells_information_list = Cells_inforamation(cells_information_list)
        self.gene_names = gene_names
        self.length = len(self.cells_information_list)

    def __len__(self):
        return len(self.cells_information_list)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.cells[item], self.cells_information_list[item]
        if isinstance(item, slice):
            return RNAseq_Dataset(self.cells[item], self.cells_information_list[item], self.gene_names)
        if isinstance(item, list):
            # identify if we are dealing with binary indexes or explicit indexes.
            if sum([(ii == 0 or ii == 1) for ii in item]) == len(item):
                # converts to explicit indexes.s
                item = [i for i in range(len(self)) if item[i]]
            return RNAseq_Dataset(self.cells[item, :], self.cells_information_list[item], self.gene_names)

    def get_post_patients_sub_dataset(self):
        indexes = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if 'Post' in p]
        return self[indexes]

    def get_baseline_patients_sub_dataset(self):
        indexes = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if 'Pre' in p]
        return self[indexes]

    def expression_of_genes(self):
        """
        :param cells: PKL format
        :param gene_names: list of gene names in the order shown in the cells.
        :return: sorted avg gene expressions and gene names sorted by avg expression.
        """
        average_val_genes = np.mean(self.cells, axis=0)
        indexes_of_sorted_expression_of_genes = np.argsort(average_val_genes)
        gene_sorted_by_expression = operator.itemgetter(*indexes_of_sorted_expression_of_genes)(self.gene_names)
        sorted_average_expression_genes = average_val_genes[indexes_of_sorted_expression_of_genes]
        return sorted_average_expression_genes, gene_sorted_by_expression

    def get_high_expressed_genes(self):
        """
        Return (only the indexes) of the highest expression genes. TODO: make the genes amount changeable.
        :param cells: pkl format
        :param gene_names: when given, the function return genes names with highest expression also.
        :return: indexes of highest expressed gene and sub-dataset contains cells with the highest expressed genes.
        """
        activated_genes = sum(np.sum(self.cells, axis=0) != 0)
        average_val_genes = np.mean(self.cells, axis=0)
        expression_histogram = np.histogram(average_val_genes)
        expression_histogram_df = np.concatenate((np.expand_dims(expression_histogram[0][:10], axis=0),
                                                  np.expand_dims(expression_histogram[1][:10], axis=0)))
        amount_high_expressed_genes = min(activated_genes * 0.05, sum(expression_histogram[0][-2:]))
        high_expressed_percentage = amount_high_expressed_genes / activated_genes
        indices_of_high_expressed_genes = np.argsort(average_val_genes)[-3:]
        print(f'High expressed genes percent {high_expressed_percentage}')
        return indices_of_high_expressed_genes, \
               RNAseq_Dataset(self.cells[:, indices_of_high_expressed_genes],
                              self.cells_information_list,
                              operator.itemgetter(*indices_of_high_expressed_genes)(self.gene_names))

    def filter_cells_by_supervised_classification(self, required_cell_type="T cells"):
        """
        patients_information contains 'supervised classification' field, which is the classification of the cell
        defined by gene expression in 80% of the cells, and the remaining 20% were done by manual process.
        the function filters cells by cell-type was done by this classification.
        :param required_cell_type: name of the desired cell-type.
        :return: reduced RNAseq dataset contains cells of the desired cell-type
        """
        indexes_list = self.cells_information_list.get_cells_belong_to_cells_type(required_cell_type)
        return RNAseq_Dataset(self.cells[indexes_list, :], self.cells_information_list[indexes_list], self.gene_names)

    def filter_genes_by_variance(self, required_variance, in_place=True):
        big_variance_genes = np.var(self.cells, axis=0) > required_variance
        filtered_cells = self.cells[:, big_variance_genes]
        filtered_genes = [self.gene_names[i] for i in range(len(self.gene_names)) if big_variance_genes[i]]
        if in_place:
            self.cells = filtered_cells
            self.gene_names = filtered_genes
            print(f"Dataset was cleared from genes with variance of less than {required_variance}")
        return RNAseq_Dataset(filtered_cells, self.cells_information_list, filtered_genes)

    def train_test_split(self, test_size=0.2, shuffle=True, stratify=True):
        """
        :param test_size: in proportion to the number of patients (not proportion to nubber of cells)
        :param shuffle:
        :param random_state:
        :param stratify: Always, equal ration of responder to non-responder between test-set and train-set.
        :return: x_train, x_test, and corresponding indexes
        """

        responders = list(set([p.patient_details for p in self.cells_information_list if p.response_label]))
        non_responders = list(set([p.patient_details for p in self.cells_information_list if not p.response_label]))
        if shuffle:
            random.shuffle(responders)
            random.shuffle(non_responders)

        train_responders = responders[:int(len(responders) * (1 - test_size))]
        test_responders = responders[int(len(responders) * (1 - test_size)):]
        train_non_responders = non_responders[:int(len(non_responders) * (1 - test_size))]
        test_non_responders = non_responders[int(len(non_responders) * (1 - test_size)):]

        train_patients = train_responders + train_non_responders
        test_patients = test_responders + test_non_responders

        train_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p in train_patients]
        test_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p in test_patients]

        x_train = self[train_idxs]
        x_test = self[test_idxs]

        return x_train, x_test, train_idxs, test_idxs

    def split_by_patient_names(self, patients_names):
        """
        return two sub-datasets. one, contains required patients and the seconds contains the other patients.
        :param patients_names: list of all the names of patients wanted to be in the first sub-dataset.
        :return: rnq_seq_dataset instances of the required and not required dataset.
        """
        required_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p in patients_names]
        not_required_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p not in patients_names]
        return self[required_idxs], self[not_required_idxs]

    def k_fold_cross_validation(self, k, shuffle=True, verbose=False):
        """
        Creates K-fold_cross_validation iterator which can be used for training.
        :param k: number of folds.
        :param shuffle: if you wand to shuffle the data
        :return: k_validation iterator instance.
        """
        responders = list(set([p.patient_details for p in self.cells_information_list if p.response_label]))
        non_responders = list(set([p.patient_details for p in self.cells_information_list if not p.response_label]))
        if shuffle:
            random.shuffle(responders)
            random.shuffle(non_responders)

        k_validation = K_fold_validation(self, responders, non_responders, k, verbose=verbose)
        return k_validation

    def get_all_patients_names(self):
        names = list(set([p for p in self.cells_information_list['patient_details']]))
        return names

    def __add__(self, other):
        if other.gene_names != self.gene_names:
            raise Exception("different genes")
        other_new_idxs = [idx for idx, p in enumerate(other.cells_information_list['cell_id']) if p not in self.cells_information_list['cell_id']]
        new_rna_seq = other[other_new_idxs]
        patients = self.cells_information_list.single_cell_list + new_rna_seq.cells_information_list.single_cell_list
        gene_names = self.gene_names.copy()
        cells = np.concatenate((self.cells, other.cells))
        return RNAseq_Dataset(cells, patients, gene_names)

    def __sub__(self, other):
        if other.gene_names != self.gene_names:
            raise Exception("different genes")
        idxs = [idx for idx, p in enumerate(self.cells_information_list['cell_id']) if p not in other.cells_information_list['cell_id']]
        return self[idxs]


class Cells_inforamation:
    def __init__(self, cells_information_structure=None):
        self.single_cell_list = []
        if cells_information_structure:
            if isinstance(cells_information_structure[0], dict):
                if cells_information_structure:
                    for p_dict in cells_information_structure:
                        self.single_cell_list.append(Single_cell_sample(p_dict['sample index'],
                                                                        p_dict['cell id'],
                                                                        p_dict['patient details'],
                                                                        p_dict['response'],
                                                                        p_dict['treatment'],
                                                                        p_dict.get('response label', None),
                                                                        p_dict.get('general 11 cluster', None),
                                                                        p_dict.get('supervised classification',
                                                                                      None),
                                                                        p_dict.get('T-cell 2 cluster', None),
                                                                        p_dict.get('T-cell 6 cluster', None)))
            elif isinstance(cells_information_structure[1], Single_cell_sample):
                self.single_cell_list = cells_information_structure
        self.length = len(self.single_cell_list)

    def __len__(self):
        return len(self.single_cell_list)

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.single_cell_list[item]
        if isinstance(item, slice):
            # Get the start, stop, and step from the slice
            return Cells_inforamation([self.single_cell_list[ii] for ii in range(*item.indices(len(self.single_cell_list)))])
        if isinstance(item, str):
            return [p.__getattribute__(item) for p in self.single_cell_list]
        if isinstance(item, list):
            if not len(item):
                return Cells_inforamation()
            if sum([(ii == 0 or ii == 1) for ii in item]) == len(item):
                item = [i for i in range(len(self)) if item[i]]
            return Cells_inforamation([self.single_cell_list[i] for i in item])

    def __getattr__(self, item):
        return [p.__getattribute__(item) for p in self.single_cell_list]

    def get_cells_belong_to_cells_type(self, cell_type):
        return [p.belongs_to_cell_type(cell_type) for p in self.single_cell_list]

    def slice_by_binary_indexes(self, binary_indexes):
        self.single_cell_list = [self.single_cell_list[i] for i in range(len(self.single_cell_list)) if binary_indexes[i]]

    def get_patients_names(self):
        return list(set([p.patient_details for p in self.single_cell_list]))


class Single_cell_sample:
    def __init__(self, sample_index,
                 cell_id,
                 patient_details,
                 response,
                 treatment,
                 response_label,
                 general_11_cluster,
                 supervised_classification,
                 t_cell_2_cluster,
                 t_cell_6_cluster):
        self.sample_index = sample_index
        self.cell_id = cell_id
        self.patient_details = patient_details
        self.response = response
        self.treatment = treatment
        self.response_label = response_label
        self.general_11_cluster = general_11_cluster
        self.supervised = supervised_classification
        self.t_cell_2_cluster = t_cell_2_cluster
        self.t_cell_6_cluster = t_cell_6_cluster

    def belongs_to_cell_type(self, cell_type):
        return cell_type in self.supervised


class K_fold_validation:
    """Iterator that counts upward forever."""

    def __init__(self, dataset, responders, non_responders, k, verbose=False):
        self.idx = 0
        self.k = k
        self.dataset = dataset
        self.responders = responders
        self.non_responders = non_responders
        self.verbose = verbose
        self.cells = dataset.cells
        self.cells_information_list = dataset.cells_information_list
        self.rfl = int(len(responders) / k)    # responder fold length
        self.nrfl = int(len(non_responders) / k)   # non-responder fold length

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < self.k:
            idx = self.idx
            self.idx += 1

            responders = self.responders
            non_responders = self.non_responders

            if idx < self.k - 1:
                train_responders = responders[: self.rfl * idx] + \
                                   responders[self.rfl * (idx + 1):]
                test_responders = responders[self.rfl * idx: self.rfl * (idx + 1)]

                train_non_responders = non_responders[: self.nrfl * idx] + \
                                       non_responders[self.nrfl * (idx + 1):]
                test_non_responders = non_responders[self.nrfl * idx: self.nrfl * (idx + 1)]

            else:
                train_responders = responders[:self.rfl * idx]
                test_responders = responders[self.rfl * idx:]

                train_non_responders = non_responders[:self.nrfl * idx]
                test_non_responders = non_responders[self.nrfl * idx:]

            train_patients = train_responders + train_non_responders
            test_patients = test_responders + test_non_responders

            # Prints
            if self.verbose:
                print(f"K-Fold number {self.idx}")
                print(f'    train patients (k-1 folds): {train_patients}')
                print(f'validation patients (one fold): {test_patients}')

            train_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p in train_patients]
            test_idxs = [idx for idx, p in enumerate(self.cells_information_list['patient_details']) if p in test_patients]

            x_train = self.cells[train_idxs]
            x_test = self.cells[test_idxs]
            y_train = np.array([p.response_label for p in self.cells_information_list[train_idxs]])
            y_test = np.array([p.response_label for p in self.cells_information_list[test_idxs]])

            return x_train, x_test, y_train, y_test, train_idxs, test_idxs
        else:
            raise StopIteration



