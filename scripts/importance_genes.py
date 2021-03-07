import pickle
import pandas as pd
import os

from Models.feature_explorer import Feature_Explorer
from scripts.experiments_enviroment import build_datasets
from utilities.general_helpers import load_yml

CONFIG_PATH = r'..\cfg\feature_importance_cfg.yaml'

EXPERIMENT_NAME, EXPERIMENTS_FOLDER, config = load_yml(CONFIG_PATH)


def get_respondness(gene, responders_size, non_responders_size, train_dataset, test_dataset):
    """
    The function calculates the responders/non-responders expression value of gene in both train and test sets.
    It returns the average expression level of each gene for responders and non-responders.
    :param gene: the gene to calculate the expression level.
    :param responders_size: size of the responders group.
    :param non_responders_size: size of the non-responders group.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :return: average level expression for responders and non-responders groups.
    """
    responder_test_mean = test_dataset.cells[:responders_size, test_dataset.gene_names.index(gene)].mean()
    responder_train_mean = train_dataset.cells[:responders_size, train_dataset.gene_names.index(gene)].mean()
    non_responder_test_mean = test_dataset.cells[-non_responders_size:, test_dataset.gene_names.index(gene)].mean()
    non_responder_train_mean = train_dataset.cells[-non_responders_size:, train_dataset.gene_names.index(gene)].mean()

    return (responder_train_mean + responder_test_mean) / 2, (non_responder_train_mean + non_responder_test_mean) / 2



def main(dataset_config, xgboost_config):
    model_path = xgboost_config['model_path']
    K = xgboost_config['K']

    experiment_path = os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT_NAME)

    # Builds datasets
    train_dataset, test_dataset = build_datasets(dataset_config)

    # spliting the test set to responders and non-responders groups.
    non_responders = [p.response_label for p in test_dataset.cells_information_list if p.response_label == 0]
    responders = [p.response_label for p in test_dataset.cells_information_list if p.response_label == 1]

    # a list of biologically significant known genes, to check if there is a significant difference
    # in the expression level between responders and non-responders groups.
    genes = ['GZMH', 'GZMB', 'PRF1', 'HLA-DRA', 'CD38', 'IFI6']

    responder_mean = dict()
    non_responder_mean = dict()
    for gene in genes:
        responder_mean[gene], non_responder_mean[gene] = get_respondness(gene, len(responders), len(non_responders), train_dataset, test_dataset)

    print("responders: ")
    print(responder_mean)
    print("non-responders: ")
    print(non_responder_mean)

    # loading a model for Feature Importance calculation.
    model = pickle.load(open(model_path, "rb"))

    explorer = Feature_Explorer(model, K)
    explorer.k_importance_genes(test_dataset.gene_names)
    df = pd.DataFrame(explorer.k_feature_importance, index=[0])

    # saving a DataFrame with the K most important genes and their scores.
    df.to_csv(experiment_path + '\\k_importance_features_{}.csv'.format(EXPERIMENT_NAME), index=False)
    print(explorer.k_feature_importance)


if __name__ == '__main__':
    print(config)
    dataset_config = config['DATASET']
    xgboost_config = config['XGBOOST']

    main(dataset_config, xgboost_config)
