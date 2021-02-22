import pickle

from Models.feature_explorer import Feature_Explorer
from scripts.experiments_enviroment import build_datasets
from utilities.general_helpers import load_yml

CONFIG_PATH = r'..\cfg\feature_importance_cfg.yaml'

EXPERIMENT_NAME, EXPERIMENTS_FOLDER, config = load_yml(CONFIG_PATH)


def main(dataset_config, xgboost_config):
    model_path = xgboost_config['model_path']
    K = xgboost_config['K']

    # Builds datasets
    train_dataset, test_dataset = build_datasets(dataset_config)

    model = pickle.load(open(model_path, "rb"))

    explorer = Feature_Explorer(model, K)
    explorer.k_importance_genes(test_dataset.gene_names)
    print(explorer.k_feature_importance)

    explorer.get_shaply_values(test_dataset)
    print("response genes: ", explorer.k_shaply_response_values)
    print("non-response genes: ", explorer.k_shaply_non_response_values)


if __name__ == '__main__':
    print(config)
    dataset_config = config['DATASET']
    xgboost_config = config['XGBOOST']

    main(dataset_config, xgboost_config)

