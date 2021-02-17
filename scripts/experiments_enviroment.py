import sys
sys.path.append(r'/srv01/technion/shitay/Code/classifying_response_to_immunotherapy/')
sys.path.append(r'/srv01/technion/shitay/Code/classifying_response_to_immunotherapy/cfg')
sys.path.append(r'/srv01/technion/shitay/Code/classifying_response_to_immunotherapy/Models')
from os.path import join
import os
print("\n\n\n\n########## EXPERIMENT HAS STARTED ##########\n\n\n")
from Models.enhanced_xgboost import Enhanced_XGboost
from DL.data_loading import *
from utilities.smart_seq_dataset import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, accuracy_score
from sklearn import metrics
from utilities.general_helpers import *


# CONFIG_PATH = r'/srv01/technion/shitay/Code/classifying_response_to_immunotherapy/cfg/server_cfg.yaml'
# CONFIG_PATH = r'cfg/server_cfg.yaml'
# CONFIG_PATH = r'cfg\factory_cfg\variance_2_test_percent0.30000000000000004_patients_post_cfg.yaml'
# CONFIG_PATH = r'cfg\xgboost_1_cfg.yaml'
CONFIG_PATH = r'..\cfg\dummy.yaml'
# CONFIG_PATH = sys.argv[1] if len(sys.argv)>1 else r'cfg\xgboost_1_cfg.yaml' # for terminal with outer config operation

EXPERIMENT_NAME, EXPERIMENTS_FOLDER, config = load_yml(CONFIG_PATH)


def build_datasets(dataset_config):
    # extracts params.
    data_path = dataset_config['data_path']
    split_data_path = dataset_config['split_data_path']
    save_division_path = dataset_config['save_division_path']
    test_percent = dataset_config['test_percent']
    patients_type = dataset_config['patients_type']
    variance = dataset_config['variance']

    cells, gene_names, patients_information = extract_smart_seq_data_from_pickle(data_path)
    whole_rna_seq_dataset = RNAseq_Dataset(cells, patients_information, gene_names)
    # filter by cell type
    # if filter_by_cell_type:
    #    idxs = [idx for idx, p in enumerate(whole_rna_seq_dataset.cells_information_list['t_cell_6_cluster']) if p]
    #    whole_rna_seq_dataset = whole_rna_seq_dataset[idxs]

    # 1. keeps only genes greater than given value.
    if variance:
        whole_rna_seq_dataset.filter_genes_by_variance(variance)

    # 2. keeps baseline or post patients..
    if patients_type == 'post':
        whole_rna_seq_dataset = whole_rna_seq_dataset.get_post_patients_sub_dataset()
    elif patients_type == 'pre':
        whole_rna_seq_dataset = whole_rna_seq_dataset.get_baseline_patients_sub_dataset()

    # 3. if there is already division of patients to datasets.
    if split_data_path:
        print(f"Taking exited data division from {split_data_path}")
        train_patients_names = pickle.load(open(split_data_path, "rb"))
        train_dataset, test_dataset = whole_rna_seq_dataset.split_by_patient_names(train_patients_names)
    else:   # uses the configuration to divide into sets.
        print(f"Dividing data into test/train sets")
        train_dataset, test_dataset, train_idxs, test_idxs = whole_rna_seq_dataset.train_test_split(test_size=test_percent,
                                                                                                    shuffle=True)
        if save_division_path:  # saves the new division for future use.
            pickle.dump((train_dataset.get_all_patients_names()), open(save_division_path, "wb"))
            print(f"New data sets divisions saved in {save_division_path}")

    return train_dataset, test_dataset


@experiment_manager(EXPERIMENT_NAME, EXPERIMENTS_FOLDER, CONFIG_PATH)
def main(dataset_config, xgboost_config, experiment_config):
    # Extracts params.
    experiment_path = os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT_NAME)
    num_round = xgboost_config['num_round']
    early_stopping_rounds = xgboost_config['early_stopping_rounds']
    k_folds = xgboost_config['k_folds']

    # Builds datasets
    train_dataset, test_dataset = build_datasets(dataset_config)
    print(f'Train dataset patients: {train_dataset.get_all_patients_names()}')
    print(f'Test dataset patients: {train_dataset.get_all_patients_names()}')

    # Builds enhanced XGBoost model.
    model = Enhanced_XGboost(num_round, early_stopping_rounds, k_folds)

    # Trains.
    model.train(train_dataset, True)


    # Inferences on train set.
    print("----------------------------------------------")
    print("Train inference")
    patients_preds, cells_preds, df_groupby_patients = model.inference(train_dataset)
    patients_labels = df_groupby_patients.values.T[0]
    cells_labels = np.array([p.response_label for p in train_dataset.cells_information_list])
    # print(f'   train set cells predictions AUC: {roc_auc_score(cells_labels, avg_prob_cells_predictions)}')
    # print(f'train set patients predictions AUC: {roc_auc_score(patients_labels, patients_predictions)}')
    print(df_groupby_patients)
    cells_acc = round(accuracy_score(cells_labels, cells_preds), 3)
    patients_acc = round(accuracy_score(patients_labels, patients_preds), 3)
    print(f'Train cells predictions CM:\n{visualization_confusion_matrix(cells_labels, cells_preds, f" {EXPERIMENT_NAME} inference train set cells, accuracy: {cells_acc}", join(experiment_path, f"CM train cells {cells_acc}"))}')
    print(f'Train patients predictions CM:\n{visualization_confusion_matrix(patients_labels, patients_preds, f"{EXPERIMENT_NAME} inference train set patients, accuracy: {patients_acc}", join(experiment_path, f"CM train patients {patients_acc}"))}')
    print(f'Train cells classification accuracy:\n{cells_acc}')
    print(f'Train patients classification accuracy:\n{patients_acc}')


    # Inferences on test set.
    print("----------------------------------------------")
    print("Test inference")
    patients_preds, cells_preds, df_groupby_patients = model.inference(test_dataset)
    patients_labels = df_groupby_patients.values.T[0]
    cells_labels = np.array([p.response_label for p in test_dataset.cells_information_list])
    # print(f'   test set cells predictions AUC: {roc_auc_score(cells_labels, avg_prob_cells_predictions)}')
    # print(f'test set patients predictions AUC: {roc_auc_score(patients_labels, patients_predictions)}')
    print(df_groupby_patients)
    cells_acc = round(accuracy_score(cells_labels, cells_preds), 3)
    patients_acc = round(accuracy_score(patients_labels, patients_preds), 3)
    print(f'Test cells predictions CM:\n{visualization_confusion_matrix(cells_labels, cells_preds, f" {EXPERIMENT_NAME} inference test set cells, accuracy: {cells_acc}", join(experiment_path, f"CM test cells {cells_acc}"))}')
    print(f'Test patients predictions CM:\n{visualization_confusion_matrix(patients_labels, patients_preds, f"{EXPERIMENT_NAME} inference test set patients, accuracy: {patients_acc}", join(experiment_path, f"CM test patients {patients_acc}"))}')
    print(f'Test cells classification accuracy:\n{cells_acc}')
    print(f'Test patients classification accuracy:\n{patients_acc}')


    # Feature importance
    print("----------------------------------------------")
    print("Feature Importance")
    model.most_important_5 = model.get_feature_importance()
    for k, v in model.most_important_5.items():
        print("The score of '{}' is {}".format(train_dataset.gene_names[int(k[1:])], model.most_important_5[k]))
    print(model.most_important_5)


    # Save model.
    if experiment_config['save_model']:
        model.save_model_in_pkl(os.path.join(experiment_config['experiments_folder'], experiment_config['experiment_name']))


if __name__ == '__main__':
    print(config)
    dataset_config = config['DATASET']
    xgboost_config = config['XGBOOST']
    experiment_config = config['EXPERIMENT']

    main(dataset_config, xgboost_config, experiment_config)
