from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from sklearn.metrics import accuracy_score
from Models.enhanced_xgboost import Enhanced_XGboost
from DL.data_loading import *
from utilities.smart_seq_dataset import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn import metrics
from utilities.general_helpers import *


# def some_weird_func(args):
#     """
#     3*13+20+30=89     opt params x=13, y=20, z=14
#     :param x: [0-100] opt 20
#     :param y: [0 - 100] opt 20
#     :param z: [0 - 100] opt 1-14  if z>x
#     :return:
#     """
#     x, y, z = args
#     ret = 0
#     if x>20:
#         ret += -x
#     else:
#         ret += 3 * x
#
#     if y>20:
#         ret -= y
#     elif y<5:
#         ret -= y
#     else:
#         ret += y
#
#     if z>x and z<15:
#         ret += 30
#
#     return -ret
#
#
# def some_weird_func2(args):
#     """
#     3*13+20+30=89     opt params x=13, y=20, z=14
#     :param x: [0-100] opt 20
#     :param y: [0 - 100] opt 20
#     :param z: [0 - 100] opt 1-14  if z>x
#     :return:
#     """
#     x, y, z = args
#     return x+y+z

# def obj(args):
#
#     return accuracy_score(y, predict)
DATA_PATH = r'DATA\RNAseq_DATA.p'
MAX_EVAL = 500


# Daatset params
patients_type = 'all'
split_data_path = r'DATA\patients_divisions\train_patients_0.9percent_all.p'
test_percent = 0.1
cells, gene_names, patients_information = extract_smart_seq_data_from_pickle(DATA_PATH)
whole_rna_seq_dataset = RNAseq_Dataset(cells, patients_information, gene_names)

#region Data preprocess
if patients_type == 'post':
    whole_rna_seq_dataset = whole_rna_seq_dataset.get_post_patients_sub_dataset()
elif patients_type == 'pre':
    whole_rna_seq_dataset = whole_rna_seq_dataset.get_baseline_patients_sub_dataset()
#endregion


def build_datasets(variance=0):
    if variance:
        filtered_rna_seq_dataset = whole_rna_seq_dataset.filter_genes_by_variance(variance, in_place=False)
    if split_data_path:
        print(f"Taking exited data division from {split_data_path}")
        train_patients_names = pickle.load(open(split_data_path, "rb"))
        train_dataset, test_dataset = filtered_rna_seq_dataset.split_by_patient_names(train_patients_names)
    else:  # uses the configuration to divide into sets.
        print(f"Dividing data into test/train sets")
        train_dataset, test_dataset, train_idxs, test_idxs = filtered_rna_seq_dataset.train_test_split(
            test_size=test_percent,
            shuffle=True)
    return train_dataset, test_dataset


def objective(args, best_lr, best_var):
    # print(args)
    # Extracts params.
    num_round, early_stopping_rounds, k_folds = args
    # Builds datasets
    train_dataset, test_dataset = build_datasets(best_var)

    # Builds enhanced XGBoost model.
    model = Enhanced_XGboost(num_round, early_stopping_rounds, k_folds, learning_rate=best_lr)

    # Trains.
    model.train(train_dataset)

    # Inferences on train set.

    # patients_preds, cells_preds, df_groupby_patients = model.inference(train_dataset)
    # patients_labels = df_groupby_patients.values.T[0]
    # cells_labels = np.array([p.response_label for p in train_dataset.cells_information_list])
    #
    # train_cells_classification_accuracy = accuracy_score(cells_labels, cells_preds)
    # train_patients_classification_accuracy = accuracy_score(patients_labels, patients_preds)


    # Inferences on test set.
    patients_preds, cells_preds, df_groupby_patients = model.inference(test_dataset)
    patients_labels = df_groupby_patients.values.T[0]
    cells_labels = np.array([p.response_label for p in test_dataset.cells_information_list])
    test_cells_classification_accuracy = accuracy_score(cells_labels, cells_preds)
    test_patients_classification_accuracy = accuracy_score(patients_labels, patients_preds)

    # score = -1 * (test_cells_classification_accuracy +
    #               test_patients_classification_accuracy +
    #               0.2 * train_cells_classification_accuracy +
    #               0.2 * train_patients_classification_accuracy)

    score = -1 * (test_cells_classification_accuracy +
                  test_patients_classification_accuracy)
    return score


def learning_rate_objective(args, train_dataset, test_dataset):
    # print(args)
    # Extracts params.
    learning_rate = args[0]
    # Builds enhanced XGBoost model.
    model = Enhanced_XGboost(num_round=10, early_stopping_rounds=10, k_folds=5, learning_rate=learning_rate)

    # Trains.
    model.train(train_dataset)

    patients_preds, cells_preds, df_groupby_patients = model.inference(test_dataset)
    patients_labels = df_groupby_patients.values.T[0]
    cells_labels = np.array([p.response_label for p in test_dataset.cells_information_list])
    test_cells_classification_accuracy = accuracy_score(cells_labels, cells_preds)
    test_patients_classification_accuracy = accuracy_score(patients_labels, patients_preds)

    score = -1 * (test_cells_classification_accuracy +
                  test_patients_classification_accuracy)
    return score


def variance_objective(args, lr):
    # print(args)
    # Extracts params.
    variance = args[0]
    train_dataset, test_dataset = build_datasets(variance=variance)
    # Builds enhanced XGBoost model.
    model = Enhanced_XGboost(num_round=10, early_stopping_rounds=10, k_folds=5, learning_rate=lr)

    # Trains.
    model.train(train_dataset)

    patients_preds, cells_preds, df_groupby_patients = model.inference(test_dataset)
    patients_labels = df_groupby_patients.values.T[0]
    cells_labels = np.array([p.response_label for p in test_dataset.cells_information_list])
    test_cells_classification_accuracy = accuracy_score(cells_labels, cells_preds)
    test_patients_classification_accuracy = accuracy_score(patients_labels, patients_preds)

    score = -1 * (test_cells_classification_accuracy +
                  test_patients_classification_accuracy)
    return score


if __name__ == '__main__':
    """
    regularization = over-fit
    max_depth [2, 30] >> too short under-fitting, too high over-fitting 
    sub_sample = [0.1 - 1] >> regularization, which part of your data it'll take for model building. 1- all the data (not in use)
    colsample_bytree = [0.1- 1] >>
    colsample_bylevel = [0.1- 1] >> regularization, which part of columns (genomes) it'll take in each node. 1- all columns(not in use)
    min_child_weight = [1 - 100] >> regularization,  limits the amount of min children in leaf, in regression XGBoost it's exactly the min number of cildren in leaf.
    colsample_bynode
    reg_alpha 
    reg_lambda
    n_estimators
    max_depth = [10 - 1000] >> raising it makes it better till some point.
    learning_rate = [0.01 - 0.8] >> important tuning - too small will never converge. too high not proper result.
    gamma >> regularization
    """

    # Only LR
    lr_space = [hp.uniform('learning_rate', 0.01, 0.8)]
    train_dataset, test_dataset = build_datasets(variance=6)
    lr_trials = Trials()
    best_lr = fmin(lambda args: learning_rate_objective(args, train_dataset, test_dataset), lr_space, algo=tpe.suggest, max_evals=MAX_EVAL, trials=lr_trials)
    print(space_eval(lr_space, best_lr))
    best_lr = best_lr['learning_rate']
    print(f'Best LR {best_lr}')

    # Only Variance
    variance_space = [hp.uniformint('variance', 0, 8)]
    variance_trials = Trials()
    best_var = fmin(lambda args: variance_objective(args, best_lr), variance_space, algo=tpe.suggest, max_evals=9, trials=variance_trials)
    print(space_eval(variance_space, best_var))
    best_var = best_var['variance']
    print(f'Best Variance {best_var}')


    # Other variables
    list_space = [hp.choice('num_round', [10]),
                  hp.choice('early_stopping_rounds', [10]),
                  hp.choice('k_folds', [5])]

    trials = Trials()
    best_values = fmin(lambda args: objective(args, best_lr, best_var), list_space, algo=tpe.suggest, max_evals=MAX_EVAL, trials=trials)

    print(best_values)

    print(space_eval(list_space, best_values))
    # print(trials.trials)
