from DL.data_loading import *
import xgboost as xgb
from utilities.smart_seq_dataset import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn import metrics
from utilities.general_helpers import *
import pickle
import pandas as pd
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'
# EXPERIMENT_NAME = "Dummy"
PICKLE_PATH = r'DATA\RNAseq_DATA.p'
CONFIG_PATH = sys.argv[1] if len(sys.argv)>1 else r'cfg\xgboost_1_cfg.yaml' # for terminal with outer config operation
# CONFIG_PATH = r'cfg\dummy.yaml'
# CONFIG_PATH = r'cfg\factory_cfg\variance_2_test_percent0.30000000000000004_patients_post_cfg.yaml'

EXPERIMENT_NAME, EXPERIMENTS_FOLDER, config = load_yml(CONFIG_PATH)

# EXPERIMENT_NAME = "XGBOOST_all_patients_10percent_testset"
# EXPERIMENTS_FOLDER = r'DATA\experiments'


def pick_best_threshold(labels, predictions_probs):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions_probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold


def combine_cells_classification_to_predict_patient_response(dataset, test_idxs, y_pred):
    test_set = dataset[test_idxs]
    ll = [[p.patient_details, p.response_label, y_pred[idx]] for idx, p in enumerate(test_set.patients)]
    df = pd.DataFrame(ll, columns=["patients", "labels", 'predictions probabilities']).groupby(['patients']).mean()
    labels = df.values.T[0]
    predictions_probs = df.values.T[1]
    print(f'TEST PATIENT CLASSIFICATION - AUC: {roc_auc_score(labels, predictions_probs)}')
    np.argmax(metrics.roc_curve(labels, predictions_probs)[1] - metrics.roc_curve(labels, predictions_probs)[0])
    best_threshold_cell_probs = pick_best_threshold(labels, predictions_probs)
    print(f"Best threshold {best_threshold_cell_probs}")
    predictions = threshold_predict(predictions_probs, best_threshold_cell_probs)
    df['final predictions'] = predictions
    # visualization_confusion_matrix(labels, predictions)
    df = df[["labels", 'final predictions', 'predictions probabilities']]
    print(df)


def threshold_predict(a, threshold):
    return (a>=threshold).astype(np.int)


def majority_vote(all_predictions):
    all_predictions = np.array(all_predictions)
    maj_vote_predictions = np.mean(all_predictions, axis=0)
    return maj_vote_predictions


def save_model_in_pkl(model):
    pickle.dump((model), open(os.path.join(EXPERIMENTS_FOLDER, EXPERIMENT_NAME, "model.pkl"), "wb"))


# # @experiment_manager(EXPERIMENT_NAME, EXPERIMENTS_FOLDER)
# def main(test_percent, patients, num_round, early_stopping_rounds, k_folds, variance):
#     cells, gene_names, patients_information = extract_data_from_pickle(PICKLE_PATH)
#     origin_dataset = RNAseq_Dataset(cells, patients_information, gene_names)
#     if patients == 'post':
#         origin_dataset = origin_dataset.get_post_patients_sub_dataset()
#     elif patients == 'pre':
#         origin_dataset = origin_dataset.get_baseline_patients_sub_dataset()
#     if variance:
#         origin_dataset.filter_genes_by_variance(variance)
#     _, _, _, _, train_idxs, test_idxs = origin_dataset.train_test_split(test_size=test_percent, shuffle=True)
#     training_dataset = origin_dataset[train_idxs]
#     test_dataset = origin_dataset[test_idxs]
#
#     # K fold cross validation
#     print("\n\n\n###########################################")
#
#     k_validation = training_dataset.k_fold_cross_validation(k_folds)
#     bsts = []
#     for x_train, x_val, y_train, y_val, _, _ in k_validation:
#         dtrain = xgb.DMatrix(x_train, label=y_train)
#         dval = xgb.DMatrix(x_val, label=y_val)
#
#         param = {'max_depth': 20, 'eta': 1, 'objective': 'binary:logistic'}
#         param['nthread'] = 4
#         param['eval_metric'] = 'auc'
#         # evallist = [(dtest, 'eval'), (dtrain, 'train')]
#         evallist = [(dtrain, 'train'), (dval, 'validation')]
#
#         # 2. Train model
#         bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=early_stopping_rounds)
#         bsts.append(bst)
#         print("\n\n\n###########################################")
#
#     test_labels = np.array([p.response_label for p in test_dataset.patients])
#     dtest = xgb.DMatrix(test_dataset.cells, label=test_labels)
#     all_predictions = []
#
#     for idx, bst in enumerate(bsts):
#         print(f"INFERENCE - Classifier number {idx+1}")
#         ypred_probs = bst.predict(dtest)
#         all_predictions.append(ypred_probs)
#         print(f'TEST CELLS CLASSIFICATION AUC: {roc_auc_score(test_labels, ypred_probs)}')
#         ypred = threshold_predict(ypred_probs, pick_best_threshold(test_labels, ypred_probs))
#         visualization_confusion_matrix(test_labels, ypred)
#         combine_cells_classification_to_predict_patient_response(origin_dataset, test_idxs, ypred_probs)
#         print("\n\n\n--------------------------------------------------")
#
#     maj_vote_predictions_probs = majority_vote(all_predictions)
#     print(f'maj vote test AUC: {roc_auc_score(test_labels, maj_vote_predictions_probs)}')
#     maj_vote_predictions = threshold_predict(maj_vote_predictions_probs, pick_best_threshold(test_labels, maj_vote_predictions_probs))
#     visualization_confusion_matrix(test_labels, maj_vote_predictions)
#     combine_cells_classification_to_predict_patient_response(origin_dataset, test_idxs, maj_vote_predictions_probs)
#     save_model_in_pkl(bsts)


@experiment_manager(EXPERIMENT_NAME, EXPERIMENTS_FOLDER)
def main(test_percent, patients, num_round, early_stopping_rounds, k_folds, variance):
    cells, gene_names, patients_information = extract_smart_seq_data_from_pickle(PICKLE_PATH)
    origin_dataset = RNAseq_Dataset(cells, patients_information, gene_names)
    if patients == 'post':
        origin_dataset = origin_dataset.get_post_patients_sub_dataset()
    elif patients == 'pre':
        origin_dataset = origin_dataset.get_baseline_patients_sub_dataset()
    if variance:
        origin_dataset.filter_genes_by_variance(variance)
    # _, _, _, _, train_idxs, test_idxs = origin_dataset.train_test_split(test_size=test_percent, shuffle=True)

    train_patients_names = pickle.load(open(r"DATA\patients_divisions\train_patients_0.9percent_all.p", "rb"))
    train_idxs = [idx for idx, p in enumerate(origin_dataset.cells_information_list['patient_details']) if p in train_patients_names]
    test_idxs = [idx for idx, p in enumerate(origin_dataset.cells_information_list['patient_details']) if p not in train_patients_names]
    training_dataset = origin_dataset[train_idxs]
    test_dataset = origin_dataset[test_idxs]

    # K fold cross validation
    print("\n\n\n###########################################")

    k_validation = training_dataset.k_fold_cross_validation(k_folds)
    bsts = []
    for x_train, x_val, y_train, y_val, _, _ in k_validation:
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dval = xgb.DMatrix(x_val, label=y_val)

        param = {'max_depth': 20, 'eta': 1, 'objective': 'binary:logistic'}
        param['nthread'] = 4
        param['eval_metric'] = 'auc'
        # evallist = [(dtest, 'eval'), (dtrain, 'train')]
        evallist = [(dtrain, 'train'), (dval, 'validation')]

        # 2. Train model
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=early_stopping_rounds)
        bsts.append(bst)
        print("\n\n\n###########################################")

    test_labels = np.array([p.response_label for p in test_dataset.patients])
    dtest = xgb.DMatrix(test_dataset.cells, label=test_labels)
    all_predictions = []

    for idx, bst in enumerate(bsts):
        print(f"INFERENCE - Classifier number {idx+1}")
        ypred_probs = bst.predict(dtest)
        all_predictions.append(ypred_probs)
        print(f'TEST CELLS CLASSIFICATION AUC: {roc_auc_score(test_labels, ypred_probs)}')
        ypred = threshold_predict(ypred_probs, pick_best_threshold(test_labels, ypred_probs))
        visualization_confusion_matrix(test_labels, ypred)
        combine_cells_classification_to_predict_patient_response(origin_dataset, test_idxs, ypred_probs)
        print("\n\n\n--------------------------------------------------")

    maj_vote_predictions_probs = majority_vote(all_predictions)
    print(f'maj vote test AUC: {roc_auc_score(test_labels, maj_vote_predictions_probs)}')
    maj_vote_predictions = threshold_predict(maj_vote_predictions_probs, pick_best_threshold(test_labels, maj_vote_predictions_probs))
    visualization_confusion_matrix(test_labels, maj_vote_predictions)
    combine_cells_classification_to_predict_patient_response(origin_dataset, test_idxs, maj_vote_predictions_probs)
    save_model_in_pkl(bsts)



if __name__ == '__main__':
    print(config)
    test_percent = config['DATASET']['test_percent']
    patients = config['DATASET']['patients_type']
    variance = config['DATASET']['variance']
    num_round = config['XGBOOST']['num_round']
    early_stopping_rounds = config['XGBOOST']['early_stopping_rounds']
    k_folds = config['XGBOOST']['k_folds']
    main(test_percent, patients, num_round, early_stopping_rounds, k_folds, variance)


