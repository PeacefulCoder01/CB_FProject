import xgboost as xgb
from sklearn import metrics
from utilities.general_helpers import *
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score


def pick_best_threshold(labels, predictions_probs):
    """
    Uses ROC (FTR, TPR) values, in which we decide what is the best threshold. when 1-TPR = 1-FPR for simplicity.
    :param labels: in order to calculate ROC.
    :param predictions_probs: in order to calculate ROC.
    :return: the best threshold.
    """
    fpr, tpr, thresholds = metrics.roc_curve(labels, predictions_probs)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return best_threshold


def make_threshold_prediction(probabilities, threshold):
    """
    Binary decision based on threshold.
    :param probabilities: which we check threshold.
    :return: list of 0/1 (greater or smaller than threshold).
    """
    return (probabilities >= threshold).astype(np.int)


def patients_average_cells_predictions(rna_seq_dataset, pred_prob):
    """
    mark probability of each patient as the average of all their prediction cells probabilities.
    :param rna_seq_dataset: the dataset which has been predicted.
    :param pred_prob: prediction cells probabilities of all rna_seq_dataset cells.
    :return: patients labels, patients predictions probs, df_groupby_patients
    """
    match_patient_pred = [[p.patient_details, p.response_label, pred_prob[idx]] for
                          idx, p in enumerate(rna_seq_dataset.cells_information_list)]
    df_groupby_patients = pd.DataFrame(match_patient_pred, columns=["patients",
                                                                    "labels",
                                                                    'predictions probabilities']).\
        groupby(['patients']).mean()
    patients_labels = df_groupby_patients.values.T[0]
    patients_predictions_probs = df_groupby_patients.values.T[1]
    return patients_labels, patients_predictions_probs, df_groupby_patients


class Enhanced_XGboost:
    def __init__(self, num_round, early_stopping_rounds, k_folds):
        self.params = {'max_depth': 20,
                       'eta': 1,
                       'objective': 'binary:logistic',
                       'nthread': 4,
                       'eval_metric': 'auc'}
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.k_folds = k_folds
        self.model_layers = []
        self.patient_prediction_threshold = None
        self.cells_presictions_threshold = None

    def train(self, rna_seq_dataset, verbose=False):
        """
        important Note: it's a k_fold_cross_val training.
        Splits RNAseq dataset into folds and sets each fold as validation set iteratively. Each iteration trains XGBoost
        model on training-set. Then, predicts binary response to validation fold using trained XGBoost,
        calculate average probabilities per patient. and set threshold of each XGBoost model that maximizes TPR and
        minimizes FTR.
        :param rna_seq_dataset: The set in which the k-folds of train/validation sets.
        :param verbose: to print information during training.
        """
        self.patient_prediction_threshold = 0
        self.cells_presictions_threshold = 0
        k_fold_validation = rna_seq_dataset.k_fold_cross_validation(self.k_folds)
        for idx, (x_train, x_val, y_train, y_val, train_idxs, val_idxs) in enumerate(k_fold_validation):
            if verbose:
                print(f'Train XGBoost on fold number {idx + 1}|{self.k_folds}')
            # Trains current k-1 folds and validate by the k fold.
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)
            bst = xgb.train(self.params, dtrain, self.num_round,
                            evals=[(dtrain, 'train'), (dval, 'validation')],
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval=verbose)

            # Defines threshold - majority vote over all cells predictions of patient and then correlation with labels.
            pred_cells_prob_val = bst.predict(dval)
            val_set = rna_seq_dataset[val_idxs]
            best_threshold_cell_probs = pick_best_threshold(labels=y_val,
                                                            predictions_probs=pred_cells_prob_val)
            val_cells_preds = make_threshold_prediction(pred_cells_prob_val, best_threshold_cell_probs)
            patients_labels, patients_predictions_probs, _ = patients_average_cells_predictions(val_set, val_cells_preds)
            best_threshold_patient_probs = pick_best_threshold(labels=patients_labels,
                                                            predictions_probs=patients_predictions_probs)

            # Adds layer (XGBoost) to the model.
            self.model_layers.append((bst, best_threshold_cell_probs, best_threshold_patient_probs))
            # self.patient_prediction_threshold += best_threshold_patient_probs / self.k_folds
            # self.cells_presictions_threshold += best_threshold_cell_probs / self.k_folds
            if verbose:
                print("\n\n\n###########################################")

    def inference(self, rna_seq_dataset, verbose=False):
        """
        Model's Inference. Each XGBoost predicts cells probabilities of response. Then activates threshold on the
        average cells response.
        :param rna_seq_dataset: All the data will be predicted.
        :return: patients predictions, avg_prob_cells_predictions, df_groupby_patients['patients']
        """
        if len(self.model_layers) == 0:
            return "model hasn't trained"

        data_labels = np.array([p.response_label for p in rna_seq_dataset.cells_information_list])
        dcells = xgb.DMatrix(rna_seq_dataset.cells, label=data_labels)
        xgboost_patients_binary_preds = []
        xgboost_cells_binary_preds = []
        for idx, (bst, best_threshold_cell_probs, best_threshold_patient_probs) in enumerate(self.model_layers):
            if verbose:
                print(f"XGBoost number {idx+1}")
            # Gets XGBoost predictions of cells.
            cells_prob_preds = bst.predict(dcells)
            cells_binary_preds = make_threshold_prediction(cells_prob_preds, best_threshold_cell_probs)

            # Gets patients predictions by looking at cells predictions of each patient
            patients_labels, patients_prob_preds, df_groupby_patients = patients_average_cells_predictions(
                rna_seq_dataset, cells_binary_preds)
            patients_binary_preds = make_threshold_prediction(patients_prob_preds, best_threshold_patient_probs)

            # appends classifications to the lists
            xgboost_patients_binary_preds.append(patients_binary_preds)
            xgboost_cells_binary_preds.append(cells_binary_preds)

        df_groupby_patients.drop(columns=['predictions probabilities'], inplace=True)
        # Calculates avg classifications of all XGBoosts.
        patients_preds_XGBoost_votes = np.sum(np.array(xgboost_patients_binary_preds), axis=0)/self.k_folds
        cellss_preds_XGBoost_votes = np.sum(np.array(xgboost_cells_binary_preds), axis=0)/self.k_folds
        df_groupby_patients['predictions XGBoost votes'] = patients_preds_XGBoost_votes
        # majority vote
        final_patients_preds = (patients_preds_XGBoost_votes >= 0.5).astype(np.int)
        final_cells_preds = (cellss_preds_XGBoost_votes >= 0.5).astype(np.int)
        df_groupby_patients['final predictions'] = final_patients_preds

        # visualization_confusion_matrix(labels, predictions)
        df_groupby_patients = df_groupby_patients[["labels", 'final predictions', 'predictions XGBoost votes']]
        return final_patients_preds, final_cells_preds, df_groupby_patients

    def save_model_in_pkl(self, path, filename="Enhanced_XGboost_Model.pkl"):
        """
        Save model local in PKL file.
        :param path: where the model be saved
        :param filename: name of the new file'll be created.
        """
        pickle.dump(self, open(os.path.join(path, filename), "wb"))



class hands_on_Enhanced_XGboost:
    """
    Hands-on: full-control of the hyper-parameters.
    """
    def __init__(self, num_round, early_stopping_rounds=10, k_folds=5,
                 max_depth=6, learning_rate=0.3, gamma=None, max_delta_step=0, lambda_reg=None,
                 alpha=0, nthread=4):
        if gamma and lambda_reg:
            self.params = {'max_depth': max_depth,
                           'eta': learning_rate,
                           'gamma ': gamma,
                           'max_delta_step': max_delta_step,
                           'lambda ': lambda_reg,
                           'alpha': alpha,
                           'objective': 'binary:logistic',
                           'nthread': nthread,
                           'eval_metric': 'auc'}
        else:
            self.params = {'max_depth': max_depth,
                           'eta': learning_rate,
                           'max_delta_step': max_delta_step,
                           'alpha': alpha,
                           'objective': 'binary:logistic',
                           'nthread': nthread,
                           'eval_metric': 'auc'}
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds
        self.k_folds = k_folds
        self.model_layers = []
        self.patient_prediction_threshold = None
        self.cells_presictions_threshold = None

    def train(self, rna_seq_dataset, verbose=False):
        """
        important Note: it's a k_fold_cross_val training.
        Splits RNAseq dataset into folds and sets each fold as validation set iteratively. Each iteration trains XGBoost
        model on training-set. Then, predicts binary response to validation fold using trained XGBoost,
        calculate average probabilities per patient. and set threshold of each XGBoost model that maximizes TPR and
        minimizes FTR.
        :param rna_seq_dataset: The set in which the k-folds of train/validation sets.
        :param verbose: to print information during training.
        """
        self.patient_prediction_threshold = 0
        self.cells_presictions_threshold = 0
        k_fold_validation = rna_seq_dataset.k_fold_cross_validation(self.k_folds, verbose)
        for idx, (x_train, x_val, y_train, y_val, train_idxs, val_idxs) in enumerate(k_fold_validation):
            if verbose:
                print(f'Train XGBoost on fold number {idx + 1}|{self.k_folds}')
            # Trains current k-1 folds and validate by the k fold.
            dtrain = xgb.DMatrix(x_train, label=y_train)
            dval = xgb.DMatrix(x_val, label=y_val)
            bst = xgb.train(self.params, dtrain, self.num_round,
                            evals=[(dtrain, 'train'), (dval, 'validation')],
                            early_stopping_rounds=self.early_stopping_rounds,
                            verbose_eval=verbose)

            # Defines threshold - majority vote over all cells predictions of patient and then correlation with labels.
            pred_cells_prob_val = bst.predict(dval)
            val_set = rna_seq_dataset[val_idxs]
            best_threshold_cell_probs = pick_best_threshold(labels=y_val,
                                                            predictions_probs=pred_cells_prob_val)
            val_cells_preds = make_threshold_prediction(pred_cells_prob_val, best_threshold_cell_probs)
            patients_labels, patients_predictions_probs, _ = patients_average_cells_predictions(val_set, val_cells_preds)
            best_threshold_patient_probs = pick_best_threshold(labels=patients_labels,
                                                            predictions_probs=patients_predictions_probs)

            # Adds layer (XGBoost) to the model.
            self.model_layers.append((bst, best_threshold_cell_probs, best_threshold_patient_probs))
            # self.patient_prediction_threshold += best_threshold_patient_probs / self.k_folds
            # self.cells_presictions_threshold += best_threshold_cell_probs / self.k_folds
            if verbose:
                print("\n\n\n###########################################")

    def inference(self, rna_seq_dataset, verbose=False):
        """
        Model's Inference. Each XGBoost predicts cells probabilities of response. Then activates threshold on the
        average cells response.
        :param rna_seq_dataset: All the data will be predicted.
        :return: patients predictions, avg_prob_cells_predictions, df_groupby_patients['patients']
        """
        if len(self.model_layers) == 0:
            return "model hasn't trained"

        data_labels = np.array([p.response_label for p in rna_seq_dataset.cells_information_list])
        dcells = xgb.DMatrix(rna_seq_dataset.cells, label=data_labels)
        xgboost_patients_binary_preds = []
        xgboost_cells_binary_preds = []
        for idx, (bst, best_threshold_cell_probs, best_threshold_patient_probs) in enumerate(self.model_layers):
            if verbose:
                print(f"XGBoost number {idx+1}")
            # Gets XGBoost predictions of cells.
            cells_prob_preds = bst.predict(dcells)
            cells_binary_preds = make_threshold_prediction(cells_prob_preds, best_threshold_cell_probs)

            # Gets patients predictions by looking at cells predictions of each patient
            patients_labels, patients_prob_preds, df_groupby_patients = patients_average_cells_predictions(
                rna_seq_dataset, cells_binary_preds)
            patients_binary_preds = make_threshold_prediction(patients_prob_preds, best_threshold_patient_probs)

            # appends classifications to the lists
            xgboost_patients_binary_preds.append(patients_binary_preds)
            xgboost_cells_binary_preds.append(cells_binary_preds)

        df_groupby_patients.drop(columns=['predictions probabilities'], inplace=True)
        # Calculates avg classifications of all XGBoosts.
        patients_preds_XGBoost_votes = np.sum(np.array(xgboost_patients_binary_preds), axis=0)/self.k_folds
        cellss_preds_XGBoost_votes = np.sum(np.array(xgboost_cells_binary_preds), axis=0)/self.k_folds
        df_groupby_patients['predictions XGBoost votes'] = patients_preds_XGBoost_votes
        # majority vote
        final_patients_preds = (patients_preds_XGBoost_votes >= 0.5).astype(np.int)
        final_cells_preds = (cellss_preds_XGBoost_votes >= 0.5).astype(np.int)
        df_groupby_patients['final predictions'] = final_patients_preds

        # visualization_confusion_matrix(labels, predictions)
        df_groupby_patients = df_groupby_patients[["labels", 'final predictions', 'predictions XGBoost votes']]
        return final_patients_preds, final_cells_preds, df_groupby_patients

    def save_model_in_pkl(self, path, filename="Enhanced_XGboost_Model.pkl"):
        """
        Save model local in PKL file.
        :param path: where the model be saved
        :param filename: name of the new file'll be created.
        """
        pickle.dump(self, open(os.path.join(path, filename), "wb"))


