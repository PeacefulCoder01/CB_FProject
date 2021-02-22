from collections import Counter
import shap
import numpy as np
import xgboost as xgb


class Feature_Explorer:
    def __init__(self, model, k):
        self.model = model
        self.K = k
        self.features_importance = []
        self.k_feature_importance = None
        self.k_shaply_response_values = None
        self.k_shaply_non_response_values = None

    def model_features_importance(self):
        features_importance = []
        for bst, _, _ in self.model.model_layers:
            feature_importance = bst.get_score(importance_type='gain')
            features_importance.append(feature_importance)

        self.features_importance = features_importance

    def get_denominator(self, feature):
        return (feature in self.features_importance[0]) + (feature in self.features_importance[1]) + (feature in self.features_importance[2]) + (feature in self.features_importance[3]) + (feature in self.features_importance[4])

    def get_k_feature_importance(self):
        features = dict.fromkeys(self.model.model_layers[0][0].feature_names, 0)
        for feature_lst in self.features_importance:
            for feature in feature_lst.keys():
                denominator = 0
                if feature in self.features_importance[0]:
                    features[feature] += self.features_importance[0][feature]
                    denominator += 1
                if feature in self.features_importance[1]:
                    features[feature] += self.features_importance[1][feature]
                    denominator += 1
                if feature in self.features_importance[2]:
                    features[feature] += self.features_importance[2][feature]
                    denominator += 1
                if feature in self.features_importance[3]:
                    features[feature] += self.features_importance[3][feature]
                    denominator += 1
                if denominator == 0:
                    continue
                features[feature] /= denominator

        return dict(Counter(features).most_common(self.K))

    def map_features_to_genes(self, k_features, gene_names):
        genes = {}
        for k, v in k_features.items():
            gene = gene_names[int(k[1:])]
            genes[gene] = v
            print("The score of '{}' is {}".format(gene, k_features[k]))
        self.k_feature_importance = genes

    def k_importance_genes(self, gene_names):
        self.model_features_importance()
        k_features = self.get_k_feature_importance()
        return self.map_features_to_genes(k_features, gene_names)

    def get_shaply_values(self, data):
        data_labels = np.array([p.response_label for p in data.cells_information_list])
        dcells = xgb.DMatrix(data.cells, label=data_labels)

        response_indices = [i for i, x in enumerate(data_labels) if x == 1]
        non_response_indices = [i for i, x in enumerate(data_labels) if x == 0]

        response_patients = xgb.DMatrix(data.cells[response_indices], label=np.ones(len(response_indices)))
        non_response_patients = xgb.DMatrix(data.cells[non_response_indices], label=np.zeros(len(non_response_indices)))

        self.k_shaply_response_values = self.shaply_values_calc(response_patients, data.gene_names)
        self.k_shaply_non_response_values = self.shaply_values_calc(non_response_patients, data.gene_names)

    def shaply_values_calc(self, dcells, gene_names):
        xgboost_shap_values = []
        xgboost_shap_values_dict = {}
        xgboost_shap_values_mean = np.zeros(len(dcells.feature_names))
        for idx, (bst, best_threshold_cell_probs, best_threshold_patient_probs) in enumerate(self.model.model_layers):
            explainer = shap.TreeExplainer(bst)
            shap_values = explainer.shap_values(dcells)
            shap_values_mean = np.abs(shap_values).mean(axis=0)
            xgboost_shap_values_mean += shap_values_mean
        indices = sorted(range(len(xgboost_shap_values_mean)), key=lambda i: xgboost_shap_values_mean[i])[-self.K:]
        indices = indices[::-1]
        for i in indices:
            xgboost_shap_values.append(gene_names[i])
            xgboost_shap_values_dict[gene_names[i]] = xgboost_shap_values_mean[i]
        return xgboost_shap_values_dict
