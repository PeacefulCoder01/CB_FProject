from collections import Counter


class Feature_Explorer:
    def __init__(self, model, k):
        self.model = model
        self.K = k
        self.features_importance = []
        self.k_feature_importance = None

    def model_features_importance(self):
        """
        The model has 5 XGBoost models,
        The function calculates the 'Feature Importance' of each of them.

        :return: list of lists, in each list, the feature importance of the specific XGBoost model.
        """
        features_importance = []
        for bst, _, _ in self.model.model_layers:
            feature_importance = bst.get_score(importance_type='gain')
            features_importance.append(feature_importance)

        self.features_importance = features_importance

    def get_k_feature_importance(self):
        """
        The function calculates the K most important genes across the 5 XGBoost models.

        :return: dictionary with the K most important genes and their score.
        """
        features = dict.fromkeys(self.model.model_layers[0][0].feature_names, 0)
        for feature_lst in self.features_importance:
            for feature in feature_lst.keys():
                # for calculating the relative value of the gene's score
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
        """
        The XGBoost model names the features in ascending order ('f1', 'f2', etc).
        The function maps the features to the original names of the K most important genes.

        :param k_features: dictionary of the K most important genes.
        :param gene_names: list of the genes (in the same order as the features).
        :return: dictionary of the genes (by their names) and their feature importance score.
        """
        genes = {}
        for k, v in k_features.items():
            if int(k[1:]) > len(gene_names):
                continue
            gene = gene_names[int(k[1:])]
            genes[gene] = v
            print("The score of '{}' is {}".format(gene, k_features[k]))
        self.k_feature_importance = genes

    def k_importance_genes(self, gene_names):
        """
        The function gets the K most important genes of the Enhanced XGBoost model.

        :param gene_names: list of gene names for reverse mapping the features to genes.
        :return: dictionary of the genes (by their names) and their feature importance score.
        """
        self.model_features_importance()
        k_features = self.get_k_feature_importance()
        return self.map_features_to_genes(k_features, gene_names)
