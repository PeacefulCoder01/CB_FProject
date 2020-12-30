"""
That script helped to explore at the first time the smart-seq data of 2018.
We used it to cluster, visualize and reproduce steps from the article.
"""

from sklearn.metrics import roc_auc_score
from utilities.smart_seq_dataset import *
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from Bio.Cluster import kcluster
import seaborn as sns
from DL.data_creation import filter_genes_by_variance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc
from DL.data_loading import *
from utilities.general_helpers  import *
# PICKLE_PATH = r'DATA\1-16291cells.p'
PICKLE_PATH = r'DATA\RNAseq_DATA.p'
CHECKPOINT_TSNE_PATH = r'DATA\TSNE_Embedded_1-16291cells_randInt21'  # comes into play as import OR export path.
TSNE_IMPORT_EXPORT = False  # FALSE - Import, TRUE - EXPORT




def kmeans(data, n_clusters):
    """
    Activates sklearn.kmeans.
    :param data: cells in PKL format.
    :param n_clusters: desired number of cluster for kmeans
    :return: kmeans labels - list of cluster for each cell.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)

    # print(kmeans.labels_)
    # print(kmeans.predict([[0, 0], [12, 3]]))
    # print(kmeans.cluster_centers_)
    return kmeans.labels_


def TSNE_embedded(cells):
    """
    Activates t-SNE algorithm to find 2D representation of the cells.
    calculate and export if TSNE_IMPORT_EXPORT else import from CHECKPOINT_TSNE_PATH.
    :param cells: pkl format.
    :return: embedded 2D-representation .
    """
    if TSNE_IMPORT_EXPORT:  # perform TSNE and save embedded vector in CHECKPOINT_TSNE_PATH
        cells_embedded = TSNE(n_components=2, random_state=21).fit_transform(cells)
        cells_embedded = cells_embedded.T.tolist()
        pickle.dump(cells_embedded, open(CHECKPOINT_TSNE_PATH, "wb"))
    else:
        cells_embedded = pickle.load(open(CHECKPOINT_TSNE_PATH, "rb"))
    return cells_embedded


def visualize(cells, clusters_labels, title=None, centroids=None):
    """
    Visualize 2D representation.
    :param cells: embedded cells
    :param clusters_labels: list in number of cells length indicates each cell its cluster.
    :param title: plot title
    :param centroids: of the cluster algorithm
    :return:
    """
    X = cells[0]
    Y = cells[1]

    # plt.plot(X, Y, 'ro')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'lime', 'lavender', 'darkred', 'olive']
    for cluster in np.unique(clusters_labels):
        Xi = [X[i] for i in range(len(clusters_labels)) if clusters_labels[i] == cluster]
        Yi = [Y[i] for i in range(len(clusters_labels)) if clusters_labels[i] == cluster]
        color = colors[cluster]
        plt.plot(Xi, Yi, 'ro', color=color)
    if centroids:
        for centroid in centroids:
            plt.plot(centroid[0], centroid[1], 'ro', color='y')
    # plt.plot([1, 2], [1, 4], 'ro')
    # plt.plot([3, 4], [9, 16], 'ro', color='green')
    plt.title(title)
    plt.show()


def build_confusion_matrix(classification1, classification2):
    """
    Given two different classifications of the same group (different classification lists
    with the same length), builds confusion matrix.
    :param classification1: list in length of the sample size. each place is the classification of the
     corresponding cell. for example L[i] is the classification of cell number i.
    :param classification2: the second classification list. identified to classification1 structure.
    :return: confusion matrix in DataFrame format.
    """
    match_classifications = list(zip(classification1, classification2))
    confusion_matrix = np.zeros((np.max(classification1) + 1, np.max(classification2) + 1))
    for appearance in match_classifications:
        confusion_matrix[appearance[0], appearance[1]] += 1

    columns = ['V' + str(i) for i in range(max(classification2) + 1)]
    index = ['G' + str(i) for i in range(max(classification1) + 1)]
    confusion_matrix = pd.DataFrame(confusion_matrix, columns=columns, index=index)

    return confusion_matrix


def calculate_auc(values, thresholds):
    X = np.zeros(len(thresholds))
    Y = np.zeros(len(thresholds))
    for idx, threshold in enumerate(thresholds):
        TP = sum([val[0] > threshold for val in values if val[1]])
        FP = len([val for val in values if val[1]]) - TP
        TN = sum([val[0] <= threshold for val in values if not val[1]])
        FN = len([val for val in values if not val[1]]) - TN
        X[idx] = TP / (TP + FP)
        Y[idx] = TN / (TN + FN)
    return auc(X, Y)


def correlation_response_clusters(dataset, clusters):  # cells_patients_names, response_labels, clusters):
    """
    Playground checking correlation of clusters and responders. TODO: Make some order here.
    :param cells_patients_names:
    :param response_labels:
    :param clusters:
    :return:
    """
    clusters_ratio = [[], []]  # ration between 2 clusters for responder/non-responder.
    for patient in set(dataset.cells_information_list['patient_details']):
        patient_cells_idx = [i for i in range(len(dataset)) if dataset.cells_information_list[i].patient_details == patient]
        patient_response = dataset[patient_cells_idx[0]][1].response_label
        clusters_cells_of_patient = [clusters[i] for i in patient_cells_idx]
        cluster_1_amount = sum(clusters_cells_of_patient)
        patient_ratio = cluster_1_amount / len(clusters_cells_of_patient)
        clusters_ratio[patient_response].append(patient_ratio)

    avg_1 = sum(clusters_ratio[1]) / len(clusters_ratio[1])
    avg_0 = sum(clusters_ratio[0]) / len(clusters_ratio[0])
    print(f'responder (average ration {avg_1}:')
    print(sorted(clusters_ratio[1]))
    print(f'non-responder: (average ration {avg_0}')
    print(sorted(clusters_ratio[0]))

    y_lbl = [1] * len(clusters_ratio[0]) + [0] * len(clusters_ratio[1])
    y_score = clusters_ratio[0] + clusters_ratio[1]
    print(f"op auc:{roc_auc_score(y_lbl, y_score)}")

    y_lbl = [0] * len(clusters_ratio[0]) + [1] * len(clusters_ratio[1])
    y_score = clusters_ratio[0] + clusters_ratio[1]

    return roc_auc_score(y_lbl, y_score)

    # condition = avg_1 > avg_0
    # _range = [min(clusters_ratio[1]), max(clusters_ratio[0])] if condition \
    #     else [min(clusters_ratio[0]), max(clusters_ratio[1])]
    # values = [(c, 0) for c in clusters_ratio[0]] + [(c, 1) for c in clusters_ratio[1]] if condition \
    #     else [(c, 0) for c in clusters_ratio[1]] + [(c, 1) for c in clusters_ratio[0]]
    # values = sorted(values)
    # if _range[1] < _range[0]:
    #     return 1
    # eps = 0.00005
    # # thresholds = sorted([i+eps for i in values[0] if _range[1] >= i >= _range[0]] +
    # #                     [i-eps for i in values[1] if _range[1] >= i >= _range[0]])
    # # threshold building:
    # thresholds = []
    # i = 0
    # while i < len(values):
    #     try:
    #         next_threshold_idx = [v[1] for v in values].index(1, i)
    #         thresholds.append(values[next_threshold_idx][0] - eps)
    #         i = [v[1] for v in values].index(0, next_threshold_idx)
    #     except:
    #         i = len(values)
    # return calculate_auc(values, thresholds)


def heatmap_high_epresssed_gene_of_cluster(dataset, clusters=None):
    """
    Shows Seaborn visualized heat-map of clusters of cells and their high expressed genes.
    Made by expression_of_genes function that returns the high expression genes and is activated
    for each cluster filtered by filter_by_indexes function.
    :param cells: pkl format.
    :param clusters: list in cells amount length, that tells for each cell what is its cluster.
    clusters[i] is the cluster of cell i.
    """
    indices_of_high_expressed_genes = []
    heatmap_order = []
    if clusters is None:
        clusters = dataset.patients['general_11_cluster']
    for cluster in set(clusters):
        cluster_indices = [i for i in range(len(clusters)) if cluster == clusters[i]]
        heatmap_order += cluster_indices
        cluster_dataset = dataset[cluster_indices]
        # cluster_cells[:, expression_of_genes(cluster_cells, gene_names)]
        indices_of_high_expressed_genes += cluster_dataset.get_high_expressed_genes()[0].tolist()
    # indices_of_high_expressed_genes = set(indices_of_high_expressed_genes)
    heatmap_x = cells[heatmap_order, :][:, list(indices_of_high_expressed_genes)]
    ax = sns.heatmap(heatmap_x)
    plt.show()


def article_heatmap(dataset):
    """
    taking the genes mentioned in article5 and draw heatmap of expression of those gene of the input cells
    :param dataset: cells
    """
    path = r'DATA\high expressed genes list based on article5.txt'
    with open(path, 'r') as f:
        lines = f.readlines()
    genes = [l.replace("\n", "") for l in lines if len(l) > 1]
    indices_of_high_expressed_genes = []
    li = [(i, dataset.gene_names[i]) for i in range(len(dataset.gene_names))]
    for gene in genes:
        for idx, gg in li:
            if gene == gg:
                indices_of_high_expressed_genes.append(idx)
    heatmap_order = []
    clusters = dataset.patients['general_11_cluster']
    for cluster in set(clusters):
        cluster_indices = [i for i in range(len(clusters)) if cluster == clusters[i]]
        heatmap_order += cluster_indices

    heatmap_x = cells[heatmap_order, :][:, list(indices_of_high_expressed_genes)]
    ax = sns.heatmap(heatmap_x, vmin=4, vmax=8.5, cmap="viridis")
    plt.show()


def visualization_confusion_matrix_notitle(labels, predictions):
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['non-response', 'response'])
    disp.plot(include_values=True,
              cmap='viridis', ax=None, xticks_rotation='horizontal',
              values_format=None)

    plt.show()


def main(cells, gene_names, patients_information):
    cells, gene_names = filter_genes_by_variance(cells, gene_names)
    dataset = RNAseq_Dataset(cells, patients_information, gene_names)
    # article_heatmap(dataset)
    indexes = [idx for idx in range(len(patients_information)) if patients_information[idx]['T-cell 6 cluster']]
    dataset = dataset[indexes]
    clusters, error, nfound = kcluster(dataset.cells, nclusters=2, dist='c')  # dist 'c' is pearson correlation distance

    # dataset = dataset.filter_cells_by_supervised_classification()
#    article_heatmap(dataset)

    converted_tcell_clusters = [1 if tt == 'CD8_B' else 0 for tt in dataset.cells_information_list['t_cell_2_cluster']]

    visualization_confusion_matrix(clusters, converted_tcell_clusters, 'overlap CD8_2_Cluster', display_labels=['CD8_A', 'CD8_B'])
    # print(build_confusion_matrix(clusters, converted_tcell_clusters))
    auc = correlation_response_clusters(dataset, clusters)
    print(f'current AUC: {auc}', end="\n\n\n")
    auc = correlation_response_clusters(dataset, converted_tcell_clusters)

    print(f'previous AUC: {auc}')
    _breakpoint = 0
    # cells, patients_information = filter_cells_by_supervised_classification(cells, patients_information)
    # ret = expression_of_genes(cells, gene_names)
    # [i for i in range(len(ret[1])) if ret[1][i] == 'CD28'] + ['length' + str(len(ret[1]))]
    # kmeans_clusters_euc = kmeans(cells, 2)
    # kmeans_clusters_pearson, error, nfound = kcluster(cells, nclusters=2, dist='c')   # dist 'c' is pearson correlation distance
    #
    # # some_correlation_function(patients_information, response_labels, kmeans_clusters)
    # some_correlation_function(patients_information, response_labels, kmeans_clusters_euc)
    # print(build_confusion_matrix(response_labels, kmeans_clusters_euc))


if __name__ == '__main__':
    """
    Defining T-cells states article steps.
    """
    cells, gene_names, patients_information = extract_smart_seq_data_from_pickle(PICKLE_PATH)

    main(cells, gene_names, patients_information)
    # cells, patients_information = filter_cells_by_supervised_classification(cells, patients_information)
    # expression_of_genes(cells, patients_information, gene_names)
    # It has to be decided the order of actions of tSNE and k-means.
    # embedded_cells = TSNE_embedded(cells)
    # cells = np.array(embedded_cells).T    # in order to perform cluster on embedded points.

    # keren_clusters = [p['keren cluster'] for p in patients_information]
    # response_labels = [p['response label'] for p in patients_information]
    # kmeans_clusters = cluster(cells, 2)
    #
    # clusterid, error, nfound = kcluster(cells, nclusters=2, dist='c')   # dist 'c' is pearson correlation distance
    # some_correlation_function(patients_information, response_labels, kmeans_clusters)
    # some_correlation_function(patients_information, response_labels, clusterid)
    # print(build_confusion_matrix(response_labels, clusterid))
    # heatmap(cells, keren_clusters)
    # kmeans_clusters = cluster(cells, 2)
    # print(build_confusion_matrix(response_labels, kmeans_clusters))
    #
    #
    # some_correlation_function(patients_information, response_labels, clusterid)
    # visualize(embedded_cells, keren_clusters, 'keren_clusters')
    # visualize(embedded_cells, response_labels, 'response_labels')
    # print(build_confusion_matrix(keren_clusters, clusterid))

    # Cluster algorithm
    # kmeans_clusters = cluster(cells, 2)
    # clusterid, error, nfound = kcluster(cells, nclusters=2, dist='c')   # dist 'c' is pearson correlation distance
    # clusterid2, error2, nfound2 = kcluster(cells, nclusters=2, dist='e')
    #
    # visualize(embedded_cells, kmeans_clusters, 'euc')
    # visualize(embedded_cells, clusterid, 'pearson')
    # visualize(embedded_cells, clusterid2, 'euc BIO')
    # visualize(cells, labels)

