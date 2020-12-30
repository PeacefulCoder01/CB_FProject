import os
import numpy as np
import yaml
import os
import pandas
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
from shutil import copyfile

def flatten_list(l):
    return [item for sublist in l for item in sublist]


def plot_counter_list(l):
    letter_counts = Counter(l)
    df = pandas.DataFrame.from_dict(letter_counts, orient='index')
    df.plot(kind='bar')


def search_in_list(count_list, key):
    d = {v[0]: v[1] for v in count_list}
    return d.get(key, 0)


def is_there_overlap_in_lists(l1 , l2):
    return len([f for f in l1 if f in l2])!=0


def intersection_of_lists(l1, l2):
    """
    Returns unique appearance of each value of both lists.
    :param l1: list
    :param l2: list
    """
    s1 = list(set(l1))
    s2 = list(set(l2))
    inter_l = [v for v in s1 if v in s2]
    return inter_l


def avg(l):
    return sum(l)/len(l)


class Experiments_manager:
    def __init__(self, experiment_name, experiment_folder, config_path=None):
        # Builds experiment folder
        exp_folder_path = os.path.join(experiment_folder, experiment_name)
        if not os.path.isdir(exp_folder_path):
            os.mkdir(exp_folder_path)
        # copy config
        if config_path:
            copyfile(config_path, os.path.join(exp_folder_path, 'config.yaml'))
        self.print_file_path = os.path.join(exp_folder_path, 'prints.txt')
        self.out_file = open(self.print_file_path, 'w')
        self.orig_stdout = None

    def activate_prints_to_file(self):
        self.orig_stdout = sys.stdout
        sys.stdout = self.out_file
        return self

    def finish_run(self):
        sys.stdout = self.orig_stdout
        self.out_file.close()
        with open(self.print_file_path, 'r+') as f:
            lines = f.readlines()
            for line in lines:
                print(line, end='')

    def print(self, txt, end='\n'):
        print(txt, end=end)
        self.out_file.write(txt+end)


def experiment_manager(experiment_name, experiment_folder, experiment_config=None):
    def experiment_manager_wrapper(func):
        def inner(*args, **kwargs):
            print(f"Experiment \'{experiment_name}\' has started, prints will be saved in \'{experiment_folder}\'")
            em = Experiments_manager(experiment_name, experiment_folder, experiment_config).activate_prints_to_file()
            output = func(*args, **kwargs)
            em.finish_run()
            return output
        return inner
    return experiment_manager_wrapper


def load_yml(yml_path):
    with open(yml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['EXPERIMENT']['experiment_name'], config['EXPERIMENT']['experiments_folder'], config


def binary_search(lst, target):
    def _binary_search_loop(lst, i, j, target):
        avg = int((i + j) / 2)
        if j <= i:
            return -1
        if lst[avg] == target:
            return avg
        elif lst[avg] > target:
            return _binary_search_loop(lst, i, avg, target)
        else:
            return _binary_search_loop(lst, avg + 1, j, target)
    i = 0
    j = len(lst)
    return _binary_search_loop(lst, i, j, target)


def visualization_confusion_matrix(labels, predictions, title=None, save_path=None, display_labels=None):
    cm = confusion_matrix(labels, predictions)
    if display_labels:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=display_labels)
    else:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=['non-response', 'response'])
    import matplotlib
    _, ax = plt.subplots()
    if title:
        ax.set_title(title)
    disp.plot(include_values=True,
              cmap='viridis', ax=ax, xticks_rotation='horizontal',
              values_format=None)
    if save_path:
        plt.savefig(os.path.join(save_path+".png"))
    plt.show()

    return cm


def create_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)