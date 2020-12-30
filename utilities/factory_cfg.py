import os
import itertools
import yaml


class factory_cfg:
    def __init__(self):
        variance = [i for i in range(0, 10, 2)]
        test_percent = [i*0.1 for i in range(1, 8)]
        patients = ['ALL', 'pre', 'post']
        k_folds = [i for i in range(2, 7)]
        num_round = [i for i in range(2, 12)]
        early_stopping_rounds = [i for i in range(2, 12)]
        # self.a = [variance, test_percent, patients, k_folds, num_round, early_stopping_rounds]
        self.triplets = [variance, test_percent, patients]

    def create(self, folder):
        k_fold = 5
        num_round = 10
        early_stopping_rounds = 10
        combinations = list(itertools.product(*self.triplets))
        for combination in combinations:
            variance, test_percent, patients = combination[0], combination[1], combination[2]
            experiment_name = "variance_"+str(variance)+"_test_percent"+str(test_percent)+"_patients_"+patients+""
            path = os.path.join(folder, experiment_name)
            dict_file = {'EXPERIMENT': {"experiment_name": experiment_name,
                                        "experiments_folder": r"DATA\Experiments\factory_experiments"},
                         'DATASET': {"test_percent": test_percent,
                                     "variance": variance,
                                     "patients": patients},
                         'XGBOOST': {'k_folds': k_fold,
                                     'num_round': num_round,
                                     'early_stopping_rounds': early_stopping_rounds}}
            with open(path+'_cfg.yaml', 'w') as file:
                documents = yaml.dump(dict_file, file)



if __name__ == '__main__':
    path = r"cfg\factory_cfg"
    # factory_cfg().create(path)
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for file in onlyfiles:
        print(file)

