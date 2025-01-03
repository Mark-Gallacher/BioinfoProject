from Pipeline import Pipeline
from Model import Model
from Model import Hyperparametres
from HelperFunctions import expand_metrics

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.dummy import DummyClassifier

from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

import os

threads = os.cpu_count()

if threads is None:
    threads = 1

print()
print(f"There appear to be {threads} threads available!!")
print()

#### Loading in the Data ####
_raw_data = pd.read_csv("../data/TidyData.csv")
raw_data = _raw_data.drop(["DiseaseSubtype", "PseudoID"], axis = 1)

sss = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 1)


for train_i, test_i in sss.split(raw_data,  raw_data["DiseaseSubtypeFull"]):
    train_set = raw_data.loc[train_i]
    test_set = raw_data.loc[test_i]

_unscaled_train_data = train_set.drop("DiseaseSubtypeFull", axis = 1)
train_labels = train_set["DiseaseSubtypeFull"].copy()

scaler = StandardScaler()

columns = _unscaled_train_data.columns
train_data = scaler.fit_transform(_unscaled_train_data[columns])

# print(train_labels.value_counts(), train_labels.value_counts()/ len(train_labels))

#### Metrics ####
base_metrics = ["precision", "recall", "accuracy", "balanced_accuracy", "f1", "fbeta"]

metrics = expand_metrics(base_metrics)

#### Other Global Params ####
folds = 10
metrics_output_folder = "../data/metrics/"
params_output_folder = "../data/params/"
#threads = 10

#### LogisticRegression ####
## penalty was None but that generated an error??
log_reg_params = Hyperparametres(
                model_name = "LogisticRegression", 
                model_code = "LG",
                params = [ 
                {"penalty" : [ "none" ]},
                {"penalty" : ["l1", "l2"], 
                "C" : [1, 2, 4, 8, 16]},
                {"penalty" : ["elasticnet"], 
                 "C" : [1, 2, 4, 8, 16], 
                 "l1_ratio" : [.2, .4, .6, .8]}
                ])

## Define the Type of Model with the Hyperparametres
log_reg_model = Model(model = LogisticRegression,
                params = log_reg_params,
                solver = "saga",
                max_iter = 5000, 
                n_jobs = threads,
                folds = folds)

#### K-Nearest Neighbours ####
knn_params = Hyperparametres(
            model_name = "KNearestNeighbours", 
            model_code = "KNN", 
            params = {
            "n_neighbors" : [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40], 
            "weights" : ["uniform", "distance"]
            })

knn_model = Model(model = KNeighborsClassifier, 
                  params = knn_params,
                  n_jobs = threads,
                  folds = folds)

#### Naive Bayes ####
gnb_params = Hyperparametres(params = {}, 
                             model_name = "GaussianNaiveBayes", 
                             model_code = "GNB")


gnb_model = Model(model = GaussianNB, 
                  params = gnb_params, 
                  n_jobs = threads,
                  folds = folds)

#### RandomForest ####
rf_params = Hyperparametres(
        model_name = "RandomForest", 
        model_code = "RF", 
        params = { 
        "n_estimators" : [10, 25, 50, 75, 100, 125, 150, 200, 250], 
        # "n_estimators" : [10, 100], 
        "min_samples_split" : [2, 4, 6, 8, 10, 12, 16]
        # "min_samples_split" : [2, 4]
            })

rf_model = Model(model = RandomForestClassifier, 
                 params = rf_params, 
                 n_jobs = threads, 
                 folds = folds, 
                 min_samples_leaf = 1, 
                 max_features = "sqrt")

#### GradientBoosting Trees ####

gb_params = Hyperparametres(
        model_name = "GradientBoosting", 
        model_code = "GB", 
        params = {
        "learning_rate" : [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25], 
        # "learning_rate" : [0.01, 0.1], 
        "n_estimators" : [10, 25, 50, 75, 100, 125, 150, 200, 250] 
        # "n_estimators" : [10, 100], 
            })

gb_model = Model(model = GradientBoostingClassifier, 
                 params = gb_params, 
                 n_jobs = threads, 
                 folds = folds, 
                 max_features = "sqrt", 
                 min_samples_leaf = 1
                 )

#### SVM ####

svc_params = Hyperparametres(
         model_name = "SupportVectorMachine", 
         model_code = "SVM", 
         params = [
            {"kernel" : ["linear"],
             "C" : [1, 2, 4, 8, 16]},
 
             {"kernel": ["rbf", "poly"],
             "C" : [1, 2, 4, 8, 16], 
             "gamma" : ["scale", "auto"]}
            ])

svc_model = Model(
         model = SVC, 
         params = svc_params, 
         n_jobs = threads, 
         folds = folds, 
         max_iter = 10000, 
         cache_size = 2000, 
         tol = 1e-4)

svc_params_2 = Hyperparametres(
            model_name = "LinearSVC", 
            model_code = "LSCM", 
            params = {
            "penalty" : ["l2"], 
            "loss" : ["hinge", "squared_hinge"],
             "C" : [1, 2, 4, 8, 16], 
        })

svc_model_2 = Model(
            model = LinearSVC, 
            params = svc_params_2, 
            n_jobs = threads, 
            folds = folds, 
            max_iter = 10000, 
            dual = True, 
            tol = 1e-4
            )

#### Dummy Classifier
dummy_params = Hyperparametres(model_name = "Dummy", 
                               model_code= "Dum", 
                               params= {})

dummy_model = Model(model = DummyClassifier,
                    params = dummy_params,
                    n_jobs = threads, 
                    folds = folds,
                    strategy = "most_frequent")

#### Collection of Models

model_collection = [svc_model, svc_model_2]

param_collection = [svc_params, svc_params_2]

#### Run the Pipeline ####

for model in param_collection:

    model.save_as_csv(folder = params_output_folder)

for model in model_collection:

    pipeline = Pipeline(model, metrics)

    df = pipeline.generate_metric_dataframe(X = train_data, y = train_labels)

    pipeline.save_as_csv(df, metrics_output_folder)


