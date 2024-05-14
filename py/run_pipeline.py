from Pipeline import Pipeline
from Model import Model
from Model import Hyperparametres
from HelperFunctions import expand_metrics

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from HelperFunctions import expand_metrics

#### Loading in the Data ####
_raw_data = pd.read_csv("../data/TidyData.csv")
raw_data = _raw_data.drop(["DiseaseSubtype", "PseudoID"], axis = 1)

sss = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 1)


for train_i, test_i in sss.split(raw_data,  raw_data["DiseaseSubtypeFull"]):
    train_set = raw_data.loc[train_i]
    test_set = raw_data.loc[test_i]

train_data = train_set.drop("DiseaseSubtypeFull", axis = 1)
train_labels = train_set["DiseaseSubtypeFull"].copy()

#### Metrics ####
# Key is my name, value is their API
# metrics = {
        # "precision", 
        # "recall", 
        # "accuracy",
        # "balanced_accuracy", 
        # "f1_micro" : "f1_micro", 
        # "f1_macro" : "f1_macro", 
        # "f1_weighted" : "f1_weighted"
        # }

base_metrics = ["precision", "recall", "accuracy", "balanced_accuracy", "f1"]

metrics = expand_metrics(base_metrics)

#### Other Global Params ####
folds = 3
metrics_output_folder = "../data/metrics/"

#### LogisticRegression ####
log_reg_params = Hyperparametres(params = {
                "penalty" : ["l1", "l2", None]
                })

## Define the Type of Model with the Hyperparametres
log_reg_model = Model(name = "LogisticRegression",
                code = "LG",
                model = LogisticRegression,
                params = log_reg_params,
                solver = "saga",
                max_iter = 1000, 
                folds = folds)

#### K-Nearest Neighbours ####
knn_params = Hyperparametres(params = {
            "n_neighbors" : [3, 5, 8, 12]})

knn_model = Model(name = "KNearestNeighbours", 
                  code = "KNN", 
                  model = KNeighborsClassifier, 
                  params = knn_params,
                  folds = folds)

#### Collection of Models

collection = [log_reg_model, knn_model]

#### Run the Pipeline ####

for model in collection:


    pipeline = Pipeline(model, metrics)

    df = pipeline.generate_metric_dataframe(X = train_data, y = train_labels)

    pipeline.save_as_csv(df, metrics_output_folder)


