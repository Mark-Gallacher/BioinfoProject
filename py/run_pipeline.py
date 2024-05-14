from Pipeline import Pipeline
from Model import Model
from Model import Hyperparametres
from HelperFunctions import expand_metrics

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from HelperFunctions import expand_metrics

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
base_metrics = ["precision", "recall", "accuracy", "balanced_accuracy", "f1"]

metrics = expand_metrics(base_metrics)

#### Other Global Params ####
folds = 10
metrics_output_folder = "../data/metrics/"

#### LogisticRegression ####
log_reg_params = Hyperparametres(params = {
                "penalty" : ["l1", "l2", None], 
                "C" : [.5, 1, 3, 9]
                })

## Define the Type of Model with the Hyperparametres
log_reg_model = Model(name = "LogisticRegression",
                code = "LG",
                model = LogisticRegression,
                params = log_reg_params,
                solver = "saga",
                max_iter = 5000, 
                folds = folds)

#### K-Nearest Neighbours ####
knn_params = Hyperparametres(params = {
            "n_neighbors" : [3, 5, 8, 12, 20], 
            "weights" : ["uniform", "distance"]})

knn_model = Model(name = "KNearestNeighbours", 
                  code = "KNN", 
                  model = KNeighborsClassifier, 
                  params = knn_params,
                  folds = folds)

#### Naive Bayes ####
gnb_params = Hyperparametres(params = {})
# cnb_params = Hyperparametres(params = {
            # "alpha" : [0.25, .5, 1, 2]}
                             # )


gnb_model = Model(name = "GaussianNaiveBayes", 
                  code = "GNB", 
                  model = GaussianNB, 
                  params = gnb_params, 
                  folds = folds)

# cnb_model = Model(name = "ComplementNaiveBayes", 
                  # code = "CNB", 
                  # model = ComplementNB, 
                  # params = cnb_params, 
                  # folds = folds)



#### Collection of Models

collection = [log_reg_model, knn_model, gnb_model]

#### Run the Pipeline ####

for model in collection:


    pipeline = Pipeline(model, metrics)

    df = pipeline.generate_metric_dataframe(X = train_data, y = train_labels)

    pipeline.save_as_csv(df, metrics_output_folder)


