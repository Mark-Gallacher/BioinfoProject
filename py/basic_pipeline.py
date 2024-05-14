from collections.abc import KeysView
from Model import Hyperparametres, Model
from HelperFunctions import parse_cv_results, tidy_metric_name, merge_metric

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.preprocessing import OneHotEncoder

import pandas as pd

# Key is my name, value is their API
metrics = {"f1_micro" : "f1_micro", 
           "f1_macro" : "f1_macro", 
           "f1_weighted" : "f1_weighted"}

folds = 5

# read in the data
raw_data = pd.read_csv("../data/TidyData.csv").drop(["DiseaseSubtype", "PseudoID"], axis = 1)

# split the data into train and test
# then remove the labels
sss = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 1)

for train_i, test_i in sss.split(raw_data,  raw_data["DiseaseSubtypeFull"]):
    train_set = raw_data.loc[train_i]
    test_set = raw_data.loc[test_i]

train_data = train_set.drop("DiseaseSubtypeFull", axis = 1)
train_labels = train_set["DiseaseSubtypeFull"].copy()

## encode the catagorical variable. This might be unneeded since it is classification
# ohe = OneHotEncoder()
# disease_labels = ohe.fit_transform(train_labels) 


log_reg_params = Hyperparametres(params = {
                "penalty" : ["l1", "l2", None]
})

## Define the Type of Model with the Hyperparametres
log_reg = Model(name = "LogisticRegression",
                code = "LG",
                model = LogisticRegression,
                params = log_reg_params,
                solver = "saga",
                max_iter = 1000, 
                folds = folds)

## Perform the Cross-Validation
cv = log_reg.cross_validate(
        X = train_data, 
        y = train_labels, 
        metrics = metrics)

## Obtain the result dictionary to parse
result = cv.cv_results_

## The base dictionary will have the indices of the models - that will be added
model_id = log_reg.generate_ids()

metric_dict = {
        "model_type" : log_reg.model_name,
        "id" : model_id
        }

parse_cv_results(result, metric_dict)

log_reg_df = pd.DataFrame(metric_dict)

log_reg_df.to_csv(f"../data/{log_reg.model_name}.csv", index = False)


# print(log_reg_df)
# print()
