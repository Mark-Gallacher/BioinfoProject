## My Classes
from Pipeline import Pipeline
from Model import Model
from Model import Hyperparameters
from Metrics import Metric, ConfusionMetrics

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.dummy import DummyClassifier

## sklearn utils
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

## misc modules
import pandas as pd
import argparse
import os

##### ~~~~~~~~~~~~~~~~~~~ #####
##### Setting up Argparse #####
##### ~~~~~~~~~~~~~~~~~~~ #####

input_parse = argparse.ArgumentParser(
        prog = "microRNA-ML-Pipeline", 
        description = "Requires one argument, which must be full, subtypes or feature!")

input_parse.add_argument("-m", "--mode", 
                         required = True, 
                         dest = "mode", 
                         type = str)

args = input_parse.parse_args()

mode = args.mode # full / subtypes / feature - what type of data are we passing to the model


## using the data from RFE
if mode == "feature" :
    input_data = "../data/feature_selection/GradientBoostingRFE_Features.csv"
    labels = ["CS", "PA", "PHT", "PPGL"]


## using the full dataset - including healthy controls
elif mode == "full" :
    input_data = "../data/TidyData.csv"
    labels = ["HV", "CS", "PA", "PHT", "PPGL"]


## using just the data on the hypertensive patients
elif mode == "subtypes":
    input_data = "../data/SubTypeData.csv"
    labels = ["CS", "PA", "PHT", "PPGL"]

## unsupported mode / typo
else:
    raise SystemError(f"The mode of analysis is not supported - received {mode}, expected feature, full or subtypes")


print(f"Using data from the folder: {input_data}\n")

metrics_output_folder = f"../data/{mode}/metrics/"
params_output_folder = f"../data/{mode}/params/"
confusion_matrix_output_folder = f"../data/{mode}/conf_mat/"

print(f"output metric data to: {metrics_output_folder}")
print(f"output params data to: {params_output_folder}\n")


##### ~~~~~~~~~~~~~~~ #####
##### Run Main script #####
##### ~~~~~~~~~~~~~~~ #####


threads = os.cpu_count()

if threads is None:
    threads = 1

print(f"There appear to be {threads} threads available!!\n")



#### Other Global Params ####
num_folds = 5
print(f"Using K-fold Cross Validation with a K of {num_folds}\n")



##### ~~~~~~~~~~~~~~~~ #####
##### Parsing the Data #####
##### ~~~~~~~~~~~~~~~~ #####

_raw_data = pd.read_csv(input_data)
raw_data = _raw_data.drop(["DiseaseSubtype", "PseudoID"], axis = 1)

### Explicitly controlling the folds, so they are the same across all models
folds = StratifiedKFold(n_splits = num_folds, random_state = 1, shuffle = True)

### Creating the test and train sets
sss = StratifiedShuffleSplit(n_splits = 1, test_size = .2, random_state = 1)

for train_i, test_i in sss.split(raw_data,  raw_data["DiseaseSubtypeFull"]):
    train_set = raw_data.loc[train_i]
    test_set = raw_data.loc[test_i]

_unscaled_train_data = train_set.drop("DiseaseSubtypeFull", axis = 1)
train_labels = train_set["DiseaseSubtypeFull"].copy()

### Scale the Features - Only looking at the training data for now, not the testing set
scaler = StandardScaler()
columns = _unscaled_train_data.columns
train_data = scaler.fit_transform(_unscaled_train_data[columns])



##### ~~~~~~~~~~~~~~~~~~~~ #####
##### Defining the Metrics #####
##### ~~~~~~~~~~~~~~~~~~~~ #####


#### Metrics ####
base_metrics = ["precision", "recall", "accuracy", "balanced_accuracy", "f1", "fbeta", "cohen_kappa", "matthew_coef"]

## get the standard metrics - listed above
metrics = {}
for base_metric in base_metrics:
    ## for each type of metrics, generate the scorer function(s)
    ## ensure all have the same labels to explicitly control order values
    ## this is more importance for ConfusionMetrics though
    metric = Metric(base_metric, labels = labels)
    metric.generate_scorer_func()

    ## some metrics return multiple scorers - ie macro, micro and weighted
    for key, value in metric.scorer_func.items():
        metrics[key] = value

## Get the TP, FP, TN and FN from the confusion matrix
confusion_metrics = ConfusionMetrics(labels = labels)
confusion_scorers = confusion_metrics.generate_scorers() 

## Both confusion_scorers and metrics are dictionarys, in format 
## - { name : scorer_function }
## let's merge the two dictionary so we don't need to run GridSearchCV twice.
metrics.update(confusion_scorers)

##### ~~~~~~~~~~~~~~~~~~~~ #####
##### Defining some params #####
##### ~~~~~~~~~~~~~~~~~~~~ #####

## for Linear models - i.e Logistic Regression and SVM
c_values = [0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5, 1, 3, 10, 30, 100]

## for RandomForest and Gradient Boosted Trees
estimators = [50, 100, 500, 1000]
min_samples_splits = [5, 8, 10, 20, 40]
min_samples_leaf = [1, 4, 8]
max_features = [None, "sqrt"]
max_depth = [None, 5, 10, 30]
criterion = ["gini", "entropy"]
learning_rate = [0.05, 0.1, 0.2]

## 1,440 models for GB
## 1,920 models for RF

## for K-nearest Neighbours
n_neighbors = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30, 35, 40, 45, 50]

##### ~~~~~~~~~~~~~~~~~~~~~ #####
##### Setting up the Models #####
##### ~~~~~~~~~~~~~~~~~~~~~ #####

#### LogisticRegression ####
## penalty was None but that generated an error??
log_reg_params = Hyperparameters(
                model_name = "LogisticRegression", 
                model_code = "LG",
                params = [ 
                {"penalty" : [ "none" ]},
                {"penalty" : ["l1", "l2"], 
                "C" : c_values},
                {"penalty" : ["elasticnet"], 
                "C" : c_values, 
                 "l1_ratio" : [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]}
                ])

## Define the Type of Model with the Hyperparameters
log_reg_model = Model(model = LogisticRegression,
                params = log_reg_params,
                solver = "saga",
                max_iter = 5000, 
                n_jobs = threads,
                folds = folds)



#### K-Nearest Neighbours ####
knn_params = Hyperparameters(
            model_name = "KNearestNeighbours", 
            model_code = "KNN", 
            params = {
            "n_neighbors" : n_neighbors, 
            "weights" : ["uniform", "distance"]
            })

knn_model = Model(model = KNeighborsClassifier, 
                  params = knn_params,
                  n_jobs = threads,
                  folds = folds)



#### Naive Bayes ####
gnb_params = Hyperparameters(params = {}, 
                             model_name = "GaussianNaiveBayes", 
                             model_code = "GNB")


gnb_model = Model(model = GaussianNB, 
                  params = gnb_params, 
                  n_jobs = threads,
                  folds = folds)



#### RandomForest ####
rf_params = Hyperparameters(
        model_name = "RandomForest", 
        model_code = "RF", 
        params = { 
        "n_estimators" : estimators, 
        "min_samples_split" : min_samples_splits,
        "min_samples_leaf" : min_samples_leaf,
        "max_features" : max_features,
        "max_depth" : max_depth, 
        "criterion": criterion,
        "class_weight" : [None, "balanced"]
            })



rf_model = Model(model = RandomForestClassifier, 
                 params = rf_params, 
                 n_jobs = threads, 
                 folds = folds)


#### ExtraTress ####
erf_params = Hyperparameters(
        model_name = "ExtraRandomForest", 
        model_code = "ERF", 
        params = { 
        "n_estimators" : estimators, 
        "min_samples_split" : min_samples_splits,
        "min_samples_leaf" : min_samples_leaf,
        "max_features" : max_features,
        "max_depth" : max_depth, 
        "criterion": criterion,
        "class_weight" : [None, "balanced"]
            })

erf_model = Model(model = ExtraTreesClassifier,
                  params= erf_params, 
                  n_jobs = threads, 
                  folds = folds)


#### GradientBoosting Trees ####
gb_params = Hyperparameters(
        model_name = "GradientBoosting", 
        model_code = "GB", 
        params = {
        "learning_rate" : learning_rate, 
        "n_estimators" : estimators, 
        "min_samples_split" : min_samples_splits,
        "min_samples_leaf" : min_samples_leaf,
        "max_features" : max_features,
        "max_depth" : max_depth
            })

gb_model = Model(model = GradientBoostingClassifier, 
                 params = gb_params, 
                 n_jobs = threads, 
                 folds = folds, 
                 n_iter_no_change = 10 
                 )



#### SVM ####
svc_params = Hyperparameters(
         model_name = "SupportVectorMachine", 
         model_code = "SVM", 
         params = [
            {"kernel" : ["linear"],
             "C" : c_values, 
             "class_weight" : [None, "balanced"]}, 
             {"kernel": ["rbf", "sigmoid"],
             "C" : c_values, 
             "gamma" : ["scale", "auto"], 
             "class_weight" : [None, "balanced"]}, 
             {"kernel": ["poly"], 
              "degree": [2, 3, 4], 
             "C" : c_values, 
             "gamma" : ["scale", "auto"],
             "class_weight" : [None, "balanced"]} 
            ])

svc_model = Model(
         model = SVC, 
         params = svc_params, 
         n_jobs = threads, 
         folds = folds, 
         max_iter = 10000, 
         cache_size = 2000, 
         tol = 1e-4)

svc_params_2 = Hyperparameters(
            model_name = "LinearSVC", 
            model_code = "LSVM", 
            params = [
                {"penalty" : ["l2"], 
                 "loss" : ["hinge"],
                 "C" : c_values, 
                 "dual" : [True], 
                 "class_weight" : [None, "balanced"]}, 
                {"penalty" : ["l1", "l2"], 
                 "loss" : ["squared_hinge"],
                 "C" : c_values, 
                 "dual" : [False], 
                 "class_weight" : [None, "balanced"]}
                ])

svc_model_2 = Model(
            model = LinearSVC, 
            params = svc_params_2, 
            n_jobs = threads, 
            folds = folds, 
            max_iter = 10000, 
            tol = 1e-3)


#### Dummy Classifier
dummy_params = Hyperparameters(model_name = "Dummy", 
                               model_code= "Dum", 
                               params= {})

dummy_model = Model(model = DummyClassifier,
                    params = dummy_params,
                    n_jobs = threads, 
                    folds = folds,
                    strategy = "most_frequent")



#### Collection of Models
model_collection = [log_reg_model, knn_model, gnb_model, 
                    rf_model, erf_model, gb_model, 
                    svc_model, svc_model_2,
                    dummy_model]

param_collection = [log_reg_params, knn_params, gnb_params, 
                    rf_params, erf_params, gb_params, 
                    svc_params, svc_params_2,
                    dummy_params]



##### ~~~~~~~~~~~~~~~~~~~~ #####
##### Running the Pipeline #####
##### ~~~~~~~~~~~~~~~~~~~~ #####

if __name__ == "__main__":

    for param in param_collection:

        param.save_as_csv(folder = params_output_folder)

    for model in model_collection:

        ## with the standard metrics
        pipeline = Pipeline(model, metrics)

        df = pipeline.generate_metric_dataframe(X = train_data, y = train_labels)

        pipeline.save_as_csv(df, metrics_output_folder)


print("End of Python Script\n")

