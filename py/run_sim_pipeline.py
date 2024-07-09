
## My Classes
from Pipeline import Pipeline
from Model import Model
from Model import Hyperparametres
from Metrics import Metric, ConfusionMetrics

## Models
from sklearn.dummy import DummyClassifier

## sklearn utils
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

## misc modules
import pandas as pd
import os



labels = [1, 0]

mode = "sim"
input_data = "../data/sim_data"
print(f"Using data from the folder: {input_data}\n")

metrics_output_folder = f"../data/{mode}/metrics/"
print(f"output metric data to: {metrics_output_folder}")


##### ~~~~~~~~~~~~~~~ #####
##### Run Main script #####
##### ~~~~~~~~~~~~~~~ #####


threads = os.cpu_count()

if threads is None:
    threads = 1

print(f"There appear to be {threads} threads available!!\n")






##### ~~~~~~~~~~~~~~~~~~~~ #####
##### Defining the Metrics #####
##### ~~~~~~~~~~~~~~~~~~~~ #####


#### Metrics ####
base_metrics = ["precision", "recall", "accuracy", "f1", "fbeta", "cohen_kappa", "matthew_coef"]

## get the standard metrics - listed above
metrics = {}
for base_metric in base_metrics:
    ## for each type of metrics, generate the scorer function(s)
    ## ensure all have the same labels to explicitly control order values
    ## this is more importance for ConfusionMetrics though
    metric = Metric(base_metric, labels = labels, is_binary = True)
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
##### Running the Pipeline #####
##### ~~~~~~~~~~~~~~~~~~~~ #####

if __name__ == "__main__":

    ## os.walk returns current path, directories and files
    for root, dirs, files in os.walk(input_data):
        for file in files:
            if file.endswith(".csv"):
            
                filename = f"{input_data}/{file}"
                file = file.rstrip(".csv")

                print(file)

                ##### ~~~~~~~~~~~~~~~~ #####
                ##### Parsing the Data #####
                ##### ~~~~~~~~~~~~~~~~ #####

                _raw_data = pd.read_csv(filename)
                raw_data = _raw_data.drop(["DiseaseSubtype"], axis = 1)

                raw_data["DiseaseSubtypeFull"] = raw_data["DiseaseSubtypeFull"].map({"PHT" : 1, "PA" : 0})

                _unscaled_data = raw_data.drop("DiseaseSubtypeFull", axis = 1)
                labels = raw_data["DiseaseSubtypeFull"].copy()

                ### Scale the Features - Only looking at the training data for now, not the testing set
                scaler = StandardScaler()
                columns = _unscaled_data.columns
                data = scaler.fit_transform(_unscaled_data[columns])



                #### Dummy Classifier
                dummy_params = Hyperparametres(model_name = file, 
                                               model_code= "Dum", 
                                               params= {})

                dummy_model = Model(model = DummyClassifier,
                                    params = dummy_params,
                                    n_jobs = threads, 
                                    folds = 2,
                                    strategy = "most_frequent")


                
                pipeline = Pipeline(dummy_model, metrics)

                df = pipeline.generate_metric_dataframe(X = data, y = labels)

                pipeline.save_as_csv(df, metrics_output_folder)


print("End of Python Script\n")

