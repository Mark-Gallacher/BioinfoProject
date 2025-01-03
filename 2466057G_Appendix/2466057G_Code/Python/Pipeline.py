from Model import Model

import pandas as pd
import re


class Pipeline():
    """ A Class which takes in a Model to extract and parse the results to a csv file.
    """
    def __init__(self, model:Model, metric_spec:dict):
        """
        - model should a Model class or FeatureElimination class
        - metric_spec should be a dictionary of metrics
            - with "name" : "metric_api"
        """
        self.model_name = model.model_name
        self.model_class = model
        self.metric_dict = self.init_metric_dict()
        self.metric_spec = metric_spec


    def init_metric_dict(self) -> dict:
        """Initialised the metric_dict by injecting the model name and all the ids (contained in a list.)
        """
        
        ## the generate_ids handles the absense of hyperparametres
        models_id = self.model_class.generate_ids()

        metric_dict = {
                "model_type" : self.model_name,
                "id" : models_id
                }

        return metric_dict

    def tidy_metric_name(self, name:str) -> str:
        """ REGEX for the metric name out from GridSearchCV.cv_result_
        Example Converts:
            - "split0_test_f1_micro" --> "f1_micro"
            - "test_f1_macro" --> "f1_macro"
        """
        search = re.search("(test_)(.*)", name)

        if search is not None:
            return search.group(2)
        
        return None

    def merge_metric(self, name:str, values:list) -> dict:
        """ Merges multiple metrics into one dictionary whilst appending if the metric is already present.
        """
        if name in self.metric_dict.keys():
            self.metric_dict[name] += values

        else:
            self.metric_dict[name] = values

        return self.metric_dict

    def parse_cv_results(self, results:dict) -> dict:
        """ Extracts the Metrics from the cv_results_ dictionary. 
        Transposes the lists to output the form 
            - "metric" : [model1, model2, model3, ...]
            - for all metrics supplied
        """
        # print(results)
        for key in results.keys():
            if key.startswith("split") | key.startswith("test"):
                    
                ## get the values then tidy the name of the metric
                metric_name = self.tidy_metric_name(key)

                if metric_name is None:
                    raise ValueError(f"No Metrics were found in this Results Dictionary with name {metric_name}!!")

                ## extract the values if we know there is some metricx info.
                values = results[key].tolist()
                
                ## generate a dictionary containing all the metrics
                self.metric_dict = self.merge_metric(metric_name, values)

        return self.metric_dict

    def run_cv(self, X, y) -> dict:
        """Runs the GridSearchCV or RFECV defined in the Model class.
        Returns the dictionary of cv_results_

        This has a side effect of updating the values inside metric_dict["id"]
        """
        try:
            _cv = self.model_class.cross_validate(
                        X = X, 
                        y = y, 
                        metrics = self.metric_spec
                        )

        except Exception as e:
        
            print(f"Issue doing cross validation! \nModel: {self.model_name} \n - Error: {e}")
            raise SystemExit(1)            

        ## Update the Metric Dictionary
        self.metric_dict["id"] = self.model_class.generate_ids()

        
        return _cv

    def generate_metric_dataframe(self, X, y) -> pd.DataFrame:
        """ Generates the Pandas.DataFrame after running the GridSearchCV. 
        With details of the model(s) and the metric(s) in the column.
        """
        ## generate the results, so we can extract the relevant values later
        _results = self.run_cv(X, y)

        ## the output is a dictionary, stored in self.metric_dict
        self.parse_cv_results(_results)

        # print(self.metric_dict)

        df = pd.DataFrame(self.metric_dict)

        return df

    def save_as_csv(self, df:pd.DataFrame, folder:str):
        """ Saves Pandas.DataFrame as a csv file in defined path.  
        Puts a csv of the results with the path:
        - <folder> / <name of model>.csv

        i.e.
        - ~/project/data/LogisticRegression.csv
        """
        ## check if the path already indicates a folder
        if folder.endswith("/"):
            folder = folder.rstrip("/")


        try: 

            df.to_csv(f"{folder}/{self.model_name}.csv", index = False)

        except Exception as e:

            raise SystemError(f"File was not found - please check the path: {folder} \nModel: {self.model_name} \n- Error: {e}")





