from Model import Model

import pandas as pd
import re


class Pipeline():
    """ A Class which takes in a Model to extract and parse the results to a csv file.
    """
    def __init__(self, model:Model, metric_spec:dict):
        """
        - model should a Model class
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
        model_ids = self.model_class.generate_ids()

        metric_dict = {
                "model_type" : self.model_name,
                "id" : model_ids
                }

        return metric_dict

    def tidy_metric_name(self, name:str) -> str | None:
        """ REGEX for the metric name out from GridSearchCV.cv_result_
        Example Converts:
            - "split0_test_f1_micro" --> "f1_micro"
        """
        search = re.search("(split[0-9]+_test_)(.*)", name)

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
        for key in results.keys():
            if key.startswith("split"):
                    
                ## get the values then tidy the name of the metric
                metric_name = self.tidy_metric_name(key)

                if metric_name is None:
                    raise ValueError("No Metrics were found in this Results Dictionary!!")

                ## extract the values if we know there is some metricx info.
                values = results[key].tolist()
                
                ## generate a dictionary containing all the metrics
                self.metric_dict = self.merge_metric(metric_name, values)

        return self.metric_dict

    def run_gridsearch(self, X, y) -> dict:
        """Runs the GridSearchCV defined in the Model class.
        Returns the dictionary of cv_results_
        """
        gridsearch = self.model_class.cross_validate(
                        X = X, 
                        y = y, 
                        metrics = self.metric_spec
                        )
        
        return gridsearch.cv_results_

    def generate_metric_dataframe(self, X, y) -> pd.DataFrame:
        """ Generates the Pandas.DataFrame after running the GridSearchCV. 
        With details of the model(s) and the metric(s) in the column.
        """
        _results = self.run_gridsearch(X, y)

        self.parse_cv_results(_results)

        df = pd.DataFrame(self.metric_dict)

        return df

    def save_as_csv(self, df:pd.DataFrame, folder:str):
        """ Saves Pandas.DataFrame as a csv file in defined path.  
        Puts a csv of the results with the path:
        - <folder> / <name of model>.csv

        i.e.
        - ~/project/data/LogisticRegression.csv
        """
        
        df.to_csv(f"{folder}/{self.model_name}.csv", index = False)



        







