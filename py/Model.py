from sklearn.base import is_classifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid

import pandas as pd

## A class for the hyperparametres of a model, which is basically a dictionary
## this allows it to integrate with the sci-kit API down the line but allows 
## us to keep track of what parameters we could change
class Hyperparametres():

    def __init__(self, model_name:str, model_code:str, params:dict )-> None:
        
        if len(model_name) < 3 :
            raise ValueError(f"Please Supply a more useful name than {model_name}, for example - LogisticRegression")

        if len(model_code) < 2 :
            raise ValueError(f"Please supply a more useful code than {model_code}, for example - LG for Logistic Regression")

        self.params = params
        self.model_code = model_code.upper()
        self.model_name = model_name 
        self.grid = self.create_grid()
        self.param_ids = self.create_ids()

    def create_grid(self) -> list:

        ## if the dictionary has content
        if self.params:
            grid = ParameterGrid(self.params)
            grid = list(grid)

        ## if the dictionary is empty
        else:

            grid = [{"NA" : "NA"}]

        return grid

    def create_ids(self) -> list:

        if self.grid is None:
            raise AttributeError(f"Expected a dictionary, but got None instead, from grid attribute, received: {self.grid}")
        
        if len(self.grid) < 1:
            raise AttributeError("Expected a populated dictionary, but received an empty dictionary instead")
            
        num_params = range(1, len(self.grid) + 1)

        return [self.model_code + "-" + str(num) for num in num_params]
    

    def parse_param_dicts(self) -> dict:

        if self.grid is None:
            raise AttributeError("Expected a dictionary, but got None instead, from grid attribute")
        
        if self.param_ids is None:
            raise AttributeError("Expected a dictionary, but got None instead, from grid attribute")
        
        params_with_id = list(zip(self.param_ids, self.grid))

        parsed_dict = {"model_id" : [], 
                       "param" : [], 
                       "value" : []}

        for model in params_with_id:
            for param, value in model[1].items():
                parsed_dict["model_id"].append(model[0])
                parsed_dict["param"].append(param)
                parsed_dict["value"].append(value)

        return parsed_dict
                
    def generate_params_dataframe(self) -> pd.DataFrame:

        param_dict = self.parse_param_dicts()

        return pd.DataFrame(param_dict)


    def save_as_csv(self, folder:str) -> None:

        df = self.generate_params_dataframe()

        if folder.endswith("/"):
            folder = folder.rstrip("/")

        try: 
            df.to_csv(f"{folder}/{self.model_name}.csv", index = False)

        except Exception as e:

            print(f"File was not found - please check the path: {folder}")
            print(f"Model - {self.model_name} - created error - {e}")
            raise SystemError(1)


## A class for the model, which contains the parameters to train the model across
## Some basic input validation will be nice but using a setter could make the __init__ method tidier
## and allow for more complex validations or conditions. 
class Model():
    
    def __init__(self, model, params:Hyperparametres, folds, n_jobs:int = 1,  **kwargs) -> None:
        
        if not is_classifier(model):
            raise ValueError("model should be a classifer from sci-kit learn!")

        self.model_name = params.model_name 
        self.code = params.model_code.upper()
        self.model = model(**kwargs)
        self.params_grid = params 
        self.trained_params = {}
        self.folds = folds
        self.cores = n_jobs



    def cross_validate(self, X, y, metrics:dict) -> dict:

        ## this is true if the dictionary is not empty
        if self.params_grid.params:

            try:

                cv = GridSearchCV(
                estimator = self.model,
                param_grid = self.params_grid.params,
                cv = self.folds,
                scoring = metrics, 
                refit = False, 
                n_jobs = self.cores)

                cv_ = cv.fit(X, y)
           
                ## extract the params used in the cv
                self.trained_params = cv_.cv_results_["params"]

                ## return a dictionary of the metric results - so it can be parsed
                return cv_.cv_results_
            
            except Exception as e:
                print(f"Issue when running GridSearchCV!!\nModel: {self.model_name} \nError: {e}")            
                raise SystemExit(1)



        ## if the params dictionary is empty - no need for GridSearch
        else:
            
            try:

                cv_:dict = cross_validate(
                        estimator = self.model, 
                        X = X, 
                        y = y, 
                        cv = self.folds, 
                        scoring = metrics, 
                        n_jobs = self.cores)

                ## return a dictionary of the metric results - so it can be parsed
                return cv_

            except Exception as e:
                print(f"Issue when running cross_validate!!\nModel: {self.model_name} \nError: {e}")            
                raise SystemExit(1)



    def generate_ids(self):
        """ Generates IDs for the individual folds for each model. 
        The ID has the format "CODE-MODEL-FOLD", where Model and Fold 
        are 1-based indices. This allows us to connect the params to the metrics.
        """

        folds = self.folds.n_splits

        model_id = []

        ## if we didn't do grid search - we only need to do fold cv.
        if not self.trained_params:

            for fold in range(1, folds + 1):

                model_id.append(self.code + "-" + "1" + "-" + str(fold))

        else:
            num_models = len(self.trained_params)
            for fold in range(1, folds + 1):
                for model in range(1, num_models + 1):

                    model_id.append(self.code + "-" + str(model) + "-" + str(fold))

        return model_id





