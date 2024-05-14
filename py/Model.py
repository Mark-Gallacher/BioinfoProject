from sklearn.base import is_classifier 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

## A class for the hyperparametres of a model, which is basically a dictionary
## this allows it to integrate with the sci-kit API down the line but allows 
## us to keep track of what parameters we could change
class Hyperparametres():

    def __init__(self, params: dict | list[dict]) -> None:
        self.params = params


## A class for the model, which contains the parameters to train the model across
## Some basic input validation will be nice but using a setter could make the __init__ method tidier
## and allow for more complex validations or conditions. 
class Model():
    
    def __init__(self, name:str, code:str, model, params:Hyperparametres, folds : int = 1, **kwargs) -> None:
        
        if not is_classifier(model):
            raise ValueError("model should be a classifer from sci-kit learn!")

        if len(name) < 1:
            raise ValueError("name should have a string with a length greater than 1!")

        self.model_name = name 
        self.code = code.upper()
        self.model = model(**kwargs)
        self.params_grid = params 
        self.trained_params = {}
        self.folds = folds

    def cross_validate(self, X, y, metrics : dict) -> dict:

        ## this is true if the dictionary is not empty
        if self.params_grid.params:

            cv = GridSearchCV(
            estimator = self.model,
            param_grid = self.params_grid.params,
            cv = self.folds,
            scoring = metrics, 
            refit = False)

            cv_ = cv.fit(X, y)
        
            self.trained_params = cv_.cv_results_["params"]

        ## if the params dictionary is empy - no need for GridSearch
        else:
            
            cv_ = cross_validate(
                    estimator = self.model, 
                    X = X, 
                    y = y, 
                    cv = self.folds, 
                    scoring = metrics)

        return cv_

    def generate_ids(self):
        """ Generates IDs for the individual folds for each model. 
        The ID has the format "CODE-MODEL-FOLD", where Model and Fold 
        are 1-based indices. This allows us to connect the params to the metrics.
        """

        model_id = []

        ## if we didn't do grid search - we only need to do fold cv.
        if not self.trained_params:

            for fold in range(1, self.folds + 1):

                model_id.append(self.code + "-" + "1" + "-" + str(fold))

        else:
            num_models = len(self.trained_params)
            for fold in range(1, self.folds + 1):
                for model in range(1, num_models + 1):

                    model_id.append(self.code + "-" + str(model) + "-" + str(fold))

        return model_id





