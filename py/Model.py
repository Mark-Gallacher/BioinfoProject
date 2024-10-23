from sklearn.base import is_classifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ParameterGrid

import pandas as pd
from HelperFunctions import check_types


## A class for the hyperparametres of a model, which is basically a dictionary
## this allows it to integrate with the sci-kit API down the line but allows
## us to keep track of what parameters we could change
class Hyperparameters:
    """`Hyperparameters` represents the hyperparameter grid for a model.
    Requires three arguments for initialisation:
        - `model_name` - a String to name the model - used to name created files
        - `model_code` - a String for a short name for the model - used to create IDs for the individual combinations of hyperparameters.
        - `params` - a Dictionary (or List) containing the hyperparameters with the values, in the format:
            - 'name_of_hyperparameter' : [x_1, x_2, ... , x_n] or
            - [ {'hyperparameter_1' : [x_1, x_2, ..., x_n]},
            -   {'hyperparameter_2' : [y_1, y_2, ..., y_n]} ] or
            - {} ## if no hyperparameters are to be passed to model

    For Example:
        - `model_name` could be 'LogisticRegression' with the `model_code` of 'LG'.
        - `params` could be {"C" : [0.1, 1, 10]} - to control the penalty term 'C'.
    """

    def __init__(self, model_name: str, model_code: str, params: dict) -> None:

        ## validate the inputs are the correct type
        check_types(
            [
                (model_name, "model_name", str),
                (model_code, "model_code", str),
                (params, "params", [list, dict]),
            ]
        )

        ## ensure a useful model name is supplied - defined by length
        if len(model_name) < 1:
            raise ValueError(
                f"Please Supply a more useful name than {model_name}, for example - LogisticRegression"
            )

        ## ensure a useful model code is supplied - defined by length
        if len(model_code) < 1:
            raise ValueError(
                f"Please supply a more useful code than {model_code}, for example - LG for Logistic Regression"
            )

        self.params = params
        self.model_code = model_code.upper()
        self.model_name = model_name

        ## process the params supplied - form the grid of all the combinations
        self.grid = self.create_grid()

        ## assign each unique model with a ID using the model_code
        self.param_ids = self.create_ids()

    def create_grid(self) -> list:
        """Returns a list of the unique combinations of all the hyperparameters.
        Uses the ParameterGrid from Scikit-Learn.
        """

        ## if the dictionary has content
        if self.params:
            grid = ParameterGrid(self.params)
            grid = list(grid)

        ## if the dictionary is empty
        else:

            grid = [{"NA": "NA"}]

        return grid

    def create_ids(self) -> list:
        """Returns a list of the unique IDs.
        - Used to identify the individual combination of hyperparameter.
        - IDs have form `model_code`-number (if code is LG, then LG-1, LG-2, ..., LG-n)
        """

        ## check grid has a truthy value
        if self.grid is None:
            raise AttributeError(
                f"Expected a dictionary, but got None instead, from grid attribute, received: {self.grid}"
            )

        ## check that value has a length greater than one
        if len(self.grid) < 1:
            raise AttributeError(
                "Expected a populated dictionary, but received an empty dictionary instead"
            )

        num_params = range(1, len(self.grid) + 1)

        ## IDs use model_code with a dash then a number.
        return [self.model_code + "-" + str(num) for num in num_params]

    def parse_param_dicts(self) -> dict:
        """Parses the Dictionary of hyperparametrics to a long format Dictionary.
        Returns a dictionary with three keys ('model_id', 'param' and 'value').
        Used to generate a CSV file that can handle different number of hyperparameters.
        """

        if self.grid is None:
            raise AttributeError(
                "Expected a dictionary, but got None instead, from grid attribute"
            )

        if self.param_ids is None:
            raise AttributeError(
                "Expected a dictionary, but got None instead, from grid attribute"
            )

        ## pair up the unique IDs with the combinations of hyperparameters
        params_with_id = list(zip(self.param_ids, self.grid))

        ## define the output dictionary
        parsed_dict = {"model_id": [], "param": [], "value": []}

        ## iterate over the models
        for model in params_with_id:

            ## model is ["model_id", {dict of params} ]
            ## some models can have several hyperparameters, all get parsed into the
            ## same three 'columns'
            for param, value in model[1].items():
                parsed_dict["model_id"].append(model[0])
                parsed_dict["param"].append(param)
                parsed_dict["value"].append(value)

        return parsed_dict

    def generate_params_dataframe(self) -> pd.DataFrame:
        """Simply converts the Dictionary of Hyperparameters into a Pandas.DataFrame."""

        param_dict = self.parse_param_dicts()

        return pd.DataFrame(param_dict)

    def save_as_csv(self, folder: str) -> None:
        """Saves a CSV file in `folder` with name: `model_name`.csv"""

        df = self.generate_params_dataframe()

        ## incase the folder is supplied like "output/"
        ## the format "output//" might cause an issue
        if folder.endswith("/"):
            folder = folder.rstrip("/")

        try:
            df.to_csv(f"{folder}/{self.model_name}.csv", index=False)

        except Exception as e:

            print(f"File was not found - please check the path: {folder}")
            print(f"Model - {self.model_name} - created error - {e}")
            raise SystemError(1)


## A class for the model, which contains the parameters to train the model across
## Some basic input validation will be nice but using a setter could make the __init__ method tidier
## and allow for more complex validations or conditions.
class Model:

    def __init__(
        self, model, params: Hyperparameters, folds, n_jobs: int = 1, **kwargs
    ) -> None:

        if not is_classifier(model):
            raise ValueError("model should be a classifer from sci-kit learn!")

        self.model_name = params.model_name
        self.code = params.model_code.upper()
        self.model = model(**kwargs)
        self.params_grid = params
        self.trained_params = {}
        self.folds = folds
        self.cores = n_jobs

    def cross_validate(self, X, y, metrics: dict) -> dict:

        ## this is true if the dictionary is not empty
        if self.params_grid.params:

            try:

                cv = GridSearchCV(
                    estimator=self.model,
                    param_grid=self.params_grid.params,
                    cv=self.folds,
                    scoring=metrics,
                    refit=False,
                    n_jobs=self.cores,
                )

                cv_ = cv.fit(X, y)

                ## extract the params used in the cv
                self.trained_params = cv_.cv_results_["params"]

                ## return a dictionary of the metric results - so it can be parsed
                return cv_.cv_results_

            except Exception as e:
                print(
                    f"Issue when running GridSearchCV!!\nModel: {self.model_name} \nError: {e}"
                )
                raise SystemExit(1)

        ## if the params dictionary is empty - no need for GridSearch
        else:

            try:

                cv_: dict = cross_validate(
                    estimator=self.model,
                    X=X,
                    y=y,
                    cv=self.folds,
                    scoring=metrics,
                    n_jobs=self.cores,
                )

                ## return a dictionary of the metric results - so it can be parsed
                return cv_

            except Exception as e:
                print(
                    f"Issue when running cross_validate!!\nModel: {self.model_name} \nError: {e}"
                )
                raise SystemExit(1)

    def generate_ids(self):
        """Generates IDs for the individual folds for each model.
        The ID has the format "CODE-MODEL-FOLD", where Model and Fold
        are 1-based indices. This allows us to connect the params to the metrics.
        """

        if isinstance(self.folds, int):
            folds = self.folds

        else:
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
