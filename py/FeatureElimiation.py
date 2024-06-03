from sklearn.feature_selection import RFECV


class FeatureElimiation():

    def __init__(self, model_name, model_code, model, metrics, folds, n_jobs, **kwargs ) -> None:
        self.model_name = model_name 
        self.code = model_code.upper()
        self.model = model(**kwargs)
        self.metrics = metrics
        self.folds = folds
        self.cores = n_jobs
    
    def cross_validate(self, X, y) -> dict:


        ## run the Recurcive Feauture Elimiation Cross Validation
        ## ideally we would supply the same metric dictionary to the hyperparameter
        ## tuning stage, so we could see if the metrics behave the same.
        
        try :
            _rfe = RFECV(estimator = self.model, 
                    cv = self.folds, 
                    scoring = "balanced_accuracy", ## change later on though
                    step = 1,
                    min_features_to_select = 1, 
                    n_jobs = self.cores)


            # generate the fitted object, then extract the dictionary of metrics from it.
            rfe = _rfe.fit_transform(X = X, y = y)

        except Exception as e:
        
            print(f"Issue when running RFECV!!\nModel: {self.model_name} \nError: {e}")
            raise SystemError(1)

        return rfe


    def generate_ids(self):
   
        model_id = []
        
        for fold in range(1, self.folds + 1):
            
            model_id.append(self.code + "-" + "1" + "-" + str(fold))



