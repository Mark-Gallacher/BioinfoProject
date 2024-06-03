from sklearn.feature_selection import RFECV
import pandas

class FeatureElimination():

    def __init__(self, model_name, model_code, model, folds, n_jobs, out_dir, **kwargs ) -> None:
        self.model_name = model_name 
        self.code = model_code.upper()
        self.model = model(**kwargs)
        self.folds = folds
        self.cores = n_jobs
        self.fit = None
        self.out_dir = out_dir
    
    def cross_validate(self, X, y, metrics = "balanced_accuracy") -> dict:


        ## run the Recurcive Feauture Elimiation Cross Validation
        ## ideally we would supply the same metric dictionary to the hyperparameter
        ## tuning stage, so we could see if the metrics behave the same.
        
        try :
            _rfe = RFECV(estimator = self.model, 
                    cv = self.folds, 
                    scoring = metrics, ## change later on though
                    step = 1,
                    min_features_to_select = 1, 
                    n_jobs = self.cores)


            # generate the fitted object, then extract the dictionary of metrics from it.
            self.fit = _rfe.fit(X = X, y = y)

        except Exception as e:
        
            # print(f"Issue when running RFECV!!\nModel: {self.model_name} \nError: {e}")
            raise SystemError(f"Issue when running RFECV!!\nModel: {self.model_name} \nError: {e}")


        return _rfe.cv_results_

    def extract_best_features(self, raw_df : pandas.DataFrame) -> None:

        if self.fit is None:

            raise AttributeError(f"Attribute - fit - appears to be not set - it has value None, have you ran cross_validatate?")


        ## define the name of the file, use the dir provided and the model name
        file_name = self.out_dir + "/" + self.model_name + "_Features.csv"

        ## filter the original data
        columns_to_keep = self.fit.get_support(indices = True)
        filter_df = raw_df.iloc[:, columns_to_keep].copy()

        ## add on the PseudoID and the Labels to filter_df
        label_df = raw_df[["DiseaseSubtypeFull", "DiseaseSubtype"]]
        df = pandas.concat([filter_df, label_df], axis = 1)

        ## write to a csv to use in the full pipeline
        df.to_csv(file_name, index = False)


    def generate_ids(self) -> list:
   
        model_id = []
        folds = self.folds.n_splits

        for fold in range(1, folds + 1):

            ## there are 178 features, numbered from 1 t0 178
            for num in range(1, 179):
            
                model_id.append(self.code + "-" + str(num) + "-" + str(fold))

        return model_id



