from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, log_loss, accuracy_score, balanced_accuracy_score

class Metric():

    def __init__(self, metric_name:str, is_binary:bool = False):
        
        if self.check_metric_input(metric_name):

            self.metric_name = metric_name
            self.is_binary = is_binary
            self.scorer_func = {} 
            self.is_expandable = self.check_metric_type()

    def check_metric_input(self, metric_name):
        """ Checks the `metric_name` and Initialises the `metric_func` attr.
        Returns True if `metric_name` is in the accepted list then sets the value of
        `metric_func` to the corresponding scikit-learn metric function.
        """

        all_metrics = {"f1" : f1_score,
                       "fbeta": fbeta_score,
                       "precision": precision_score, 
                       "recall" : recall_score, 
                       "roc_auc" : roc_auc_score, 
                       "cohen_kappa" :cohen_kappa_score, 
                       "matthew_coef" : matthews_corrcoef, 
                       "log_loss" : log_loss, 
                       "accuracy" : accuracy_score, 
                       "balanced_accuracy" : balanced_accuracy_score}

        ## ensure the names are standardised across the metrics - ensures consistency and ease of comparison
        if metric_name not in all_metrics.keys():
            raise ValueError(f"metric_name should be a member of {all_metrics.keys()} but received {metric_name} instead.")

        ## ensure the given name relates to the same function - so "f1" is not paired with precision_score
        # if metric_func is not all_metrics[metric_name]:
        #     raise ValueError(f"metric_func was expected to be {all_metrics[metric_name]} but received {metric_func} instead.")
        
        self.metric_func = all_metrics[metric_name]

        ## there is a suitable name and corresponding function 
        return True

    def check_metric_type(self) -> bool:
        
        ## Define a list of metrics which can be expanded by averaging strategies.
        expandable_metrics = ["f1", "precision", "recall", "fbeta", "roc_auc"]

        if self.metric_name in expandable_metrics:
            return True

        else:
            return False

    def generate_scorer_func(self) -> None:

        ## check if we are using it in a binary setting
        if self.is_binary:

            self.scorer_func = self.generate_single_scorer()

        ## check if the metric can be expanded by averaging types
        elif not self.is_expandable:

            self.scorer_func = self.generate_single_scorer()

        else:
            ## generate all the scoring functions for each metric
            expanded_metrics = self.generate_expanded_scorers()

            ## then add to the dictionary
            for new_metric in expanded_metrics.keys():
                self.scorer_func[new_metric] = expanded_metrics[new_metric]
    
    def generate_expanded_scorers(self) -> dict:

        
        labels = ['CS', 'HV', 'PA', 'PHT', 'PPGL']
        metrics_suffix = ["macro", "micro", "weighted"]

        ## fbeta requires additional arguments - handle separately
        if self.metric_name == "fbeta":
            betas = [2, 3, 4]

            ## generate a dictionary of all the variations of the scoring functions
            scorer = {self.metric_name + "_" + str(beta) + "_" + average : 
                      make_scorer(
                          self.metric_func,
                          beta = beta,
                          average = average, 
                          zero_division = 0,
                          labels = labels
                          )

                      for average in metrics_suffix
                      for beta in betas}

            return scorer

        if self.metric_name == "roc_auc":
            ## generate a dictionary of all the variations of the scoring functions
            scorer = {self.metric_name + "_" + average : 
                      make_scorer(
                          self.metric_func,
                          multi_class = "ovr",  ## One vs the Rest - default is raise.
                          average = average, 
                          labels = labels
                          )

                      for average in metrics_suffix}

            return scorer

        ## generate a dictionary of all the variations of the scoring functions
        scorer = {self.metric_name + "_" + average : 
                  make_scorer(
                      self.metric_func, 
                      average = average, 
                      zero_division = 0,
                      labels = labels, 
                      pos_label = None
                      )
                  for average in metrics_suffix}

        return scorer

    def generate_single_scorer(self) -> dict:
        """ When we have a binary problem or the metric is not 'expandable' by averaging methods
        Returns a single scorer as dictionary:
            {`metric_name` : make_scorer(`metric_func`, ...)}
        This allows the scorers to be consistent across multiple settings and expanding the available metrics we
        can use in the cross validation process.
        """

        scorer = {self.metric_name : make_scorer(self.metric_func)}

        return scorer

