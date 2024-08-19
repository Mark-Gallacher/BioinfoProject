from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, log_loss, accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix

import numpy as np

class Metric():
    """ `Metric` represents the scoring function(s) of a single metrics. 
    Requires three arguments for initialisation:
        - `metric_name` - a String of the name of the metric - as only a limited range are supported.
        - `labels` - a List of the class labels to ensure consistent order.
        - `is_binary` (default = False) - a Boolean, if metric is used in multi-class setting, set to True.

    `Metric.generate_scorer_func() is the main method, which generates a dictionary scikit-learn scoring function(s). 
    This handles the metrics which can have variants through averaging (Macro, Micro and Weighted), and ensure the other methods
    are in a consistent format. The output could be passes to `Model` to collect all the metrics during cross-validation. 
    """

    def __init__(self, metric_name:str, labels:list, is_binary:bool = False):
        
        if self.check_metric_input(metric_name):

            self.metric_name = metric_name
            self.is_binary = is_binary
            self.labels = labels
            self.scorer_func = {} 
            self.is_expandable = self.check_metric_type()

    def check_metric_input(self, metric_name):
        """ Checks the `metric_name` and Initialises the `metric_func` attr.
        - Returns True if `metric_name` is in the accepted list.
        - Sets the value of `metric_func` to the corresponding scikit-learn metric function.
        - Returns ValueError if `metric_name` is not in accepted list. 
        """

        ## The collection of metrics we could use for this project - not all are used/presented
        ## log_loss would be nice to include but this does not apply to all methods - as the model most assign a probability to the labels
        ## for SVM this would require additionally computation, and the bootstrapping is perfect either. 
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

        ## store the metric function for conveniences - sometimes the name is different to name of function. 
        self.metric_func = all_metrics[metric_name]

        ## there is a suitable name and corresponding function 
        return True

    def check_metric_type(self) -> bool:
        """ Checks if `metric_name` is a metric which could be expanded. 
        - Returns True if the metric has Macro, Micro and Weighted Averaging options. 
        - Returns False if used in binary setting or has no averaging options. 
        """
        
        ## Define a list of metrics which can be expanded by averaging strategies.
        expandable_metrics = ["f1", "precision", "recall", "fbeta", "roc_auc"]

        if self.is_binary:
            return False

        if self.metric_name in expandable_metrics:
            return True

        else:
            return False

    def generate_scorer_func(self) -> None:
        """ Defines the function(s) stored inside `scorer_func`. 
        - If the metric can be expanded - see `check_metric_type()` - it generates the variants. 
        - The function(s) are stored in a dictionary, even when only one scorer function is created. 
        Does not return a value.
        """

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
        """ Expands a Metric with the three forms of averaging (Macro, Micro and Weighted). 
        - Returns a dictionary of `Scorers`.
        """

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
                          labels = self.labels
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
                          labels = self.labels
                          )

                      for average in metrics_suffix}

            return scorer

        ## only reach this point if metric_name is not roc_auc or fbeta. 

        ## generate a dictionary of all the variations of the scoring functions
        scorer = {self.metric_name + "_" + average : 
                  make_scorer(
                      self.metric_func, 
                      average = average, 
                      zero_division = 0,
                      labels = self.labels, 
                      pos_label = None
                      )
                  for average in metrics_suffix}

        return scorer

    def generate_single_scorer(self) -> dict:
        """ When we have a binary problem or the metric is not 'expandable' by averaging methods
        - Returns a single scorer as dictionary: {`metric_name` : make_scorer(`metric_func`, ...)}
        - This allows the scorers to be consistent across multiple settings and expanding the available metrics we
        can use in the cross validation process.
        """

        ## fbeta requires additional arguments - handle separately
        if self.metric_name == "fbeta":
            betas = [2, 3, 4]

            ## generate a dictionary of all the variations of the scoring functions
            scorer = {self.metric_name + "_" + str(beta) + "_": 
                      make_scorer(
                          self.metric_func,
                          labels = self.labels,
                          beta = beta)

                      for beta in betas}


            return scorer

        ## some metrics require additionally handling of zero_division - or they generate verbose warnings/errors
        elif self.metric_name in ["precision", "recall", "f1_score"]:

            scorer = {self.metric_name : make_scorer(self.metric_func, 
                                                     labels = self.labels, 
                                                     zero_division = 0)}

        ## some metrics require labels - could be a bug - but it silenced a few warnings/errors
        elif self.metric_name in ["cohen_kappa", "roc_auc"]:

            scorer = {self.metric_name : make_scorer(self.metric_func, 
                                                     labels = self.labels)}

        else:

            scorer = {self.metric_name : make_scorer(self.metric_func)}


        return scorer


class ConfusionMetrics():
    """ `ConfusionMetrics` calculates the relevant metrics for each class from a Confusion Matrix. 
    Required one argument for initialisation:
        - `labels` - a List of the class labels to ensure consistent order.

    ConfusionMetrics.generate_scorers() is the main method to use - returns a dictionary of scikit-learn scorers. 
    """
    def __init__(self, labels:list) -> None:
        self.labels = labels

    def label_confusion_values(self, cm) -> dict:
        """Takes in a confusion matrix, and puts values into a dictionary"""

        ##      A   B   C
        ## A    10  2   3
        ## B    3   12  4
        ## C    0   1   14

        ## False Positive is the sum of column minus the the true positives (the diagonal)
        FP = cm.sum(axis=0) - np.diag(cm)  
        ## False Negative is the sum of the row minus the true positions
        FN = cm.sum(axis=1) - np.diag(cm)
        ## True positives are on the diagonal
        TP = np.diag(cm)
        ## Sum of entire matrix, then subtract all the previous values
        TN = cm.sum() - (FP + FN + TP)

        ## returns {"tp" : [10, 12, 14], ...}
        return {"tp" : TP, "fp" : FP, "tn" : TN, "fn" : FN}

    def generate_extractor(self, metric, label):
        """ Returns a Function that is initialised to a given `metric` and `label`.
        - The returned function takes in an array of the true and for the predicted values, termed y_true and y_pred. 
        - These arguments are passed to the scikit-learn `confusion_matrix` function.
        - When the returned function is executed, the defined `metric` and `label` are extracted.
        """

        ## check if the label exists by letting an error occur then catching it. 
        ## get the index of the label in the list of labels
        ## this is used to alter the output of `extract_class_metric`
        
        try:

            index = self.labels.index(label)

        except ValueError:

            raise SystemError(f"Received {label} but expected an item in {self.labels}")

        
        def extract_class_metric(y_true, y_pred):
            """ Obtains the confusion matric, with the labels in the defined order. 
            Returns the specified `metric` and `label` - specified when calling `ConfusionMetrics.generate_extractor()`.
            """

            ## generate the confusion matrix from predicted and true lables
            cm = confusion_matrix(y_true, y_pred, labels = self.labels)

            ## add the labels, ie the TP, FP, TN and FP - with their values in the list
            cm_dict = self.label_confusion_values(cm)

            ## extract the specific value
            ## if we want the True Positives for the thrid class
            ## we would want cm_dict["tp"][2]. 
            return cm_dict[metric][index]

        return extract_class_metric

    def generate_scorers(self) -> dict:
        """ Returns a Dictionary containing a scikit-learn `Scorer` for each metric, for each class. 
        - The metrics are the counts for True Positive (TP), False Positive (FP), True Negative (TN) and False Negative (FN). 
        - If classes are A and B, returns a dictionary with keys "tp_A", "tp_B", "fp_A", ... , "fn_B". 

        In General, the number of keys are 4 time the number of classes. 
        """

        scorer_funcs = {}

        ## iterate over type of metric
        for metric in ["tp", "fp", "tn", "fn"]:

            ## iterate over all the labels
            for label in self.labels:

                ## key should be in form "tp_class1", when class1 is the label used in the data.
                ## if we had labels A and B: key = "tp_A", "tp_B", "fp_A", ... , "fn_B" 
                key = str(metric) + "_" + str(label)

                ## define a custom scoring function, which returns a single value from the confusion matrix. 
                scorer_funcs[key] = make_scorer(self.generate_extractor(metric, label))

        return scorer_funcs

