from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score, fbeta_score, roc_auc_score, cohen_kappa_score, matthews_corrcoef, log_loss
# import numpy as np

def generate_scorers(metric:str) -> dict:

    metric_scorers = {"f1" : f1_score,
                      "fbeta": fbeta_score,
                      "precision": precision_score, 
                      "recall" : recall_score, 
                      "roc_auc" : roc_auc_score}

    metrics_suffix = ["macro", "micro", "weighted"]

    ## ensure the metric string is in our dictionary
    if metric not in metric_scorers.keys():
        return None
    
    ## fbeta requires additional arguments - handle separately
    if metric == "fbeta":
        betas = [2, 3, 4]

        ## generate a dictionary of all the variations of the scoring functions
        scorer = {metric + "_" + str(beta) + "_" + average : 
                  make_scorer(
                      metric_scorers[metric],
                      beta = beta,
                      average = average, 
                      zero_division = 0, 
                      # response_method = "predict",
                      labels = ['CS', 'HV', 'PA', 'PHT', 'PPGL']
                      )

                  for average in metrics_suffix
                  for beta in betas}

        return scorer

    if metric == "roc_auc":
        ## generate a dictionary of all the variations of the scoring functions
        scorer = {metric + "_" + average : 
                  make_scorer(
                      metric_scorers[metric],
                      multi_class = "ovr",  ## One vs the Rest - default is raise.
                      average = average, 
                      # response_method = "predict",
                      labels = ['CS', 'HV', 'PA', 'PHT', 'PPGL']
                      )

                  for average in metrics_suffix}

        return scorer

    ## generate a dictionary of all the variations of the scoring functions
    scorer = {metric + "_" + average : make_scorer(
                                       metric_scorers[metric], 
                                       # response_method = "predict",
                                       average = average, 
                                       zero_division = 0,
                                       labels = ['CS', 'HV', 'PA', 'PHT', 'PPGL'], 
                                       pos_label = None
                                       )
              for average in metrics_suffix}

    return scorer

# print(generate_scorers("fbeta"))


def expand_metrics(metrics:list) -> dict:

    ## check the metrics supplied are a list and have more than one element in them
    if not isinstance(metrics, list):
        raise ValueError("metrics should be a list of strings")

    if len(metrics) < 1:
        raise ValueError("metrics should not be an empty list")

    expandable_metrics = ["f1", "precision", "recall", "fbeta", "roc_auc"]

    output = {}
    for metric in metrics:
        if metric not in expandable_metrics:

            output[metric] = metric
        
        else:

            ## generate all the scoring functions for each metric
            expanded_metrics = generate_scorers(metric)

            if expanded_metrics is None:
                continue

            ## then add to the dictionary
            for new_metric in expanded_metrics.keys():
                output[new_metric] = expanded_metrics[new_metric]

    return output

# assert expand_metrics(["hello"]) == {"hello" : "hello"}
# assert expand_metrics(["f1"]) == {"f1_micro" : "f1_micro", 
                                # "f1_macro" : "f1_macro", 
                                # "f1_weighted" : "f1_weighted"}
