import re


def tidy_metric_name(name:str) -> str | None:
    """ REGEX for the metric name out from GridSearchCV.cv_result_
    Example Converts:
        - "split0_test_f1_micro" --> "f1_micro"
    """
    search = re.search("(split[0-9]+_test_)(.*)", name)

    if search is not None:
        return search.group(2)
    
    return None

def merge_metric(metric_dict:dict, name:str, values:list) -> dict:
    """ Merges multiple metrics into one dictionary whilst appending if the metric is already present.
    """
    if name in metric_dict.keys():
        metric_dict[name] += values

    else:
        metric_dict[name] = values

    return metric_dict

def parse_cv_results(results:dict, metric_output:dict) -> dict:
    """ Extracts the Metrics from the cv_results_ dictionary. 
    Transposes the lists to output the form 
        - "metric" : [model1, model2, model3, ...]
        - for all metrics supplied
    """
    for key in results.keys():
        if key.startswith("split"):
                
            ## get the values then tidy the name of the metric
            values = results[key].tolist()
            metric_name = tidy_metric_name(key)

            if metric_name is None:
                raise ValueError("No Metrics were found in this Results Dictionary!!")

            ## generate a dictionary containing all the metrics
            metric_dict = merge_metric(metric_output, metric_name, values)

    return metric_dict


