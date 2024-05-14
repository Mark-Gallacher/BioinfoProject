

def expand_metrics(metrics:list) -> dict[str]:

    if not isinstance(metrics, list):
        raise ValueError("metrics should be a list of strings")

    if len(metrics) < 1:
        raise ValueError("metrics should be a list of string")

    expandable_metrics = ["f1", "precision", "recall"]
    metrics_suffix = ["macro", "micro", "weighted"]

    output = {}

    for metric in metrics:
        if metric not in expandable_metrics:

            output[metric] = metric
        
        else:

            expanded_metrics = [metric + "_" + suffix for suffix in metrics_suffix]

            for new_metric in expanded_metrics:
                output[new_metric] = new_metric

    return output

# print(expand_metrics(["hello"])) 
# print(expand_metrics(["f1"])) 

assert expand_metrics(["hello"]) == {"hello" : "hello"}
assert expand_metrics(["f1"]) == {"f1_micro" : "f1_micro", 
                                "f1_macro" : "f1_macro", 
                                "f1_weighted" : "f1_weighted"}
