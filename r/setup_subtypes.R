library(tidyverse)
library(here)

source(here("r/process_merged_data_functions.R"))
source(here("r/graph_functions.R"))

### READING IN THE DATA
.raw_metrics <- read_csv(here("data/subtypes/merged/metrics_merged.csv"))
.raw_params <- read_csv(here("data/subtypes/merged/params_merged.csv"))


### PARSING THE DATA
## generate a df just for the confusion values.
confusion <- .raw_metrics |> 
  select(id, starts_with(c("tp_", "fp_", "tn_", "fn_"))) |> 
  mutate(
    model_code = str_extract(id, "[A-Z]+"), 
    model_id = str_extract(id, "[A-Z]+-[0-9]+")) |> 
  relocate(any_of(c("model_code", "model_id")), .after = id)

## removing the confusion matrix values from the standard metric values
.raw_metrics <- .raw_metrics |> 
  select(!starts_with(c("tp_", "fp_", "tn_", "fn_")))

## expand the id column to add in the model code (like RF, GNB) and the model id (like RF-1)
metrics <- .raw_metrics |> 
  mutate(
    model_code = str_extract(id, "[A-Z]+"), 
    model_id = str_extract(id, "[A-Z]+-[0-9]+")) |> 
  relocate(any_of(c("model_code", "model_id")), .after = id)

## expand the model id to get the model code - this ensures the two dfs have similar columns
params <- .raw_params |> 
  mutate(
    model_code = str_extract(model_id, "[A-Z]+")) |> 
  relocate(any_of(c("model_code", "model_id"))) |> 
  replace_na(list(param = "None")) ## If there are no hyperparameters, then give it the value "None"


### COLLECTING ALL THE CODES, NAMES, MODEL, etc...

## list of model codes
all_codes <- extract_models(params) |> 
  set_names()

## list of full name of models
all_names <- extract_fullname(metrics) |> 
  set_names(all_codes)

## list of all the hyperparameters for each model
all_hyper <- all_codes |> 
  map(~extract_hyperparams(params, .x))

## list of all the names of the model tibbles
all_models <- all_codes |> 
  str_to_lower() |> 
  paste0("_model") |> 
  set_names(all_codes)

## list of all the names of the summary tibbles
all_summaries <- all_codes |> 
  str_to_lower() |> 
  paste0("_summary") |> 
  set_names(all_codes)

### GENERATING ALL THE MODEL AND SUMMARY TIBBLES

generate_all_tibbles(metrics = metrics, params = params)

