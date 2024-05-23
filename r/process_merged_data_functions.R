


verify_code_in_tibble <- function(params, code){
  
  .all_codes <- extract_models(params)
  
  if(!is.element(code, .all_codes)) {
    message = stringr::str_c(
      "Please ensure the `code` is in the tibble!\n - Received '", code,
      "' but expected an element in '", paste0(.all_codes, collapse = "', '"), "'"
    )
    
    stop(message)
  }else{
    
    return(TRUE)
    
  }
}



verify_is_tibble <- function(arg_value, arg_name){
  
  if (missing(arg_value) || !tibble::is_tibble(arg_value)){ 
    stop(paste0("Please ensure you supplied a tibble for `", arg_name, "`!"))
  }else{
    
    return(TRUE)
    
  }
}



verify_is_string <- function(arg_value, arg_name){
  
  if (missing(arg_value) || !is.character(arg_value)){ 
    stop(paste0("Please ensure you supplied a string for `", arg_name, "`!"))
  }else{
    
    return(TRUE)
    
  }
}



extract_hyperparams <- function(params, code){
  
  # verify_is_tibble(params, "params")
  verify_is_string(code, "code")
  # verify_code_in_tibble(params, code)
  
  .hyperparams <- params |> 
    dplyr::filter(model_code == code) |> 
    purrr::pluck("param") |> 
    unique()
  
  return(.hyperparams)
  
}


extract_models <- function(params) {
  
  .all_codes <- params |> 
    purrr::pluck("model_code") |> 
    unique()
  
  return(.all_codes)
  
} 

extract_fullname <- function(metrics) {
  
  .all_names <- metrics |> 
    purrr::pluck("model_type") |> 
    unique()
  
  return(.all_names)
  
} 

get_model_from_code <- function(code){
  
  if (!exists("all_models")){ 
    stop(paste0("Please ensure you have created the all_model object!"))
  }
  ## get the name of the tibble
  .tibble_name <- all_models[code] 
  
  ## convert the string to the actual object
  .tibble <- eval(as.name(.tibble_name))
  
  return(.tibble)
}

get_summary_from_code <- function(code){
  
  if (!exists("all_summaries")){ 
    stop(paste0("Please ensure you have created the all_summaries object!"))
  }
  ## get the name of the tibble
  .tibble_name <- all_summaries[code] 
  
  ## convert the string to the actual object
  .tibble <- eval(as.name(.tibble_name))
  
  return(.tibble)
}
  


pivot_param_tibble <- function(params, code){
  
  # verify_is_tibble(params, "params")
  # verify_is_string(code, "code")
  # verify_code_in_tibble(code, params)
  
  to_title_chaarcter <- function(x) return(stringr::str_to_title(as.character(x)))
  
  numeric_hyperparameters <- c("learning_rate", "max_depth", "min_samples_split", "n_estimators", "n_neighbors", "C", "l1_ratio")
  categoric_hyperparameters <- c("weights", "loss", "penalty", "kernel", "gamma", "None")

  .replace_list_n <- as.list(setNames(rep(0, length(numeric_hyperparameters)), numeric_hyperparameters))
  .replace_list_c <- as.list(setNames(rep("None", length(categoric_hyperparameters)), categoric_hyperparameters))
  
  ## filter than pivot the params table
  .pivot_params <- params |> 
    dplyr::filter(model_code == code) |> 
    tidyr::pivot_wider(
      id_cols = c(model_code, model_id),
      names_from = param, 
      values_from = value) |> 
    dplyr::mutate(
      across(
        .cols = any_of(numeric_hyperparameters),
        .fns = as.numeric),
      across(
        .cols = any_of(categoric_hyperparameters),
        .fns =  to_title_chaarcter)
      ) |>
    replace_na(.replace_list_n) |>
    replace_na(.replace_list_c)
  
  return(.pivot_params)
}



pivot_metrics_tibble <- function(metrics, code){
  
  ## this should probably not be static, but I dont see it changing. 
  info_cols <- c("model_type", "id", "model_code", "model_id")
  
  # verify_is_tibble(metrics, "metrics")
  # verify_is_string(code, "code")
  # verify_code_in_tibble(code, params)
  
  .pivot_metrics <- metrics |> 
    dplyr::filter(model_code == code) |> 
    tidyr::pivot_longer(
      cols = -dplyr::any_of(info_cols),
      names_to = "metric", 
      values_to = "score")
  
  return(.pivot_metrics)
  
}



parse_metric_column <- function(metrics){
  
  # verify_is_tibble(metrics, "metrics")
  
  .parse_metrics <- metrics |>
    dplyr::mutate(
      average_type = stringr::str_extract(metric, "(macro|micro|weighted)$"),
      metric = stringr::str_remove(metric, "beta_"),
      metric_type = stringr::str_extract(metric, "^[a-z]+[0-9]*"), 
      average_type = dplyr::if_else(is.na(average_type), "None", stringr::str_to_title(average_type))
    )
  
  return(.parse_metrics)
  
}



link_model_params <- function(metrics, params){
  
  # verify_is_tibble(metrics, "metrics")
  # verify_is_tibble(params, "params")
  
  .linked_tibble <- dplyr::inner_join(metrics, params, 
                               by = c("model_id", "model_code"))
  
  return (.linked_tibble)
  
}



summarise_model_tibble <- function(model, params){
  

  .summary <- model |> 
    dplyr::summarise(
      .by = c(model_id, metric, metric_type, average_type),
      mean_score = mean(score, na.rm = T), 
      sd_score = sd(score, na.rm = T)) |> 
    dplyr::inner_join(params, by = "model_id") 
  
  
  return(.summary)
  
}



generate_model_tibbles <- function(metrics, params, code){
  
  verify_is_tibble(metrics, "metrics")
  verify_is_tibble(params, "params")
  verify_is_string(code, "code")
  verify_code_in_tibble(params, code)
  

  .params <- pivot_param_tibble(params = params, code = code)
  
  .model <- pivot_metrics_tibble(metrics = metrics, code = code) |> 
    parse_metric_column() |> 
    link_model_params(params = .params)
  
  .summary <- summarise_model_tibble(model = .model, params = .params)
  
  .hyperparams <- extract_hyperparams(params, code = code)
  
  return(list("model" = .model, "summary" = .summary, "hyper" = .hyperparams))
  
}


generate_all_tibbles <- function(metrics, params){
  
  purrr::walk(
    .x = extract_models(params), 
    .f = function(x){
      
      ## generate the model and summary tibbles
      .dfs <- generate_model_tibbles(metrics = metrics, params = params, code = x)
      
      ## save both of them to global objects
      assign(str_c(str_to_lower(x), "_model"), .dfs$model, envir = .GlobalEnv)
      assign(str_c(str_to_lower(x), "_summary"), .dfs$summary, envir = .GlobalEnv)
      
    }, .progress = "Generating Tibbles"
  )
}
