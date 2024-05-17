


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
  
  if (missing(arg_value) || !tibble::is.tibble(arg_value)){ 
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
  


pivot_param_tibble <- function(params, code){
  
  # verify_is_tibble(params, "params")
  # verify_is_string(code, "code")
  # verify_code_in_tibble(code, params)
  
  .hyperparams <- extract_hyperparams(params, code)
  .len_hp <- length(.hyperparams)
  
  .replace_list <- as.list(setNames(rep("None", .len_hp), .hyperparams))
  
  ## filter than pivot the params table
  .pivot_params <- params |> 
    dplyr::filter(model_code == code) |> 
    tidyr::pivot_wider(
      id_cols = c(model_code, model_id),
      names_from = param, 
      values_from = value) |> 
    tidyr::replace_na(replace = .replace_list)
  
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
      average_type = stringr::str_extract(metric, "[a-z]+$"),
      metric = stringr::str_remove(metric, "beta_"),
      metric_type = stringr::str_extract(metric, "^[a-z]+[0-9]*"), 
      average_type = dplyr::if_else(average_type == "accuracy", "None", average_type)
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
  
  return(list("model" = .model, "summary" = .summary))
  
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
