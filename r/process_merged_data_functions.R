


verify_code_in_tibble <- function(params, code){
  
  .all_codes <- params |> 
    purrr::pluck("model_code") |> 
    unique()
  
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
  # verify_is_string(code, "code")
  # verify_code_in_tibble(params, code)
  
  .hyperparams <- params |> 
    dplyr::filter(model_code == code) |> 
    purrr::pluck("param") |> 
    unique()
  
  return(.hyperparams)
  
}



pivot_param_tibble <- function(params, code){
  
  # verify_is_tibble(params, "params")
  # verify_is_string(code, "code")
  # verify_code_in_tibble(code, params)
  
  .hyperparams <- extract_hyperparams(params, code)
  
  ## filter than pivot the params table
  .pivot_params <- params |> 
    dplyr::filter(model_code == code) |> 
    tidyr::pivot_wider(
      id_cols = c(model_code, model_id),
      names_from = param, 
      values_from = value)
  
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
      average_type = dplyr::if_else(average_type == "accuracy", "NA", average_type)
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



generate_model_tibble <- function(metrics, params, code){
  
  verify_is_tibble(metrics, "metrics")
  verify_is_tibble(params, "params")
  verify_is_string(code, "code")
  verify_code_in_tibble(params, code)
  
  # extract_hyperparams(params, code)
  
  .params <- pivot_param_tibble(params = params, code = code)
  
  .model <- pivot_metrics_tibble(metrics = metrics, code = code) |> 
    parse_metric_column() |> 
    link_model_params(params = .params)
  
  return(.model)
  
}