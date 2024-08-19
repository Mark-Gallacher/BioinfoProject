source(here::here("r/process_merged_data_functions.R"))

average_colours = c("Macro" = "#FF2E00", "Micro" = "#FEA82F", "Weighted" = "#5448C8", "None" = "#423E3B")
subtype_colours = c("HV" = "#177e89", "PA" = "#084c61", "CS" = "#db3a34", "PHT" = "#ffc857", "PPGL" = "#323031")

average_shapes = c("Macro" = 15, "Micro" = 16, "Weighted" = 17, "None" = 18) 

metric_colours = c(
  "f1" = "#0077b6", 
  # "f1" = "#03045e", 
  # "f2" =  "#0077b6", 
  # "f3" =  "#00b4d8", 
  # "f4" =  "#90e0ef",
  "accuracy" = "#780000", 
  "balanced" = "#c1121f", 
  "cohen" = "#1b4332", 
  "matthew" = "#588157", 
  "precision" = "#3d0066", 
  "recall" = "#c86bfa"
  )

metric_labels = c(
  "f1" = "F1 Score", 
  "f2" = "F2 Score", 
  "f3" = "F3 Score", 
  "f4" = "F4 Score", 
  "accuracy" = "Accuracy",
  "balanced" = "Balanced Accuracy", 
  "cohen" = "Cohen Kappa", 
  "matthew" = "MCC", 
  "precision" = "Precision", 
  "recall" = "Recall"
)

model_colours <- c(
  "GNB" = "#f94144", 
  "GB" = "#f3722c", 
  "KNN" = "#f8961e", 
  "LSVM" = "#f9c74f", 
  "LG" = "#90be6d", 
  "RF" = "#43aa8b", 
  "SVM" = "#577590", 
  "DUM" = "grey30")


##### Custom report theme to use for report
report_theme <- function(base_size = 20, ...){
  
  ##use the minimal theme as a foundation, and edit where appropriate.
  ggplot2::theme_minimal(base_size = base_size) %+replace%
    ggplot2::theme(
      ## Make the legend underneath the title
      legend.position = "top",
      ## change the legend text size
      legend.text = element_text(size = rel(1.25)), 
      ## change the title and subtle format
      plot.title = element_text(face = "bold", size = rel(1.5), hjust = 0,  lineheight = 1, vjust = 1),
      plot.subtitle = element_text(size = rel(1.25), hjust = 0, lineheight = 1, vjust = 1),
      ## change the x and y axis, making them subtle for hmtl formating using ggtext
      axis.title.y = ggtext::element_markdown(size = rel(1.25), face = "bold", angle = 90),
      axis.title.x = ggtext::element_markdown(size = rel(1.25), face = "bold"),
      ## trying to align every title to the end of the plot instead of to the graph
      ## this doesn't seem to have a consistent effect even though hjust is set to zero.
      plot.title.position = "plot", 
      ...
    )
}

##### Function to Save graphs as PNG
save_graph <- function(graph, name, path = "", width = 8, height = 10){
  
  # require(svglite)
  
  ## save a graph as a svg.
  ## specify the folder, since we are creating a lot of graphs
  ## the specify the size
  ## ensure background is white
  ggplot2::ggsave(filename = paste0(name, ".png"), 
         plot = graph, 
         path = here::here("figures", path), 
         width = width, height = height, 
         # dpi = 320, 
         bg = "#FFFFFF")
  
}

## Assumes the first param is numeric and the second is categorical
plot_hyperparam_1n_1c <- function(code, param1, param2){
  
  .x_string <- rlang::as_string(rlang::ensym(param1))
  .x_title <- stringr::str_replace_all(.x_string, pattern = "_", replacement = " ") |> 
    stringr::str_to_title()
  
  .title <- all_names[code]
  
  get_summary_from_code(code) |> 
    ggplot(aes(x = {{ param1 }}, 
               y = mean_score, 
               colour = average_type, 
               group = average_type))+
    geom_point(alpha = .8, size = 1.5, show.legend = F) +
    stat_summary(fun = mean, geom = "line", linewidth = 2) +
    scale_x_continuous(.x_title) + 
    scale_y_continuous("Mean Metric Score") + 
    scale_colour_manual("Average Method", values = average_colours) + 
    ggtitle(paste0("Hyperparameters for ", .title)) +
    facet_wrap(vars({{ param2 }}))
}

## Assumes the param is numeric
plot_hyperparam_1n <- function(code, param1, log_x = FALSE){
  
  .x_string <- rlang::as_string(rlang::ensym(param1))
  .x_title <- stringr::str_replace_all(.x_string, pattern = "_", replacement = " ") |> 
    stringr::str_to_title()
  
  .title <- all_names[code]
  
  get_summary_from_code(code) |> 
    ggplot(aes(x = {{ param1 }}, 
               y = mean_score, 
               colour = average_type, 
               group = average_type))+
    geom_jitter(alpha = .5, size = 1.5, show.legend = T) +
    stat_summary(fun = mean, geom = "line", linewidth = 2) +
    scale_x_continuous(.x_title) + 
    scale_y_continuous("Mean Metric Score") + 
    scale_colour_manual("Average Method", values = average_colours) + 
    ggtitle(paste0("Hyperparameters for ", .title))
  
  }

## assume the param is categorical
plot_hyperparam_1c <- function(code, param1){
  
  .x_string <- rlang::as_string(rlang::ensym(param1))
  .x_title <- stringr::str_replace_all(.x_string, pattern = "_", replacement = " ") |> 
    stringr::str_to_title()
  
  .title <- all_names[code]
  
  get_summary_from_code(code) |> 
    ggplot(aes(x = {{ param1 }}, 
               y = mean_score, 
               colour = average_type, 
               group = average_type))+
    geom_jitter(alpha = .5, size = 1.5, show.legend = T) +
    stat_summary(fun = mean, geom = "line", linewidth = 2) +
    scale_x_discrete(.x_title) + 
    scale_y_continuous("Mean Metric Score") + 
    scale_colour_manual("Average Method", values = average_colours) + 
    ggtitle(paste0("Hyperparameters for ", .title))
  
}

## plot the models on the x-axis and the metric score on the y-axis - allow for different colouring ie by metric type, averaging type
plot_model_metrics <- function(df, colour, shape = "None", y_line = NULL, y = rel_mean, y_title = "Relative Mean Score"){
  
  point_size = 2.5
  main_line_width = 2
  minor_line_width = 1.5
  
  .colour_string <- rlang::as_string(rlang::ensym(colour))
  
  .shape_string <- rlang::as_string(rlang::ensym(shape))
  
  if(.colour_string == "metric_type"){
    
    .title <- "Comparing Metric Type Across Models"
    .colour_scale <- scale_colour_manual("Metric",
                                         values = metric_colours, 
                                         labels = metric_labels, 
                                         guide = guide_legend(override.aes = list(shape = 15, size = 4, alpha = 1)))
  }else{
    if(.colour_string == "average_type"){

    .title <- "Comparing Averaging Method Across Models"
    .colour_scale <- scale_colour_manual("Averaging Method",
                                         values = average_colours,
                                         guide = guide_legend(override.aes = list(shape = 15, size = 4, alpha = 1)))
    }
    else{
      stop(paste0("`colour` is not an expected value, received '", .colour_string, "' but was expecting 'average_type' or 'metric_type'"))
    }
  }
  

  .p <- df |> 
    filter(!model_code %in% c("DUM", "ERF")) |> 
    parse_metric_column() |>
    ggplot(aes(x = model_code, 
               y = {{ y }}, 
               colour = {{ colour }}, 
               shape = {{ shape }}, 
               group = metric))+
    geom_line(linewidth = main_line_width, show.legend = F, alpha = .5)
  
  if (!is.null(y_line) && is.numeric(y_line)){
    .p <- .p +
      geom_hline(yintercept = y_line, linewidth = minor_line_width)
  }
  
  if(.shape_string == "average_type"){
    .p <- .p +
      geom_point(size = point_size, size = 2, alpha = .8) +
      scale_shape_manual("Averaging Method", 
                         values = average_shapes, 
                         guide = guide_legend(override.aes = list(size = 4, alpha = 1))
                         )
    
  }else{
    
    .p <- .p + geom_point(size = point_size, shape = 15, size = 2, alpha = .8) 
  }
  
  .p <- .p +
    scale_x_discrete("Model Type") +
    scale_y_continuous(y_title) +
    ggtitle(.title) +
    .colour_scale +
    report_theme(base_size = 13)
  
  .p
}

plot_folds_func <- function(df, 
                            average_type = c("Micro", "Macro", "Weighted", "None"), 
                            y = score, 
                            base_size = 14, 
                            func){
  
  ## check the average_type is an acceptable choice
  average_type <- match.arg(average_type)
  
  .title <- paste0("Comparison of the Models with Metric with ", average_type, " Averaging") 
  
 .p <-  df |> 
    filter(!model_code %in% c("DUM", "ERF")) |> 
    filter(average_type == {{ average_type }})  |> 
    ggplot(aes(x = fold, y = {{ y }}, colour = model_code, group = model_id))+
    stat_summary(aes(group = model_code), 
                 geom = "line", 
                 fun = func, 
                 alpha = .7, 
                 linewidth = 2, 
                 fun.args = list(na.rm = T)) +
    ggtitle(.title) +
    scale_y_continuous(paste0(stringr::str_to_title(func), " Score")) +
    scale_x_discrete("CV Fold") +
    facet_wrap(~metric_type) +
    scale_colour_manual("Model", 
                        values = model_colours,
                        guide = guide_legend(override.aes = list(shape = 15, size = 4, alpha = 1))) + 
    report_theme(base_size)
  
 return(.p)
}


## plots the folds on the x-axis, metric_score on the y axis - to compare models - so coloured by model_code
plot_folds_average <- function(...) {
  
  plot_folds_func(..., func = "mean")

}

## plotting heatmaps of metrics correlations

plot_corr_heatmap <- function(df, method = c("pearson", "spearman")) {
  
  match.arg(method)
  
  library(ggcorrplot)
  
  .corr_df <- df |> 
    select(!starts_with("fbeta")) |> 
    select(where(is.numeric)) |> 
    rename_with( ~ str_replace_all(., "_", " ") |> str_to_title()) |>
    scale() |>
    cor(use = "complete.obs", 
        method = method)
  
  ggcorrplot(.corr_df, 
             lab = T,
             type = "lower", 
             title = paste(stringr::str_to_title(method),  "Correlation Between All the Metrics"),
             lab_col = "white")+
    scale_fill_gradientn("Corr",
                         colours = c("blue", "red"),
                         limit = c(signif(min(.corr_df), 2), 1)) +
    report_theme(base_size = 15, 
                 legend.key.width = unit(2, "cm")
    ) %+replace%
    theme(axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1), 
          axis.title.x = element_blank(), 
          axis.title.y = element_blank())
  
}


#### TESTING
# plot_hyper <- function(code){
#   FACET <-  FALSE
#   
#   .hyper <- all_hyper[[code]] 
#   
#   if(length(.hyper) < 1){
#     stop(
#       paste0("No Hyperparameters were found for '", code, 
#              "' !\nExpected a member of ", 
#              paste0(names(all_hyper), collapse = "', '"))
#     )
#   }
#   
#   if(is.na(.hyper[[1]])){
#     warning(paste0("This model appears to no have hyperparameters for '", code, "'\n"))
#     return(NULL)
#   }
#   
#   if(length(.hyper) == 1){ .x_axis <-  .hyper[[1]] }
#   
#   if(length(.hyper) == 2){ 
#     .x_axis <- .hyper[[1]]
#     .colour <-  .hyper[[2]]
#   }
#   
#   if(length(.hyper) == 3){ 
#     .x_axis <- .hyper[[1]]
#     .colour <- .hyper[[2]]
#     FACET <- TRUE
#   }
#   
#   return(.hyper)
# }
# 
# plot_hyper(all_codes[[3]])
# plot_hyper(all_codes[[1]])

