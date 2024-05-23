source(here::here("r/process_merged_data_functions.R"))

average_colours = c("Macro" = "#FF2E00", "Micro" = "#FEA82F", "Weighted" = "#5448C8", "None" = "#423E3B")
subtype_colours = c("HV" = "#177e89", "PA" = "#084c61", "CS" = "#db3a34", "PHT" = "#ffc857", "PPGL" = "#323031")



##### Custom report theme to use for report
report_theme <- function(base_size = 20){
  
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
      plot.title.position = "plot"
    )
}

##### Function to Save graphs as PNG
save_graph <- function(graph, name, path = "", width = 8, height = 10){
  
  require(svglite)
  
  ## save a graph as a svg.
  ## specify the folder, since we are creating a lot of graphs
  ## the specify the size
  ## ensure background is white
  ggsave(filename = paste0(name, ".svg"), 
         plot = graph, 
         path = here::here("Figures/svg", path), 
         width = width, height = height, 
         # dpi = 320, 
         bg = "#FFFFFF")
  
}

## Assumes the first param is numeric and the second is categorical
plot_hyperparam_1n_1c <- function(code, param1, param2){
  
  .x_string <- rlang::as_string(rlang::ensym(param1))
  .x_title <- stringr::str_replace(.x_string, pattern = "_", replacement = " ") |> 
    stringr::str_to_title()
  
  .title <- all_names[code]
  
  get_summary_from_code(code) |> 
    ggplot(aes(x = {{ param1 }}, 
               y = mean_score, 
               colour = average_type, 
               group = average_type))+
    geom_point(alpha = .8, size = 1.5, show.legend = F) +
    # geom_smooth(se = F, linewidth = 2) +
    stat_summary(fun = mean, geom = "line", linewidth = 2) +
    scale_x_continuous(.x_title) + 
    scale_y_continuous("Mean Metric Score") + 
    scale_colour_manual("Average Method", values = average_colours) + 
    ggtitle(paste0("Hyperparameters for ", .title)) +
    facet_wrap(vars({{ param2 }}))
}

## Assumes the first param is numeric and the second is categorical
plot_hyperparam_1n <- function(code, param1){
  
  .x_string <- rlang::as_string(rlang::ensym(param1))
  .x_title <- stringr::str_replace(.x_string, pattern = "_", replacement = " ") |> 
    stringr::str_to_title()
  
  .title <- all_names[code]
  
  get_summary_from_code(code) |> 
    ggplot(aes(x = {{ param1 }}, 
               y = mean_score, 
               colour = average_type, 
               group = average_type))+
    geom_point(alpha = .5, size = 1.5, show.legend = T) +
    stat_summary(fun = mean, geom = "line", linewidth = 2) +
    scale_x_continuous(.x_title) + 
    scale_y_continuous("Mean Metric Score") + 
    scale_colour_manual("Average Method", values = average_colours) + 
    ggtitle(paste0("Hyperparameters for ", .title))
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

