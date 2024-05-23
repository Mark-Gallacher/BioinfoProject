library(tidyverse)
library(here)

raw_df <- read_csv(here("data/DataForBioinfMScStudent-FakeIDs.csv"))

## need to tidy up the column names
## - remove the #
## - remove the spaces
tidy_strings <- function(string){
  
  out <- string |> 
    stringr::str_remove_all("#") |> 
    stringr::str_remove_all(" ") 
  
  return(out)
}

## tidy up the column names
## Some columns have their coded and full formats, so rename them. 
df <- raw_df |> 
  rename_with(.fn = tidy_strings) |> 
  rename("GenderFull" = "Gender.1") |> 
  rename("DiseaseSubtypeFull" = "DiseaseSubtype.1") |> 
  rename("VisibleHaemolysisFull" = "VisibleHaemolysis?")

## Get all the data and the labels - bring both so we remember how they were coded
data <- df |> 
  select(PseudoID:M530493, DiseaseSubtype, DiseaseSubtypeFull)

## Remove all the expression data - keep the disease subtypes
metadata <- df |> 
  select(!M546980:M530493)

HTdata <- data |> 
  filter(DiseaseSubtypeFull != "HV")

## save the csvs
write_csv(data, file = here("data/TidyData.csv"))
write_csv(HTdata, file = here("data/SubTypeData.csv"))
write_csv(metadata, file = here("data/MetaData.csv"))



## Next steps could be exploring the Arrow format to get practice with it. 

