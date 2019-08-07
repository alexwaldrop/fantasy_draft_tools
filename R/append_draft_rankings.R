library(WriteXLS)
library(optparse)
library(tidyr)
library(dplyr)
library(readxl)

parser = OptionParser(usage = "\n%prog [options] --projections <projections_file> --rankings <rankings_file> --output_file <output_file>", 
                      description = "Program for adding draft rankings data to an existing VBD draft cheatsheet",
                      prog="Rscript append_draft_rankings.R")
parser = add_option(object=parser, opt_str=c("--projections"), default=NULL, type="character",
                    help="[REQUIRED] Path to projections file")
parser = add_option(object=parser, opt_str=c("--rankings"), default=NULL, type="character",
                    help="[REQUIRED] Path to rankings file")
parser = add_option(object=parser, opt_str=c("--output_file"), default=NULL, type="character",
                    help="[REQUIRED] Excel draft cheatsheet output path")

# Parse command line args
argv = parse_args(parser)

# Check projections input file exists
if(!file.exists(argv$projections)){
  stop(paste0("Error: ",argv$projections), " does not exist! Check your file path and name.")
}
# Check input file exists
if(!file.exists(argv$rankings)){
  stop(paste0("Error: ",argv$rankings), " does not exist! Check your file path and name.")
}

# Read in projections
projections_file = argv$projections
my_projections = read_excel(projections_file)
if(! "Name" %in% colnames(my_projections)){
  stop("Projection file missing 'Name' column!")
}
my_projections$Name = as.character(my_projections$Name)

# Read in rankings
rankings_file = argv$rankings
ranks_df = read.csv(rankings_file)
if(! "Name" %in% colnames(ranks_df)){
  stop("Rankings file missing 'Name' column!")
}else if(! "Rank" %in% colnames(ranks_df)){
  stop("Rankings file missing 'Rank' column!")
}
ranks_df = select(ranks_df, Name, Rank)
colnames(ranks_df) <- c("Name", "ADP")
ranks_df$Name = as.character(ranks_df$Name)
old_ranks = ranks_df

# Do fuzzy match between player names
ranks_df$Name <- sapply(ranks_df$Name, function(name){
  if(name %in% my_projections$Name){
    return(name)
  }
  sim_idx = agrep(name, my_projections$Name, 0.15)
  if(length(sim_idx) == 1){
    return(my_projections$Name[sim_idx])
  }else{
    print(paste("Can't match player:", name))
    return(name)
  }
})
ranks_df$Name <- as.character(ranks_df$Name)

# Check to make sure each name appears only once
if(length(which(table(ranks_df$Name) > 1))){
  print(table(ranks_df$Name)[which(table(ranks_df$Name) > 1)])
  stop("One or more players fuzzy matched too many times. See above for detals.")
}

# Merge Ranks sheet
my_projections <- left_join(my_projections, ranks_df, by="Name")

# Throw error if any in top 150 projected don't have ranking
if(min(which(is.na(my_projections$ADP))) < 150){
  stop("At least one player inside top-150 projections doesn't have a ranking. This is probably an error.")
}

# Remove NAs
max_adp = max(na.omit(my_projections$ADP)) + 1
my_projections$ADP[is.na(my_projections$ADP)] <- max_adp

# Write output to excel file
output_file = argv$output_file
WriteXLS(my_projections, output_file)


