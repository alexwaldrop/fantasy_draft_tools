library(ffanalytics)
library(yaml)
library(WriteXLS)
library(optparse)

parser = OptionParser(usage = "\n%prog [options] --config <config_file> --output_file <output_file>", 
                      description = "Program for initializing fantasy football VBD cheatsheet from projections scraped from public sources",
                      prog="Rscript init_draft_cheatsheet.R")
parser = add_option(object=parser, opt_str=c("--config_file"), default=NULL, type="character",
                    help="[REQUIRED] Path to config file")
parser = add_option(object=parser, opt_str=c("--output_file"), default=NULL, type="character",
                    help="[REQUIRED] Excel draft cheatsheet output path")

# Parse command line args
argv = parse_args(parser)

# Check input file exists
if(!file.exists(argv$config_file)){
  stop(paste0("Error: ",argv$config_file), " does not exist! Check your file path and name.")
}

# Parse config
print("Parsing config file...")
conf = read_yaml(argv$config_file)
output_file = argv$output_file

# Scrape projections from internet
print(paste0("Scraping projections from sources: ", conf$projections$sources))
my_scrape <- scrape_data(src = conf$projections$sources, 
                         pos = conf$global$pos, 
                         season = conf$global$season, 
                         week = 0)

# Create projections from league settings
print("Generating projections from league scoring...")
my_projections <- projections_table(my_scrape, 
                                    conf$scoring,
                                    vor_baseline = unlist(conf$vorp))

# Add player information (name, etc.), consensus ranking, and risk measure
print("Adding additional info...")
my_projections <- add_player_info(my_projections) %>% add_ecr() %>% add_risk()

# Include only the average rankings for each player
my_projections <- filter(my_projections, avg_type == conf$projections$type) %>% arrange(desc(points_vor))

# Add a field for combined player name
my_projections$Name <- sapply(1:nrow(my_projections), function(i) paste(my_projections$first_name[i], my_projections$last_name[i]))


# Add empty columns for draft selections
my_projections$Drafted <- rep(NA, nrow(my_projections))
my_projections[["My Picks"]] <- rep(NA, nrow(my_projections))
my_projections[["Potential Pick"]] <- rep(NA, nrow(my_projections))

# Reorder columns 
req_columns = c("Name", "Drafted", "My Picks", "Potential Pick")
exclude_cols = c("first_name", "last_name", "id", "avg_type")
include_cols = colnames(my_projections)[-which(colnames(my_projections) %in% c(req_columns, exclude_cols))]
final_proj = select(my_projections, c(req_columns, include_cols))

# Write output to excel file
WriteXLS(final_proj, output_file)


