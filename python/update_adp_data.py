import argparse
import logging
import os
import pandas as pd
import yaml
import json

import utils
import constants as cols
import scraping

# Column names required to appear in response from ADP scrape
REQUIRED_ADP_COLS = ["name", "position", "adp", "stdev", "times_drafted"]

def configure_argparser(argparser_obj):

    def file_type(arg_string):
        """
        This function check both the existance of input file and the file size
        :param arg_string: file name as string
        :return: file name as string
        """
        if not os.path.exists(arg_string):
            err_msg = "%s does not exist! " \
                      "Please provide a valid file!" % arg_string
            raise argparse.ArgumentTypeError(err_msg)

        return arg_string

    # Path to projections spreadsheet
    argparser_obj.add_argument("--cheatsheet",
                               action="store",
                               type=file_type,
                               dest="input_file",
                               required=True,
                               help="Path to excel draft cheatsheet")

    # Path to league config
    argparser_obj.add_argument("--league-config",
                               action="store",
                               type=file_type,
                               dest="league_config",
                               required=True,
                               help="Path to league_config file")

    # Path to output file
    argparser_obj.add_argument("--out",
                               action="store",
                               type=str,
                               dest="output_file",
                               required=True,
                               help="Path to output file")

    # Path to output file
    argparser_obj.add_argument("--fuzzy-match-max",
                               action="store",
                               type=int,
                               dest="fuzzy_match_max",
                               required=False,
                               default=95,
                               help="Fuzzy match threshold above which player names are considered equivalent")

    # Path to output file
    argparser_obj.add_argument("--fuzzy-match-min",
                               action="store",
                               type=int,
                               dest="fuzzy_match_min",
                               required=False,
                               default=65,
                               help="Fuzzy match threshold below which player names are not considered equivalent")

    # Verbosity level
    argparser_obj.add_argument("-v",
                               action='count',
                               dest='verbosity_level',
                               required=False,
                               default=0,
                               help="Increase verbosity of the program."
                                    "Multiple -v's increase the verbosity level:\n"
                                    "0 = Errors\n"
                                    "1 = Errors + Warnings\n"
                                    "2 = Errors + Warnings + Info\n"
                                    "3 = Errors + Warnings + Info + Debug")

def main():
    # Configure argparser
    argparser = argparse.ArgumentParser(prog="update_adp_data")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    utils.configure_logging(args.verbosity_level)

    # Get names of input/output files
    in_file         = args.input_file
    out_file        = args.output_file
    league_config_file = args.league_config
    fuzzy_match_max = args.fuzzy_match_max
    fuzzy_match_min = args.fuzzy_match_min

    # Read config file
    with open(league_config_file, "r") as stream:
        league_config = yaml.safe_load(stream)

    league_size     = league_config["draft"]["league_size"]
    draft_year      = league_config["global"]["season"]
    score_format    = league_config["adp"]["league_format"]


    if not score_format in ["ppr", "half-ppr", "standard", "2qb"]:
        logging.error("Invalid score format in league config: {0}\n"
                      "Valid formats: ppr, half-ppr, standard, 2qb".format(score_format))
        raise IOError("Invalid score format in league config: {0}".format(score_format))

    # Create draft board
    input_df = pd.read_excel(in_file)
    utils.check_df_columns(input_df,
                           required_cols=[cols.NAME_FIELD, cols.POS_FIELD],
                           err_msg="Input cheatsheet missing either name or position column!")

    # Get ADP data for league type using API request
    request_url = "https://fantasyfootballcalculator.com/" \
                  "api/v1/adp/{0}?teams={1}&year={2}".format(score_format,
                                                             league_size,
                                                             draft_year)
    data = scraping.fetch_api_data(request_url)

    # Parse data into dataframe
    adp_df = pd.DataFrame(json.loads(data)["players"])
    logging.info("Pulled ADP for {0} players...".format(len(adp_df)))
    logging.debug(adp_df.head())

    # Check structure of response data to make sure required data is present
    utils.check_df_columns(adp_df,
                           required_cols=REQUIRED_ADP_COLS,
                           err_msg="ADP dataframe returned from web missing one or more required columns! Make sure URL is correct.")

    # Remove players that don't play positions you're interested in
    adp_df = adp_df[adp_df.position.isin(league_config["global"]["pos"])]
    logging.debug("{0} players remain after removing non-position players...".format(len(adp_df)))


    # Do fuzzy matching on names
    logging.info("Fuzzy matching ADP player names to names in input cheatsheet "
                 "(max_cutoff: {0}, min_cutoff: {1}".format(fuzzy_match_max,
                                                     fuzzy_match_min))
    def match_name(row, input_df):
        return scraping.match_reference_player_name(row["name"],
                                                    row["position"],
                                                    input_df[cols.NAME_FIELD].tolist(),
                                                    input_df[cols.POS_FIELD].tolist(),
                                                    fuzzy_match_max,
                                                    fuzzy_match_min)

    # Apply fuzzing matching
    old_adp_df = adp_df.copy()
    adp_df["name"] = adp_df.apply(match_name, axis=1, input_df=input_df)

    # Remove players that weren't found in reference dataset
    dropped_players = old_adp_df[pd.isnull(adp_df["name"])]
    if len(dropped_players):
        logging.warning("ADP players not matching player in input spreadsheet: \n{0}".format(dropped_players))

    # Remove unmatched players from ADP list
    adp_df = adp_df[~pd.isnull(adp_df["name"])]

    # Subset to include on informative columns
    adp_df = adp_df[REQUIRED_ADP_COLS]

    # Rename ADP columns to standard draftboard names
    colmap = {
        "name": cols.NAME_FIELD+"_z",
        "position" : cols.POS_FIELD+"_z",
        "adp": cols.ADP_FIELD,
        "stdev": cols.ADP_SD_FIELD,
        "times_drafted": cols.ADP_TIMES_DRAFTED_FIELD
    }
    adp_df = adp_df.rename(columns=colmap)

    # Add merge key to prevent writing ADP to players with identical names
    input_df["MergeName"] = input_df[cols.NAME_FIELD] + input_df[cols.POS_FIELD]
    adp_df["MergeName"] = adp_df[colmap["name"]] + adp_df[colmap["position"]]

    # Deduplicate if necessary
    input_df = utils.drop_duplicates_and_warn(input_df, id_col="MergeName")
    adp_df   = utils.drop_duplicates_and_warn(adp_df, id_col="MergeName")


    # Merge data frames into single dataset and raise error if overlapping colnames
    merged_df = input_df.merge(adp_df,
                               how='left',
                               on="MergeName",
                               validate='one_to_one',
                               suffixes=("_x", "_y"))

    # Remove overlapping columns from previous input_df (other than name/pos)
    cols_to_include = [col for col in merged_df.columns if col[-2:] not in ["_z", "_x"] and col != "MergeName"]
    merged_df = merged_df[cols_to_include]

    # Remove '_y" from any overlapping ADP columns
    colmap = {col: col.replace("_y","") for col in merged_df.columns}
    merged_df = merged_df.rename(columns=colmap)

    # Fill in missing ADPs
    max_adp = merged_df[cols.ADP_FIELD].max() + 1
    merged_df[cols.ADP_FIELD].fillna(max_adp, inplace=True)

    # Fill in missing ADP StdDev by calculating average of bottom 5 players
    adp_stdev_fill = merged_df.sort_values(by=cols.ADP_FIELD, ascending=False)[cols.ADP_SD_FIELD].dropna()
    adp_stdev_fill = adp_stdev_fill[0:5].mean()
    merged_df[cols.ADP_SD_FIELD].fillna(adp_stdev_fill, inplace=True)

    # Write to output file
    merged_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    main()