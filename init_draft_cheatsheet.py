import os
import argparse
import logging
import pandas as pd
import numpy as np

import utils
import constants as cols

POSITIONS = ["RB", "WR", "TE", "QB"]
NORMALIZED_NAME_FIELD = "Normalized Player Name"


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
    argparser_obj.add_argument("--proj",
                               action="store",
                               type=file_type,
                               dest="proj_file",
                               required=True,
                               help="Path to player projections spreadsheet")

    # Path to rankings spreadsheet
    argparser_obj.add_argument("--ranks",
                               action="store",
                               type=file_type,
                               dest="rank_file",
                               required=True,
                               help="Path to rankings spreadsheet")

    # Path to risk spreadsheet
    argparser_obj.add_argument("--risk",
                               action="store",
                               type=file_type,
                               dest="risk_file",
                               required=True,
                               help="Path to risk spreadsheet")

    # Path to output file
    argparser_obj.add_argument("--output",
                               action="store",
                               type=str,
                               dest="output_file",
                               required=True,
                               help="Path to value cheatsheet output file.")

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


def parse_proj_file(input_file):
    # Parse input file and check format
    input_df = pd.read_excel(input_file)

    # Check required columns present
    required_cols = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD]
    errors = False
    for required_col in required_cols:
        if required_col not in input_df.columns:
            # Output missing colname
            logging.error("Input file missing required col: {0}".format(required_col))
            errors = True

    if not errors:
        # Convert position column to uppercase
        input_df[cols.POS_FIELD] = input_df[cols.POS_FIELD].str.upper()

        # Check to make sure POS column contains all RB, WR, TE, QB
        pos_available = [x.upper() for x in set(list(input_df[cols.POS_FIELD]))]

        for pos in POSITIONS:
            if pos not in pos_available:
                logging.error("Missing players of position type: {0}".format(pos))
                errors = True

        # Check to make sure Pos column contains only RB, WR, TE, QB
        for pos in pos_available:
            if pos not in POSITIONS:
                logging.error("One or more players contains invalid position: {0}".format(pos))
                errors = True

    if errors:
        raise IOError("Improperly formatted projections file '{0}'. See above errors".format(input_file))

    # Standardize name field
    input_df[NORMALIZED_NAME_FIELD] = input_df[cols.NAME_FIELD].map(lambda x: utils.clean_string_for_file_name(x))

    return input_df


def parse_rankings_file(input_file):
    # Parse input file and check format
    input_df = pd.read_excel(input_file)

    # Check required columns present
    required_cols = [cols.NAME_FIELD, cols.ADP_FIELD]
    errors = False
    for required_col in required_cols:
        if required_col not in input_df.columns:
            # Output missing colname
            logging.error("Ranks file missing required col: {0}".format(required_col))
            errors = True

    if errors:
        raise IOError("Improperly formatted rankings file '{0}'. See above errors".format(input_file))

    # Standardize name field
    input_df[NORMALIZED_NAME_FIELD] = input_df[cols.NAME_FIELD].map(lambda x: utils.clean_string_for_file_name(x))

    return input_df

def parse_risk_file(input_file):
    # Parse input file and check format
    input_df = pd.read_excel(input_file)

    # Check required columns present
    required_cols = [cols.NAME_FIELD, cols.RISK_FIELD]
    errors = False
    for required_col in required_cols:
        if required_col not in input_df.columns:
            # Output missing colname
            logging.error("Risk file missing required col: {0}".format(required_col))
            errors = True

    if errors:
        raise IOError("Improperly formatted risk file '{0}'. See above errors".format(input_file))

    # Standardize name field
    input_df[NORMALIZED_NAME_FIELD] = input_df[cols.NAME_FIELD].map(lambda x: utils.clean_string_for_file_name(x))

    return input_df


def check_name(name, rank_names):
    if name not in rank_names:
        logging.error("Could not find match for player '{0}' in provided rankings!".format(name))
        return True
    return False

def check_names(proj_df, other_df, other_type):
    # Check to make sure all names in the projections map to someone in the rankings
    rank_names = list(other_df[NORMALIZED_NAME_FIELD])

    name_matches = proj_df[NORMALIZED_NAME_FIELD].map(lambda x: check_name(x, rank_names))
    if name_matches.any():
        raise IOError("One or more players in {0} spreadhsheet not found in rankings file!".format(other_type))

def merge_sheets(proj_df, other_df, other_type):
    # Merge dataframes
    output_df = proj_df.merge(other_df, how='left', on=NORMALIZED_NAME_FIELD, validate='one_to_one')
    if len(output_df) != len(proj_df):
        logging.error("Something went wrong merging the {0} spreadsheet. "
                      "Projections has {1} rows and merged output has {2} cols".format(other_type,
                                                                                       len(proj_df),
                                                                                       len(output_df)))

    # Fix column names
    colnames = []
    for col in output_df.columns:
        if col.endswith("_x") and not col.startswith(cols.ADP_FIELD) and not col.startswith(cols.RISK_FIELD):
            colnames.append(col[0:-2])
        elif col.endswith("_y") and (col.startswith(cols.ADP_FIELD) or col.startswith(cols.RISK_FIELD)):
            colnames.append(col[0:-2])
        else:
            colnames.append(col)
    output_df.columns = colnames
    return(output_df)

def main():
    # Configure argparser
    argparser = argparse.ArgumentParser(prog="init_draft_cheatsheet")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    utils.configure_logging(args.verbosity_level)

    # Get names of input/output files
    proj_file = args.proj_file
    rank_file = args.rank_file
    risk_file = args.risk_file
    out_file = args.output_file

    logging.info("Using projections file: {0}".format(proj_file))
    logging.info("Using rankings file: {0}".format(rank_file))

    # Check projections input file for formatting and read into pandas
    proj_df = parse_proj_file(proj_file)

    # Check rankings input file for formatting and read into pandas
    rank_df = parse_rankings_file(rank_file)

    # Check risk file for formatting
    risk_df = parse_risk_file(risk_file)

    # Check to make sure all names in projection spreadsheet exist in rankings, risk spreadsheets
    check_names(proj_df, rank_df, other_type="Rankings")
    check_names(proj_df, risk_df, other_type="Risk")

    # Merge dataframes
    output_df = merge_sheets(proj_df, rank_df, other_type="Rankings")
    output_df = merge_sheets(output_df, risk_df, other_type="Risk")

    # Get only required columns and add columns for entering draft picks
    col_order = proj_df.columns.tolist()
    col_order.remove(NORMALIZED_NAME_FIELD)
    for col in [cols.RUN_SIM_DRAFT, cols.MY_PICKS, cols.DRAFT_STATUS]:
        if col in output_df.columns:
            col_order.remove(col)
        output_df[col] = np.nan
        col_order.insert(3, col)

    # Reorder columns
    output_df = output_df[col_order]

    # Sort by ADP
    output_df.sort_values(by="ADP", inplace=True)

    # Write to output file
    output_df.to_excel(out_file, index=False)


if __name__ == "__main__":
    main()




