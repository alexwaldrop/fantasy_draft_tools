import os
import argparse
import logging
import pandas as pd

from utils import configure_logging

POSITIONS = ["RB", "WR", "TE", "QB"]

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

    # Path to VCF input file
    argparser_obj.add_argument("--input",
                               action="store",
                               type=file_type,
                               dest="input_file",
                               required=True,
                               help="Path to excel spreadsheet")

    # Path to VCF input file
    argparser_obj.add_argument("--output",
                               action="store",
                               type=str,
                               dest="output_file",
                               required=True,
                               help="Path to value cheatsheet output file.")

    # Path to recoded output file
    argparser_obj.add_argument("--qb",
                               action="store",
                               type=int,
                               dest="num_qb",
                               required=False,
                               default=12,
                               help="Expected total number of QBs drafted")

    # Path to recoded output file
    argparser_obj.add_argument("--rb",
                               action="store",
                               type=int,
                               dest="num_rb",
                               required=False,
                               default=48,
                               help="Expected total number of RBs drafted")

    # Path to recoded output file
    argparser_obj.add_argument("--wr",
                               action="store",
                               type=int,
                               dest="num_wr",
                               required=False,
                               default=48,
                               help="Expected total number of WRs drafted")

    # Path to recoded output file
    argparser_obj.add_argument("--te",
                               action="store",
                               type=int,
                               dest="num_te",
                               required=False,
                               default=12,
                               help="Expected total number of TEs drafted")

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

def parse_input_file(input_file):
    # Parse input file and check format
    input_df = pd.read_excel(input_file)

    # Check required columns present
    required_cols = ["Name", "Pos", "Rank", "Points", "AvgPick"]
    errors = False
    for required_col in required_cols:
        if required_col not in input_df.columns:
            # Output missing colname
            logging.error("Input file missing required col: {0}".format(required_col))
            errors = True

    if not errors:
        # Convert position column to uppercase
        input_df.Pos = input_df.Pos.str.upper()

        # Check to make sure POS column contains all RB, WR, TE, QB
        pos_available = [x.upper() for x in set(list(input_df.Pos))]

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
        raise IOError("Improperly formatted input file '{0}'. See above errors".format(input_file))

    return input_df

def calc_average_position_value(input_df, pos, num_players):
    # Get positional average for top-N players
    sorted_df = input_df.sort_values(by="Points", ascending=False)[input_df.Pos == pos]

    # Take top-N if N < total number of players at position, otherwise use all players
    if num_players < len(sorted_df.index):
        sorted_df = sorted_df.iloc[0:num_players]
    return sorted_df.Points.mean()
    #return sorted_df.Points.median()

def calc_points_above_replacement(row):
    return row["Points"] - row["Average Position Value"]

def calc_adp_inefficiency(row):
    return int(row["AvgPick"] - row["VBD Rank"])

def main():

    # Configure argparser
    argparser = argparse.ArgumentParser(prog="make_value_cheatsheet")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    configure_logging(args.verbosity_level)

    # Get names of input/output files
    in_file     = args.input_file
    out_file    = args.output_file
    num_qb      = args.num_qb
    num_rb      = args.num_rb
    num_wr      = args.num_wr
    num_te      = args.num_te

    logging.info("Using input variables:\n"
                 "QBs drafted: {0}\n"
                 "RBs drafted: {1}\n"
                 "WRs drafted: {2}\n"
                 "TEs drafted: {3}".format(num_qb,
                                           num_rb,
                                           num_wr,
                                           num_te))

    # Check input file
    input_df = parse_input_file(in_file)

    pos_avg_val = {}

    # Get average points for each position
    logging.info("Determining positional averages for each position...")
    for pos in POSITIONS:
        num_players = num_qb
        if pos == "RB":
            num_players = num_rb
        elif pos == "WR":
            num_players = num_wr
        elif pos == "TE":
            num_players = num_te
        pos_avg_val[pos] = calc_average_position_value(input_df, pos, num_players)

    # Add column for average position value
    input_df["Average Position Value"] = input_df.Pos.map(pos_avg_val)

    # Add column for points above average draftable replacement replacement
    input_df["Points Above Replacement"] = input_df.apply(calc_points_above_replacement, axis=1)

    # Sort by average value
    input_df.sort_values(by="Points Above Replacement", ascending=False, inplace=True)

    # Add column for value-based draft rank
    input_df["VBD Rank"] = input_df.reset_index().index + 1

    # Calculate ADP Inefficiency
    input_df["ADP Inefficiency"] = input_df.apply(calc_adp_inefficiency, axis=1)

    # Write to output file
    input_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    main()




