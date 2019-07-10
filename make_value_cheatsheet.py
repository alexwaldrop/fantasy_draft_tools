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

    # Path to projections spreadsheet
    argparser_obj.add_argument("--input",
                               action="store",
                               type=file_type,
                               dest="input_file",
                               required=True,
                               help="Path to excel spreadsheet")

    # Path to output file
    argparser_obj.add_argument("--output",
                               action="store",
                               type=str,
                               dest="output_file",
                               required=True,
                               help="Path to value cheatsheet output file.")

    # Number of QB to consider for VORP baseline
    argparser_obj.add_argument("--qb",
                               action="store",
                               type=int,
                               dest="num_qb",
                               required=False,
                               default=12,
                               help="Expected total number of QBs drafted")

    # Number of RB to consider for VORP baseline
    argparser_obj.add_argument("--rb",
                               action="store",
                               type=int,
                               dest="num_rb",
                               required=False,
                               default=48,
                               help="Expected total number of RBs drafted")

    # Number of WR to consider for VORP baseline
    argparser_obj.add_argument("--wr",
                               action="store",
                               type=int,
                               dest="num_wr",
                               required=False,
                               default=48,
                               help="Expected total number of WRs drafted")

    # Number of TE to consider for VORP baseline
    argparser_obj.add_argument("--te",
                               action="store",
                               type=int,
                               dest="num_te",
                               required=False,
                               default=12,
                               help="Expected total number of TEs drafted")

    # Number of picks until next pick
    argparser_obj.add_argument("--league-size",
                               action="store",
                               type=int,
                               dest="league_size",
                               required=True,
                               help="League size")

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
    required_cols = ["Name", "Pos", "Points", "ADP"]
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

def calc_position_replacement_value(input_df, pos, num_players):
    # Get positional average for top-N players
    sorted_df = input_df.sort_values(by="Points", ascending=False)[input_df.Pos == pos]

    # Take top-N if N < total number of players at position, otherwise use all players
    if num_players < len(sorted_df.index):
        sorted_df = sorted_df.iloc[0:num_players+1]
    return sorted_df.Points.min()

def calc_points_above_replacement(row):
    return row["Points"] - row["Replacement Value"]

def get_next_best_available(row, input_df=None):
    pos = row["Pos"]
    rank = row["Draft Rank"]
    draft_slot = row["Draft Slot"]

    # Get list of players available after current player
    sorted_df = input_df.sort_values(by="Draft Rank")[input_df["Draft Rank"] > rank]

    # Get draft rank of next available player at current draft slot
    next_player_ranks = list(sorted_df[sorted_df["Draft Slot"] == draft_slot]["Draft Rank"])

    # Remove players that likely would be drafted before next turn (if another turn is possible)
    if next_player_ranks:
        sorted_df = sorted_df[sorted_df["Draft Rank"] >= next_player_ranks[0]]

    # Get next best player at position
    sorted_df = sorted_df[sorted_df.Pos == pos].sort_values(by="Points", ascending=False)
    next_avail_name_list = list(sorted_df.Name)
    next_avail_points_list = list(sorted_df.Points)

    # If no players available at the position
    if not next_avail_name_list:
        return None, 0

    return next_avail_name_list[0], next_avail_points_list[0]

def calc_next_best_available(row, input_df=None):
   name, points = get_next_best_available(row, input_df)
   return name

def calc_value_of_next_available(row, input_df=None):
    name, points = get_next_best_available(row, input_df)
    return points

def get_draft_slots(num_players, league_size):
    draft_slots = []
    pick = 1
    for i in range(num_players):
        # Add next draft slot to list
        draft_slots.append(pick)

        # Determine current round
        round = (i // league_size) + 1

        # Determine whether we're going forwards or backwards
        reverse = round % 2 == 0

        # Repeat picks on the ends for first and last picks
        if pick == 1 and reverse:
            pick = 1
        elif pick == league_size and not reverse:
           pick = 12
        # Otherwise just increment in the correct direction
        else:
            pick = pick - 1 if reverse else pick + 1

    return draft_slots

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

    # Number of picks in between draft selections
    league_size =args.league_size
    
    # Log basic info
    logging.info("Using input variables:\n"
                 "QBs drafted: {0}\n"
                 "RBs drafted: {1}\n"
                 "WRs drafted: {2}\n"
                 "TEs drafted: {3}\n"
                 "League size: {4}".format(num_qb,
                                           num_rb,
                                           num_wr,
                                           num_te,
                                           league_size))

    # Check input file for formatting
    input_df = parse_input_file(in_file)

    pos_repl_val = {}

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
        pos_repl_val[pos] = calc_position_replacement_value(input_df, pos, num_players)

    # Add column for value over positional replacement for each player
    input_df["Replacement Value"] = input_df.Pos.map(pos_repl_val)

    # Add column for draft rank
    input_df.sort_values(by="ADP", inplace=True)
    input_df["Draft Rank"] = input_df.reset_index().index + 1

    draft_slots = pd.Series(get_draft_slots(num_players=len(input_df), league_size=league_size)).values
    input_df["Draft Slot"] = draft_slots

    # Add column for points above average draftable replacement replacement
    input_df["VORP"] = input_df.apply(calc_points_above_replacement, axis=1)

    # Sort by average value
    input_df.sort_values(by="VORP", ascending=False, inplace=True)

    # Add column for value-based draft rank
    input_df["VORP Rank"] = input_df.reset_index().index + 1

    # Calculate ADP Inefficiency
    input_df["ADP Inefficiency"] = input_df["Draft Rank"] - input_df["VORP Rank"]

    # Get name of next available player
    input_df["Next Best Draftable"] = input_df.apply(calc_next_best_available, axis=1, input_df=input_df)

    # Calculate Value of Next Available Player
    input_df["Next Best Draftable Points"] = input_df.apply(calc_value_of_next_available, axis=1, input_df=input_df)

    # Calculate Value over next available player (Opportunity cost)
    input_df["VONA"] = input_df["Points"] - input_df["Next Best Draftable Points"]

    # Write to output file
    input_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    main()




