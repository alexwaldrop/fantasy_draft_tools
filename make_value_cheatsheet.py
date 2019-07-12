import os
import argparse
import logging
import pandas as pd

import utils
import constants as cols

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

    # Next available player group size
    argparser_obj.add_argument("--next-player-group-size",
                               action="store",
                               type=int,
                               dest="next_player_group_size",
                               default=3,
                               required=False,
                               help="Number of players to consider drafting in next round for VONAT")

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
    required_cols = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD, cols.ADP_FIELD]
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
        raise IOError("Improperly formatted input file '{0}'. See above errors".format(input_file))

    return input_df


def calc_position_replacement_value(input_df, pos, num_players):
    # Get positional average for top-N players
    sorted_df = input_df.sort_values(by=cols.POINTS_FIELD, ascending=False)[input_df[cols.POS_FIELD] == pos]

    # Take top-N if N < total number of players at position, otherwise use all players
    if num_players < len(sorted_df.index):
        sorted_df = sorted_df.iloc[0:num_players+1]
    return sorted_df.Points.min()


def main():
    # Configure argparser
    argparser = argparse.ArgumentParser(prog="make_value_cheatsheet")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    utils.configure_logging(args.verbosity_level)

    # Get names of input/output files
    in_file     = args.input_file
    out_file    = args.output_file
    num_qb      = args.num_qb
    num_rb      = args.num_rb
    num_wr      = args.num_wr
    num_te      = args.num_te

    # Number of picks in between draft selections
    league_size =args.league_size

    # Number of players to consider when predicting autodraft
    next_player_group_size = args.next_player_group_size
    
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

    # Get average points for each position
    logging.info("Determining positional averages for each position...")
    pos_repl_val    = {}
    num_player_map  = {"QB": num_qb, "RB": num_rb, "WR": num_wr, "TE": num_te}
    for pos in num_player_map:
        pos_repl_val[pos] = calc_position_replacement_value(input_df, pos, num_player_map[pos])

    # Add column for value over positional replacement for each player
    input_df[cols.REPLACEMENT_VALUE_FIELD] = input_df[cols.POS_FIELD].map(pos_repl_val)

    # Add column for points above average draftable replacement
    input_df[cols.VORP_FIELD] = input_df[cols.POINTS_FIELD] - input_df[cols.REPLACEMENT_VALUE_FIELD]

    # Sort by average value
    input_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

    # Add column for value-based draft rank
    input_df[cols.VORP_RANK_FIELD] = input_df.reset_index().index + 1

    # Add column for draft rank
    input_df.sort_values(by=cols.ADP_FIELD, inplace=True)
    input_df[cols.DRAFT_RANK_FIELD] = input_df.reset_index().index + 1

    # Set autodraft generation to start the current pick at 1 if no players listed as drafted
    curr_pick = 1

    # Remove drafted players if draft status column present
    if cols.DRAFT_STATUS in input_df.columns:

        # Get number of players that have been drafted so far
        num_drafted = input_df[~pd.isnull(input_df[cols.DRAFT_STATUS])].shape[0]

        # Calculate current round and pick position based on num drafted
        curr_round = ((num_drafted + 1) // league_size) + 1
        curr_pick = num_drafted + 1

        # Log basic info
        logging.warning("Removing autodrafted players! "
                        "Draft status inferred from cheatsheet:\n"
                        "Current draft pick: {0}\n"
                        "Current draft round: {1}\n"
                        "Current slot on the board: {2}".format(curr_pick,
                                                                curr_round,
                                                                utils.get_draft_slot(curr_pick, league_size)))

        # Remove drafted players
        input_df = input_df[pd.isnull(input_df[cols.DRAFT_STATUS])]

    # Generate expected autodraft slots for undrafted players
    draft_slots = pd.Series(utils.generate_autodraft_slots(num_players=len(input_df),
                                                           league_size=league_size,
                                                           curr_pick=curr_pick)).values
    input_df[cols.DRAFT_SLOT_FIELD] = draft_slots

    # Calculate ADP Inefficiency
    input_df[cols.ADP_INEFF_FIELD] = input_df[cols.DRAFT_RANK_FIELD] - input_df[cols.VORP_RANK_FIELD]

    # Get name of next available player
    input_df[cols.NEXT_DRAFTABLE_FIELD] = input_df.apply(utils.calc_next_best_available, axis=1, input_df=input_df)

    # Calculate Value of Next Available Player
    input_df[cols.NEXT_DRAFTABLE_PTS_FIELD] = input_df.apply(utils.calc_value_of_next_available, axis=1, input_df=input_df)

    # Calculate Value over next available player (Opportunity cost)
    input_df[cols.VONA_FIELD] = input_df[cols.POINTS_FIELD] - input_df[cols.NEXT_DRAFTABLE_PTS_FIELD]

    # Get name of next available player
    input_df[cols.NEXT_DRAFTABLE_GROUP_FIELD] = input_df.apply(utils.calc_next_best_available_group, axis=1,
                                                               input_df=input_df, group_size=next_player_group_size)

    # Calculate Value of Next Available Player
    input_df[cols.NEXT_DRAFTABLE_GROUP_PTS_FIELD] = input_df.apply(utils.calc_value_of_next_available_group, axis=1,
                                                                   input_df=input_df, group_size=next_player_group_size)

    # Calculate Value over next available player (Opportunity cost)
    input_df[cols.VONAG_FIELD] = input_df[cols.POINTS_FIELD] - input_df[cols.NEXT_DRAFTABLE_GROUP_PTS_FIELD]

    # Calculate Real Value (value adjusted for opportunity cost of not choosing player)

    # Sort by average value
    input_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

    # Write to output file
    input_df.to_excel(out_file, index=False)

if __name__ == "__main__":
    main()




