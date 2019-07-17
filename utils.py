import sys
import logging
from statistics import mean
import re

def configure_logging(verbosity):
    # Setting the format of the logs
    FORMAT = "[%(asctime)s] %(levelname)s: %(message)s"

    # Configuring the logging system to the lowest level
    logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stderr)

    # Defining the ANSI Escape characters
    BOLD = '\033[1m'
    DEBUG = '\033[92m'
    INFO = '\033[94m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    END = '\033[0m'

    # Coloring the log levels
    if sys.stderr.isatty():
        logging.addLevelName(logging.ERROR, "%s%s%s%s%s" % (BOLD, ERROR, "FD_TOOLS_ERROR", END, END))
        logging.addLevelName(logging.WARNING, "%s%s%s%s%s" % (BOLD, WARNING, "FD_TOOLS_WARNING", END, END))
        logging.addLevelName(logging.INFO, "%s%s%s%s%s" % (BOLD, INFO, "FD_TOOLS_INFO", END, END))
        logging.addLevelName(logging.DEBUG, "%s%s%s%s%s" % (BOLD, DEBUG, "FD_TOOLS_DEBUG", END, END))
    else:
        logging.addLevelName(logging.ERROR, "FD_TOOLS_ERROR")
        logging.addLevelName(logging.WARNING, "FD_TOOLS_WARNING")
        logging.addLevelName(logging.INFO, "FD_TOOLS_INFO")
        logging.addLevelName(logging.DEBUG, "FD_TOOLS_DEBUG")

    # Setting the level of the logs
    level = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][verbosity]
    logging.getLogger().setLevel(level)

def get_draft_slot(pick_num, league_size):
    # Source: https://math.stackexchange.com/questions/298973/formula-for-snake-draft-pick-numbers
    # Get draft slot of current pick given a league size
    curr_round = ((pick_num - 1) // league_size) + 1

    # Handle even rounds (count forward)
    if curr_round % 2 == 0:
        draft_slot = (curr_round*league_size) + 1 - pick_num

    # Handle odd rounds (count backwards)
    else:
        draft_slot = pick_num + league_size - (league_size*curr_round)

    return draft_slot

def generate_autodraft_slots(num_players, league_size, curr_pick=1):
    return [get_draft_slot(curr_pick + i, league_size) for i in range(num_players)]

def get_next_best_available_player_group(row, input_df=None, group_size=3):
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

    # Return group size
    num_to_take = min(group_size, len(next_avail_name_list))
    return ", ".join(next_avail_name_list[0:num_to_take]), mean(next_avail_points_list[0:num_to_take])

def get_next_best_available_player(row, input_df=None):
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
   name, points = get_next_best_available_player(row, input_df)
   return name

def calc_value_of_next_available(row, input_df=None):
    name, points = get_next_best_available_player(row, input_df)
    return points

def calc_next_best_available_group(row, input_df=None, group_size=3):
   name, points = get_next_best_available_player_group(row, input_df, group_size)
   return name

def calc_value_of_next_available_group(row, input_df=None, group_size=3):
    name, points = get_next_best_available_player_group(row, input_df, group_size)
    return points

def clean_string_for_file_name(value):
    clean_val = re.sub('[^\w\s-]', '', value).strip().lower()
    clean_val = re.sub('[-\s]+', '-', clean_val)
    return clean_val
