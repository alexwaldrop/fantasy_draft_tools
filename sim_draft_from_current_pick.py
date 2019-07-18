import os
import argparse
import logging
import pandas as pd
from copy import deepcopy
import random

import utils
import constants as cols

POSITIONS = ["RB", "WR", "TE", "QB"]

# TODO: Read hard-coded config from external file
league_config = {"start_rb": 2, "start_wr": 2, "start_qb": 1, "start_te": 1, "start_flex": 1,
                 "flex_pos": ["RB", "WR"], "min_rb": 4, "min_wr": 4, "min_qb": 1, "min_te": 1,
                 "max_rb": 7, "max_wr": 7, "max_qb": 1, "max_te": 1,
                 "team_size": 14, "league_size": 12}

class Team:
    def __init__(self, league_config):
        self.league_config = league_config

        self.players = {"QB": [],
                        "RB": [],
                        "WR": [],
                        "TE": []}

        self.draft_order = []
        self.size = 0
        self.total_value = 0

    @property
    def player_names(self):
        return [x[0] for x in self.draft_order]

    @property
    def team_id(self):
        return "_".join(sorted(self.player_names))

    def draft_player(self, player_id, pos, value):
        if pos not in self.players:
            logging.error("Attempted to add player with id '{0}' with invalid position: {1}".format(player_id, pos))
            raise IOError("Attempted to add player to team with invalid position!")

        # Add to list of players
        self.players[pos].append((player_id, value))

        self.draft_order.append((player_id, value))
        self.size += 1
        self.total_value += value

    def can_add_player(self, pos):
        # Check if adding player exceeds team size
        if self.size + 1 > self.league_config["team_size"]:
            return False

        # Check if adding player exceeds team positional limit
        if len(self.players[pos]) + 1 > self.league_config["max_%s" % (pos.lower())]:
            return False

        # Check if adding player would mean other minimum position limits don't get met
        num_needed = 0
        for need_pos in self.players:
            if need_pos != pos:
                # Number needed is the difference between positional min and the number you currently have at position
                num_needed += max(0, self.league_config["min_%s" % need_pos.lower()] - len(self.players[need_pos]))

        # Return false if adding player of current position would prevent other positions getting filled
        if num_needed > self.league_config["team_size"] - (self.size+1):
            return False

        return True

    def get_startable_value(self):
        start_value = 0
        # Get value of each starting positions
        flex_pos = []
        for pos in self.players:
            data = sorted(self.players[pos], key=lambda tup: tup[1], reverse=True)
            num_start = league_config["start_%s" % pos.lower()]
            start_value += sum([x[1] for x in data[0:num_start]])

            # Add non-starting position players to pool of potential flex players
            if pos in self.league_config["flex_pos"]:
                flex_pos += [x for x in data[num_start:]]

        # Calculate value of flex players from non-starting positional players
        data = sorted(flex_pos, key=lambda tup: tup[1], reverse=True)
        start_value += sum(x[1] for x in data[0:self.league_config["start_flex"]])
        return start_value

    def __str__(self):
        player_string = ", ".join([str(x[0]) for x in self.draft_order])
        point_string = ", ".join([str(int(x[1])) for x in self.draft_order])
        return "%s\t%s\t%s\t%s" % (player_string,
                                   point_string,
                                   self.total_value,
                                   self.get_startable_value())

    def __eq__(self, team):
        return team.team_id == self.team_id


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
    argparser_obj.add_argument("--output-prefix",
                               action="store",
                               type=str,
                               dest="output_prefix",
                               required=True,
                               help="Output file prefix for simulation output file.")

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
    required_cols = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD, cols.ADP_FIELD,
                     cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT]
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

    # Check to make sure all your players have actually been drafted
    drafted_players = list(input_df[~pd.isnull(input_df[cols.DRAFT_STATUS])].index)
    my_picks = list(input_df[~pd.isnull(input_df[cols.MY_PICKS])].index)
    potential_picks = list(input_df[~pd.isnull(input_df[cols.RUN_SIM_DRAFT])].index)

    #if not potential_picks:
    #    logging.error("You didn't select any players to consider drafting! "
    #                  "Mark some players for consideration in the '%s' column!" % cols.RUN_SIM_DRAFT)
    #    errors = True

    # Check to make sure all your picks have actually been drafted
    for pick in my_picks:
        if pick not in drafted_players:
            logging.error("Player {0} listed as 'My Pick' but "
                          "hasn't actually been drafted!".format(input_df.loc[pick, cols.NAME_FIELD]))
            errors = True

    # Check to make sure all your potential picks haven't actually been drafted
    for pick in potential_picks:
        if pick in my_picks or pick in drafted_players:
            logging.error("Player {0} listed in 'Potential Picks' "
                          "but has already been drafted!".format(input_df.loc[pick, cols.NAME_FIELD]))
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


def auto_select_draft_choices(input_df, league_config):

    # Build team as it current is to see what positions are needed
    curr_players = list(input_df[~pd.isnull(input_df[cols.MY_PICKS])].index)
    curr_team = Team(league_config)
    for drafted_player in curr_players:
        name = input_df.loc[drafted_player, cols.NAME_FIELD]
        pos = input_df.loc[drafted_player, cols.POS_FIELD]
        value = input_df.loc[drafted_player, cols.VORP_FIELD]
        curr_team.draft_player(name, pos, value)

    # Sort players by VORP so we can just grab the best player at the position by VORP
    input_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

    # Go find next available players needed by team
    players_to_sim = []
    for pos in list(input_df[cols.POS_FIELD].unique()):
        # Skip position if current team already has requirements met
        if not curr_team.can_add_player(pos):
            logging.warning("Not considering drafting new {0} because position maximum reached on current team!".format(pos))
            continue

        # Otherwise get the next best player by VORP at the position
        next_best = list(input_df[(input_df[cols.POS_FIELD] == pos) &
                                  (pd.isnull(input_df[cols.DRAFT_STATUS]))][cols.NAME_FIELD])
        if next_best:
            players_to_sim.append(next_best[0])
    return players_to_sim


def get_initial_input_df(player_to_draft, input_df):
    # Make copy of input df
    new_input_df = input_df.copy(deep=True)

    # Set player to drafted
    new_input_df.loc[new_input_df[cols.NAME_FIELD] == player_to_draft, cols.DRAFT_STATUS] = "T"

    # Set player to drafted by you
    new_input_df.loc[new_input_df[cols.NAME_FIELD] == player_to_draft, cols.MY_PICKS] = "T"

    return new_input_df


def get_best_draftable_players(draftable_df, num_to_draft=3):
    # Return best draftable players from current avaiable in draft
    players = {"QB": [], "RB": [], "WR": [], "TE": []}
    draftable_df = draftable_df.sort_values(by=cols.VORP_RANK_FIELD)

    for pos in players:
        best_avail_name = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.NAME_FIELD])
        best_avail_pts = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.VORP_FIELD])
        if best_avail_name:
            num_avail = min(len(best_avail_name), num_to_draft)
            players[pos] = list(zip(best_avail_name[0:num_avail], best_avail_pts[0:num_avail]))

    return players


def log_draftable_players(draftable_players_dict, draft_round):
    for pos in draftable_players_dict:
        player_string_list = []
        max_name_len = max([len(x[0]) for x in draftable_players_dict[pos]])
        for player in draftable_players_dict[pos]:
            player_string_list.append("{0:<{width}} {1:0.2f}".format(player[0]+":", player[1], width=max_name_len+2))
        logging.debug("Rd. {0} best available {1}\n"
                      "{2}\n".format(draft_round, pos, "\n".join(player_string_list)))


def get_possible_teams(my_team, input_df, my_draft_slot, league_config, window_size=5, num_to_draft=3, size_threshold=50000, num_to_consider=8):
    teams = [my_team]
    team_ids = {my_team.team_id: True}
    team_size = my_team.size

    # Get current draft position
    curr_draft_pos = list(input_df[input_df[cols.DRAFT_SLOT_FIELD] != -1][cols.DRAFT_RANK_FIELD])[0]

    # Get draft positions available at current draft slot
    possible_positions = list(input_df[(input_df[cols.DRAFT_RANK_FIELD] >= curr_draft_pos) &
                                       (input_df[cols.DRAFT_SLOT_FIELD] == my_draft_slot)][cols.DRAFT_RANK_FIELD])

    # Add final draft window for last pick
    possible_positions.append(input_df[cols.DRAFT_RANK_FIELD].max())
    logging.debug("Autodraft slots at current slot: {0}".format(possible_positions))

    while team_size < league_config["team_size"] and curr_draft_pos <= input_df[cols.DRAFT_RANK_FIELD].max():
        logging.debug("Simulated draft round: {0}".format(team_size+1))

        # Get draft positions available at current draft slot
        next_draft_slots = [pos for pos in possible_positions if pos >= curr_draft_pos][0:window_size]
        logging.debug("Considering players within pick window: {0}-{1}".format(next_draft_slots[0],
                                                                               next_draft_slots[window_size-1]))

        # Subset players to include only those which are expected to be drafted inside current window
        draftable_df = input_df[(input_df[cols.DRAFT_RANK_FIELD] >= next_draft_slots[0]) &
                                (input_df[cols.DRAFT_RANK_FIELD] <= next_draft_slots[window_size-1])]

        # Select best N players at each position to consider drafting
        draftable_players = get_best_draftable_players(draftable_df, num_to_consider)
        log_draftable_players(draftable_players, team_size+1)

        # New teams that will be created by adding an additional player to existing team
        new_teams = []

        # Loop through existing teams and create one new team for each potential player drafted in this round
        for team in teams:

            # Make root copy of team before any players added
            original_team = deepcopy(team)

            # Boolean for whether original team in teams list has had a player added
            # Once a player has been added to the root team object, additional teams created as copy of original_team
            original_has_drafted = False

            # Create new teams by adding a potentially draftable player to a copy of the old team
            # Creates up to num_positions * num_added new teams for each existing team in teams list
            for pos in draftable_players:
                # Check to see if the root team can even add a player at the current position
                # Team object will apply rules specified in league_config to determine whether pos is addable
                if original_team.can_add_player(pos):
                    num_added = 0
                    for player in draftable_players[pos]:
                        # Loop through draftable players until num_added new teams have been created
                        if player[0] in original_team.player_names:
                            # Don't add duplicate player to team
                            continue
                        if not original_has_drafted:
                            # First player added to team this round.
                            # Create new team from actual team object
                            team.draft_player(player[0], pos, player[1])
                            original_has_drafted = True
                            num_added += 1
                        else:
                            # Create new team by adding player to copy of root team
                            new_team = deepcopy(original_team)
                            new_team.draft_player(player[0], pos, player[1])

                            # Only add team if it's truly a unique combination of players
                            # Check if sorted list of player names is identical to existing team
                            if new_team.team_id not in team_ids:
                                new_teams.append(new_team)
                                team_ids[new_team.team_id] = True
                                num_added += 1

                        # No need to add any more players if we've already created enough new teams at current position
                        if num_added >= num_to_draft:
                            break

        # Add new teams to branch of teams
        teams += new_teams
        logging.debug("Total teams: {0}".format(len(teams)))

        # Randomly subset teams if number of teams is > threshold
        # Prevents exponential growth of teams
        # Makes this program an approximate solution
        if len(teams) > size_threshold:
            logging.info("Team threshold reached. Randomly selecting subset of {0} (including top-50 best current teams)!".format(size_threshold))
            top_teams = sorted(teams, key=lambda x: x.get_startable_value(), reverse=True)[0:50]
            teams = random.sample(teams, size_threshold-50) + top_teams
            logging.info("New number of teams: {0}".format(len(teams)))

        # Increment draft position for next round
        curr_draft_pos = next_draft_slots[1]

        # Update current size of teams created
        team_size += 1

    # Return final list of teams
    return teams


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
    out_prefix  = args.output_prefix
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

    # Get number of players that have been drafted so far
    num_drafted = input_df[~pd.isnull(input_df[cols.DRAFT_STATUS])].shape[0]

    # Calculate current round and pick position based on num drafted
    curr_round = ((num_drafted - 1) // league_size) + 1
    curr_pick = num_drafted + 1
    my_draft_slot = utils.get_draft_slot(curr_pick, league_size)

    # Log basic info
    logging.warning("Draft status inferred from cheatsheet:\n"
                    "Current draft pick: {0}\n"
                    "Current draft round: {1}\n"
                    "Current slot on the board: {2}".format(curr_pick,
                                                            curr_round,
                                                            my_draft_slot))

    # Check to make sure you have the right number of players for the current pick in the draft
    theoretical_draft_slots = utils.generate_autodraft_slots(num_players=len(input_df),
                                                             league_size=league_size,
                                                             curr_pick=1)

    required_team_size = len([x for x in theoretical_draft_slots[0:curr_pick-1] if x == my_draft_slot])
    curr_team_size = len(list(input_df[~pd.isnull(input_df[cols.MY_PICKS])].index))
    if required_team_size != curr_team_size:
        logging.error("We are at pick {0} and you have {1} players. "
                      "For your current draft slot you should have {2} players!".format(curr_pick,
                                                                                        curr_team_size,
                                                                                        required_team_size))
        raise IOError("You have the wrong number of players on your team for the current spot in the draft!")

    # Get list of players you're considering drafting with next pick
    players_to_sim = list(input_df[~pd.isnull(input_df[cols.RUN_SIM_DRAFT])][cols.NAME_FIELD])
    if not players_to_sim:
        players_to_sim = auto_select_draft_choices(input_df, league_config)
        if not players_to_sim:
            logging.error("No draftable players remain!")

    # Update num drafted and current pick to reflect that you'll pick one of the potential guys
    num_drafted += 1
    curr_pick += 1

    # Run simulation to see average team value if you draft certain players with next pick
    logging.info("Simulating drafts for players: {0}".format(", ".join(players_to_sim)))
    for player_to_draft in players_to_sim:

        # Get new copy of input_df where player is drafted by you with current pick
        draft_input_df = get_initial_input_df(player_to_draft, input_df)

        # Get list of players currently on your team after drafting the potential pick
        curr_team = list(draft_input_df[~pd.isnull(draft_input_df[cols.MY_PICKS])].index)

        logging.info("Simulating drafting player: {0}\n"
                     "Current team: {1}".format(player_to_draft,
                                                ", ".join(draft_input_df.loc[curr_team, cols.NAME_FIELD])))

        # Reset draft order from ADP ranks based on remaining players
        draft_input_df.loc[~pd.isnull(draft_input_df[cols.DRAFT_STATUS]), cols.ADP_FIELD] = 1

        # Add column for draft rank
        draft_input_df.sort_values(by=cols.ADP_FIELD, inplace=True)
        draft_input_df[cols.DRAFT_RANK_FIELD] = draft_input_df.reset_index().index + 1

        # Generate expected autodraft slots for remaining undrafted players
        draft_slots = [-1]*num_drafted + utils.generate_autodraft_slots(num_players=len(draft_input_df)-num_drafted,
                                                                        league_size=league_size,
                                                                        curr_pick=curr_pick)

        draft_input_df[cols.DRAFT_SLOT_FIELD] = pd.Series(draft_slots).values
        logging.debug("Current draft board:\n{0}".format(draft_input_df.head(num_drafted+10)))

        # Initialize new Team object for holding current team
        new_team = Team(league_config)
        for drafted_player in curr_team:
            name = draft_input_df.loc[drafted_player, cols.NAME_FIELD]
            pos = draft_input_df.loc[drafted_player, cols.POS_FIELD]
            value = draft_input_df.loc[drafted_player, cols.VORP_FIELD]
            new_team.draft_player(name, pos, value)

        # Generate possible teams
        draft_results = get_possible_teams(new_team, draft_input_df, my_draft_slot, league_config)

        logging.info("Sorting draft results...")
        draft_results = sorted(draft_results, key=lambda x: x.get_startable_value(), reverse=True)

        # Create output filename
        out_file = "{0}.round{1}.slot{2}.{3}.simdraft.txt".format(out_prefix,
                                                                  curr_round,
                                                                  my_draft_slot,
                                                                  utils.clean_string_for_file_name(player_to_draft))

        logging.info("Writing draft results to file: {0}".format(out_file))
        with open(out_file, "w") as fh:
            for draft_result in draft_results:
                fh.write("%s\n" % str(draft_result))




if __name__ == "__main__":
    main()




