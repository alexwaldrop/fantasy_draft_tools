import os
import argparse
import logging
import pandas as pd
from copy import deepcopy
import random
import multiprocessing
import time
import math

import utils
import constants as cols

POSITIONS = ["RB", "WR", "TE", "QB"]

# TODO: Read hard-coded config from external file
league_config = {"start_rb": 2, "start_wr": 2, "start_qb": 1, "start_te": 1, "start_flex": 1,
                 "flex_pos": ["RB", "WR"], "min_rb": 4, "min_wr": 4, "min_qb": 1, "min_te": 1,
                 "max_rb": 7, "max_wr": 7, "max_qb": 1, "max_te": 1,
                 "team_size": 14, "league_size": 12,
                 "vorp_cutoffs": {"QB": 8, "RB": 36, "WR": 38, "TE": 8},
                 "vbd_adp_cutoff": 100}

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
        self.total_risk  = 0

    @property
    def player_names(self):
        return [x[0] for x in self.draft_order]

    @property
    def starter_names(self):
        flex_pos = []
        starters = []
        for pos in self.players:
            data = sorted(self.players[pos], key=lambda tup: tup[1], reverse=True)
            num_start = league_config["start_%s" % pos.lower()]
            starters += [x[0] for x in data[0:num_start]]

            # Add non-starting position players to pool of potential flex players
            if pos in self.league_config["flex_pos"]:
                flex_pos += [x for x in data[num_start:]]

        # Calculate value of flex players from non-starting positional players
        data = sorted(flex_pos, key=lambda tup: tup[1], reverse=True)
        starters += [x[0] for x in data[0:self.league_config["start_flex"]]]
        return starters

    @property
    def team_id(self):
        return "_".join(sorted(self.player_names))

    @property
    def startable_risk(self):
        starters = self.starter_names
        return sum([x[2] for x in self.draft_order if x[0] in starters])

    @property
    def startable_value(self):
        starters = self.starter_names
        return sum([x[1] for x in self.draft_order if x[0] in starters])

    @property
    def risk_adjusted_value(self):
        return sum([x[1] / float(max(x[2], 1)) for x in self.draft_order])

    @property
    def risk_adjusted_startable_value(self):
        starters = self.starter_names
        return sum([x[1]/float(max(1,x[2])) for x in self.draft_order if x[0] in starters])

    def draft_player(self, player_id, pos, value, risk):
        if pos not in self.players:
            logging.error("Attempted to add player with id '{0}' with invalid position: {1}".format(player_id, pos))
            raise IOError("Attempted to add player to team with invalid position!")

        # Add to list of players
        self.players[pos].append((player_id, value, risk))
        self.draft_order.append((player_id, value, risk))
        self.size += 1
        self.total_value += value
        self.total_risk += risk

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
        # Calculate average value of team under scenarios where each pick is a "bust"
        start_value = 0
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
        start_value += sum([x[1] for x in data[0:self.league_config["start_flex"]]])
        return start_value

    def get_startable_risk(self):
        # Calculate average value of team under scenarios where each pick is a "bust"
        start_risk = 0
        flex_pos = []
        for pos in self.players:
            data = sorted(self.players[pos], key=lambda tup: tup[1], reverse=True)
            num_start = league_config["start_%s" % pos.lower()]
            start_risk += sum([x[2] for x in data[0:num_start]])

            # Add non-starting position players to pool of potential flex players
            if pos in self.league_config["flex_pos"]:
                flex_pos += [x for x in data[num_start:]]

        # Calculate value of flex players from non-starting positional players
        data = sorted(flex_pos, key=lambda tup: tup[1], reverse=True)
        start_risk += sum([x[2] for x in data[0:self.league_config["start_flex"]]])
        return start_risk

    def get_risk_adj_value(self):
        return self.total_value/float(self.total_risk)

    def get_summary_string(self):
        player_string = ", ".join([str(x[0]) for x in self.draft_order])
        point_string = ", ".join([str(int(x[1])) for x in self.draft_order])
        risk_string = ", ".join([str(int(x[2])) for x in self.draft_order])
        return "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % (player_string,
                                                       point_string,
                                                       risk_string,
                                                       self.total_value,
                                                       self.total_risk,
                                                       self.startable_value,
                                                       self.startable_risk,
                                                       self.risk_adjusted_value,
                                                       self.risk_adjusted_startable_value)

    def get_summary_string_header(self, delim="\t"):
        return delim.join(["Team", "Player Points", "Player Risk",
                           "Total Points","Total Risk",
                           "Startable Points","Startable Risk",
                           "Risk Adj. Value","Risk Adj. Startable Value"])

    def __eq__(self, team):
        return team.team_id == self.team_id

    def __deepcopy__(self, memodict={}):
        copy_object = Team(self.league_config)
        copy_object.players = {key: [x for x in value] for key, value in self.players.items()}
        copy_object.draft_order = [x for x in self.draft_order]
        copy_object.size = self.size
        copy_object.total_value = self.total_value
        copy_object.total_risk = self.total_risk
        return copy_object


class DraftBoard:
    required_cols = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD, cols.ADP_FIELD,
                     cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT, cols.RISK_FIELD]

    def __init__(self, draft_df, league_config):

        self.league_config = league_config
        self.draft_df = draft_df

        # Check required cols
        self.check_required_cols()

        # Validate draft status
        self.validate_draft_status()

    @property
    def total_players(self):
        return len(self.draft_df)

    @property
    def my_players(self):
        return self.draft_df[~pd.isnull(self.draft_df[cols.MY_PICKS])]

    @property
    def drafted_players(self):
        return self.draft_df[~pd.isnull(self.draft_df[cols.DRAFT_STATUS])]

    @property
    def potential_picks(self):
        return self.draft_df[~pd.isnull(self.draft_df[cols.RUN_SIM_DRAFT])]

    @property
    def num_drafted(self):
        return len(self.drafted_players)

    @property
    def curr_round(self):
        return ((self.num_drafted - 1) // self.league_config["league_size"]) + 1

    @property
    def next_draft_slot_up(self):
        return utils.get_draft_slot(self.num_drafted+1, self.league_config["league_size"])

    def draft_player(self, player_name, on_my_team=True):
        # Check to see if player has already been drafted
        if player_name in self.drafted_players[cols.NAME_FIELD].tolist():
            logging.error("Cannot draft {0}! Player has already been drafted!".format(player_name))
            raise IOError("Cannot draft player '{0}'! See above for details.".format(player_name))

        # Check to see if player actually exists
        if player_name not in self.draft_df[cols.NAME_FIELD].tolist():
            logging.error("Cannot draft player with name '{}' as they don't exist on draft board!")
            raise IOError("Cannot draft player '{0}'! See above for details.".format(player_name))

        # Otherwise set player to drafted
        self.draft_df.loc[self.draft_df[cols.NAME_FIELD] == player_name, cols.DRAFT_STATUS] = "T"
        # Optionally specify whether the player is drafted to your own team
        if on_my_team:
            self.draft_df.loc[self.draft_df[cols.NAME_FIELD] == player_name, cols.MY_PICKS] = "T"

    def check_required_cols(self):
        # Check required columns present
        errors = False
        for required_col in self.required_cols:
            if required_col not in self.draft_df.columns:
                # Output missing colname
                logging.error("Input file missing required col: {0}".format(required_col))
                errors = True

        if not errors:
            # Convert position column to uppercase
            self.draft_df[cols.POS_FIELD] = self.draft_df[cols.POS_FIELD].str.upper()

            # Check to make sure POS column contains all RB, WR, TE, QB
            pos_available = [x.upper() for x in set(list(self.draft_df[cols.POS_FIELD]))]

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
            raise IOError("Improperly formatted draft board! See above errors")

    def validate_draft_status(self):

        drafted_players = self.drafted_players[cols.NAME_FIELD].tolist()
        my_picks = self.my_players[cols.NAME_FIELD].tolist()
        potential_picks = self.potential_picks[cols.NAME_FIELD].tolist()
        errors = False

        # Check to make sure all your picks have actually been drafted
        for pick in my_picks:
            if pick not in drafted_players:
                logging.error("Player {0} listed as 'My Pick' but "
                              "hasn't actually been drafted!".format(pick))
                errors = True

        # Check to make sure all your potential picks haven't actually been drafted
        for pick in potential_picks:
            if pick in my_picks or pick in drafted_players:
                logging.error("Player {0} listed in 'Potential Picks' "
                              "but has already been drafted!".format(pick))
                errors = True

        # Check to make sure you have the right number of players for the current pick in the draft
        theoretical_draft_slots = utils.generate_autodraft_slots(num_players=len(self.draft_df),
                                                                 league_size=self.league_config["league_size"],
                                                                 curr_pick=1)
        curr_pick = self.num_drafted + 1
        required_team_size = len([x for x in theoretical_draft_slots[0:curr_pick - 1] if x == self.next_draft_slot_up])
        curr_team_size = len(my_picks)
        if required_team_size != curr_team_size:
            errors = True
            logging.error("We are at pick {0} and you have {1} players. "
                          "For your current draft slot you should have {2} players!".format(curr_pick,
                                                                                            curr_team_size,
                                                                                            required_team_size))
        if errors:
            raise IOError("Invalid draft board! See above errors")

    def calc_VORP(self):
        # Calculation Value of Replace Player for each player in draft

        # Generate autodraft slots if they haven't already been filled
        if cols.DRAFT_RANK_FIELD not in self.draft_df.columns:
            self.generate_autodraft_slots()

        pos_repl_val = {}
        for pos in league_config["vorp_cutoffs"]:
            pos_repl_val[pos] = calc_position_replacement_value(self.draft_df,
                                                                pos,
                                                                adp_cutoff=league_config["vbd_adp_cutoff"])

        # Add column for value over positional replacement for each player
        self.draft_df[cols.REPLACEMENT_VALUE_FIELD] = self.draft_df[cols.POS_FIELD].map(pos_repl_val)

        # Add column for points above average draftable replacement
        self.draft_df[cols.VORP_FIELD] = self.draft_df[cols.POINTS_FIELD] - self.draft_df[cols.REPLACEMENT_VALUE_FIELD]

        # Sort by average value
        self.draft_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

        # Add column for value-based draft rank
        self.draft_df[cols.VORP_RANK_FIELD] = self.draft_df.reset_index().index + 1

    def generate_autodraft_slots(self):
        # Generate autodraft slots based on current draft status and ADP

        # Reset rank of drafted players to 1 so all undrafted players will have higher draft rank
        self.draft_df.loc[~pd.isnull(self.draft_df[cols.DRAFT_STATUS]), cols.ADP_FIELD] = 1

        # Sort based on expected draft order
        self.draft_df.sort_values(by=cols.ADP_FIELD, inplace=True)

        # Create column for draft rank
        self.draft_df[cols.DRAFT_RANK_FIELD] = self.draft_df.reset_index().index + 1

        # Generate expected autodraft slots for remaining undrafted players
        draft_slots = [-1] * self.num_drafted + utils.generate_autodraft_slots(num_players=len(self.draft_df) - self.num_drafted,
                                                                               league_size=self.league_config["league_size"],
                                                                               curr_pick=self.num_drafted+1)
        self.draft_df[cols.DRAFT_SLOT_FIELD] = pd.Series(draft_slots).values

    def get_current_team(self):
        # Return a team object of my team based on current draft

        # Calc VORP if it hasn't already been calculated
        if cols.VORP_FIELD not in self.draft_df.columns:
            self.calc_VORP()

        # Loop through my players and add to new Team object
        my_team = Team(league_config)
        for i in range(len(self.my_players.sort_values(by=cols.VORP_RANK_FIELD))):
            name = self.my_players.iloc[i][cols.NAME_FIELD]
            pos = self.my_players.iloc[i][cols.POS_FIELD]
            #points = self.my_players.iloc[i][cols.POINTS_FIELD]
            points = self.my_players.iloc[i][cols.VORP_FIELD]
            risk = self.my_players.iloc[i][cols.RISK_FIELD]
            my_team.draft_player(name, pos, points, risk)

        return my_team

    def get_best_available_players_in_window(self, pick_start, pick_end=None, pos_group_size=8):
        # Return list of next best available players at each position given current draft status
        if not cols.VORP_FIELD in self.draft_df.columns:
            self.calc_VORP()

        # Set end of window to last spot in draft if no end point specified
        pick_end = pick_end if pick_end is not None else len(self.draft_df)

        # Check to make sure start/end picks are valid
        if pick_start > pick_end:
            raise IOError("Start pick ({0}) cannot be greater than "
                          "End pick ({1}) when getting best available players!".format(pick_start,
                                                                                       pick_end))

        # Get players available inside of draft window
        draftable_df = self.draft_df[(self.draft_df[cols.DRAFT_RANK_FIELD] >= pick_start) &
                                     (self.draft_df[cols.DRAFT_RANK_FIELD] <= pick_end)].sort_values(by=cols.VORP_RANK_FIELD)

        # Get up to top-N best available players by VORP at each position
        players = {}
        for pos in self.league_config["vorp_cutoffs"]:
            best_avail_name = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.NAME_FIELD])
            #best_avail_pts = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.POINTS_FIELD])
            best_avail_pts = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.VORP_FIELD])
            best_avail_risk = list(draftable_df[draftable_df[cols.POS_FIELD] == pos][cols.RISK_FIELD])
            if best_avail_name:
                num_avail = min(len(best_avail_name), pos_group_size)
                players[pos] = list(zip(best_avail_name[0:num_avail],
                                        best_avail_pts[0:num_avail],
                                        best_avail_risk[0:num_avail]))
        return players

    def get_autopick_pos_window(self, draft_slot, start_pick=1, end_pick=None):
        # Return pick numbers for a current draft slot
        if cols.DRAFT_RANK_FIELD not in self.draft_df.columns:
            self.generate_autodraft_slots()

        # Set end pick to last draft slot if no end specified
        end_pick = end_pick if end_pick is not None else self.draft_df[cols.DRAFT_RANK_FIELD].max()

        if end_pick > start_pick:
            raise IOError("Invalid autodraft window specificied: end_pick {0} "
                          "cannot be greater than start_pick {1}".format(end_pick,
                                                                         start_pick))

        picks = list(self.draft_df[self.draft_df[cols.DRAFT_SLOT_FIELD] == draft_slot][cols.DRAFT_RANK_FIELD])
        return [x for x in picks if x >= start_pick and x <= end_pick]

    def get_next_draft_pick_pos(self, draft_slot, curr_pick=1):
        # Return pick numbers for a current draft slot
        if cols.DRAFT_RANK_FIELD not in self.draft_df.columns:
            self.generate_autodraft_slots()

        max_pick = self.draft_df[cols.DRAFT_RANK_FIELD].max()

        if curr_pick > max_pick:
            raise IOError("Invalid draft position! Current position ({0}) "
                          "is larger than number of players in draft ({1})! ".format(curr_pick, max_pick))

        # Get next pick available to draft slot given current pick
        picks = list(self.draft_df[self.draft_df[cols.DRAFT_SLOT_FIELD] == draft_slot][cols.DRAFT_RANK_FIELD])
        return [x for x in sorted(picks) if x >= curr_pick][0]

    def clone(self):
        return DraftBoard(self.draft_df.copy(deep=True), self.league_config)

    def sort_by_VORP(self):
        self.draft_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

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

    # Num threads
    argparser_obj.add_argument("--n",
                               action="store",
                               type=int,
                               dest="num_threads",
                               required=True,
                               help="Num threads to use for parallel processing")

    # Risk coefficient for considering risk in draft strategy (0 ignore risk, 1 factor risk complete)
    argparser_obj.add_argument("--risk-coeff",
                               action="store",
                               type=float,
                               dest="risk_coeff",
                               default=1.0,
                               required=False,
                               help="Risk coefficient. Float between [0,1] 0 being ignore risk, 1 use full risk score for players")

    # Maximum number of teams that will be generated for each round of simulated drafting
    argparser_obj.add_argument("--max-teams-per-round",
                               action="store",
                               type=int,
                               dest="max_teams_per_round",
                               default=200000,
                               required=False,
                               help="Maximum number of teams to generate per round")


    # Maximum number of top teams to save from each round of simulated drafting
    argparser_obj.add_argument("--max-top-teams-per-round",
                               action="store",
                               type=int,
                               dest="max_top_teams_per_round",
                               default=200000,
                               required=False,
                               help="Maximum number of top teams to save from each round. Rest will be randomly selected.")


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


def calc_position_replacement_value(input_df, pos, adp_cutoff):
    # Get positional average for top-N players
    sorted_df = input_df.sort_values(by=cols.DRAFT_RANK_FIELD).iloc[0:adp_cutoff]
    return sorted_df[sorted_df[cols.POS_FIELD] == pos].Points.min()


def autodraft_best_available(draft_board):

    # Get window from which to choose available players
    next_pick = draft_board.get_next_draft_pick_pos(draft_slot=draft_board.next_draft_slot_up,
                                                    curr_pick=draft_board.num_drafted+1)

    # Get possible players available to draft with next pick
    possible_players = draft_board.get_best_available_players_in_window(next_pick, pos_group_size=1)

    # Build current team to see what positions can be drafted given draft settings in league config
    curr_team = draft_board.get_current_team()

    # Check to see if current team can add players from each position
    players_to_sim = []
    for pos in possible_players:
        if not curr_team.can_add_player(pos):
            logging.warning("Not considering drafting new {0} because position maximum reached on current team!".format(pos))
            continue
        players_to_sim.append(possible_players[pos][0][0])

    return players_to_sim


def log_draftable_players(draftable_players_dict, draft_round):
    for pos in draftable_players_dict:
        player_string_list = []
        max_name_len = max([len(x[0]) for x in draftable_players_dict[pos]])
        for player in draftable_players_dict[pos]:
            player_string_list.append("{0:<{width}} {1:0.2f}".format(player[0]+":", player[1], width=max_name_len+2))
        logging.debug("Rd. {0} best available {1}\n"
                      "{2}\n".format(draft_round, pos, "\n".join(player_string_list)))


def expand_teams(teams, draftable_players, max_to_draft_per_pos, out_q, max_teams_to_keep, max_top_teams_to_keep):
    new_teams = {}
    for team in teams:
        for pos in draftable_players:
            # Check to see if the root team can even add a player at the current position
            # Team object will apply rules specified in league_config to determine whether pos is addable
            if team.can_add_player(pos):
                num_pos_drafted = 0
                for player in draftable_players[pos]:
                    # Loop through draftable players until num_added new teams have been created
                    if player[0] in team.player_names:
                        # Don't add duplicate player to team
                        continue

                    # Create new team by adding player to copy of root team
                    new_team = deepcopy(team)
                    new_team.draft_player(player[0], pos, player[1], player[2])
                    if new_team.team_id not in new_teams:
                        new_teams[new_team.team_id] = new_team
                    num_pos_drafted += 1

                    # No need to add any more players if we've already created enough new teams at current position
                    if num_pos_drafted >= max_to_draft_per_pos:
                        break

    if len(new_teams) > max_teams_to_keep:
        top_teams = sorted(new_teams.values(), key=lambda x: x.risk_adjusted_startable_value, reverse=True)
        #top_teams = sorted(new_teams.values(), key=lambda x: x.get_combined_value(), reverse=True)
        new_teams = top_teams[0:max_top_teams_to_keep] + random.sample(top_teams[max_top_teams_to_keep:], max_teams_to_keep - max_top_teams_to_keep)
        new_teams = {team.team_id: team for team in new_teams}
    out_q.put(new_teams)


def get_possible_teams(draft_board, my_draft_slot, league_config, num_to_consider_per_pos=8, num_to_draft_per_pos=1, max_teams_per_round=200000, max_top_teams_per_round=200000, num_workers=8):

    # Initialize search from current team on draft board
    root_team = draft_board.get_current_team()
    team_size = root_team.size
    teams = [root_team]
    start_time = time.time()

    # Calculate maximum number of teams and top teams to save per batch per round
    max_teams_per_batch = int(math.ceil(max_teams_per_round / float(num_workers)))
    max_top_teams_per_batch = int(math.ceil(max_top_teams_per_round / float(num_workers)))

    # Get current draft position
    curr_draft_pos = draft_board.num_drafted + 1

    while team_size < league_config["team_size"] and curr_draft_pos <= draft_board.total_players:
        logging.debug("Simulated draft round: {0}".format(team_size+1))

        # Determine draft position in current round
        curr_draft_pos = draft_board.get_next_draft_pick_pos(my_draft_slot, curr_pick=curr_draft_pos)

        # Get draft positions available at current draft slot
        logging.debug("Considering players >= pick: {0}".format(curr_draft_pos))

        # Select best N players at each position to consider drafting
        draftable_players = draft_board.get_best_available_players_in_window(curr_draft_pos, pos_group_size=num_to_consider_per_pos)
        log_draftable_players(draftable_players, team_size+1)

        # New teams that will be created by adding an additional player to existing team
        out_q = multiprocessing.Queue()
        batch_size = int(math.ceil(len(teams)/float(num_workers)))
        procs = []
        batch_count = 0
        batch = []
        for i in range(len(teams)):
            batch.append(teams[i])
            batch_count += 1
            if batch_count >= batch_size:
                # Expand next batch of teams to create new teams
                p = multiprocessing.Process(target=expand_teams,
                                            args=(batch,
                                                  draftable_players,
                                                  num_to_draft_per_pos,
                                                  out_q,
                                                  max_teams_per_batch,
                                                  max_top_teams_per_batch))
                procs.append(p)
                p.start()
                batch_count = 0
                batch = []

        # Expand remaining teams as last batch
        if batch:
            p = multiprocessing.Process(target=expand_teams,
                                        args=(batch,
                                              draftable_players,
                                              num_to_draft_per_pos,
                                              out_q,
                                              max_teams_per_batch,
                                              max_top_teams_per_batch))
            procs.append(p)
            p.start()

        # Read results into dict to remove duplicates
        new_teams = {}
        for i in range(len(procs)):
            # Wait for next batch of team expansions to finish and add to full set of new teams
            new_teams.update(out_q.get())

        # Wait for all threads to finish
        for p in procs:
            p.join()

        # Convert back to list
        teams = list(new_teams.values())

        # Replace old set of teams with new set of expanded teams
        logging.debug("Total teams: {0}".format(len(teams)))

        # Increment draft position for next round
        curr_draft_pos += 1

        # Update current size of teams created
        team_size += 1

    logging.info("ELAPSED TIME: {0} seconds".format(time.time() - start_time))
    return teams

def main():
    # Configure argparser
    argparser = argparse.ArgumentParser(prog="sim_draft_from_current_pick")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    utils.configure_logging(args.verbosity_level)

    # Get names of input/output files
    in_file     = args.input_file
    out_prefix  = args.output_prefix
    num_threads = args.num_threads
    max_teams_per_round = args.max_teams_per_round
    max_top_teams_per_round = args.max_top_teams_per_round
    risk_coeff = args.risk_coeff

    # Log basic info
    log_string = "Using input variables:\n"
    for pos in league_config["vorp_cutoffs"]:
        log_string += "{0}s drafted: {1}\n".format(pos, league_config["vorp_cutoffs"][pos])
    log_string += "League size: {0}".format(league_config["league_size"])
    log_string += "Current draft"
    logging.info(log_string)

    # Check input file for formatting
    input_df = pd.read_excel(in_file)

    # Create draft board
    draft_board = DraftBoard(input_df, league_config)

    # Get average points for each position
    logging.info("Determining positional averages for each position...")
    draft_board.calc_VORP()

    draft_board.draft_df[cols.RISK_FIELD] = draft_board.draft_df[cols.RISK_FIELD]*risk_coeff

    # Log current draft status
    logging.warning("Draft status inferred from cheatsheet:\n"
                    "Current draft pick: {0}\n"
                    "Current draft round: {1}\n"
                    "Current slot on the board: {2}".format(draft_board.num_drafted+1,
                                                            draft_board.curr_round,
                                                            draft_board.next_draft_slot_up))

    # Get list of players you're considering drafting with next pick
    players_to_sim = draft_board.potential_picks[cols.NAME_FIELD].tolist()
    if not players_to_sim:
        # Auto-select next players by VORP if no players selected
        players_to_sim = autodraft_best_available(draft_board)
        # Error out if not draftable players remain
        if not players_to_sim:
            logging.error("No draftable players remain!")

    # Run simulation to see average team value if you draft certain players with next pick
    logging.info("Simulating drafts for players: {0}".format(", ".join(players_to_sim)))

    possible_teams = []

    for player_to_draft in players_to_sim:

        # Make a clone of current draft board
        sim_draft_board = draft_board.clone()

        # Draft player on new draft board
        sim_draft_board.draft_player(player_to_draft)

        # Reset draft order after removing drafted players
        sim_draft_board.generate_autodraft_slots()

        # Get list of players currently on your team after drafting the potential pick
        curr_team = sim_draft_board.my_players[cols.NAME_FIELD].tolist()

        logging.info("Simulating drafting player: {0}\n"
                     "Current team: {1}".format(player_to_draft,
                                                ", ".join(curr_team)))
        logging.debug("Current draft board:\n{0}".format(sim_draft_board.draft_df.head(sim_draft_board.num_drafted+30)))

        # Generate possible teams
        possible_teams += get_possible_teams(sim_draft_board,
                                             my_draft_slot=draft_board.next_draft_slot_up,
                                             league_config=league_config,
                                             max_teams_per_round=max_teams_per_round,
                                             max_top_teams_per_round=max_top_teams_per_round,
                                             num_workers=num_threads)

    # Sort by start team value
    logging.info("Sorting draft results...")
    possible_teams = sorted(possible_teams, key=lambda x: x.risk_adjusted_startable_value, reverse=True)

    # Create output filename
    out_file = "{0}.round{1}.slot{2}.simdraft.txt".format(out_prefix,
                                                              draft_board.curr_round,
                                                              draft_board.next_draft_slot_up)

    logging.info("Writing draft results to file: {0}".format(out_file))
    with open(out_file, "w") as fh:
        fh.write("%s\n" % possible_teams[0].get_summary_string_header())
        for draft_result in possible_teams:
            fh.write("%s\n" % str(draft_result.get_summary_string()))


if __name__ == "__main__":
    main()




