import os
import numpy as np
import math
from copy import deepcopy
import logging
import pandas as pd
import scipy.stats as sp
from multiprocessing import Process, Value, Lock, Queue

import constants as cols
import utils
import yaml

class Player:
    def __init__(self, name, pos, points, points_sd, vorp):
        self.name = name
        self.pos = pos
        self.points = points
        self.points_sd = points_sd
        self.vorp = vorp
        self.vorp_baseline = self.points - self.vorp

    def simulate_n_seasons(self, n=1000):
        # Simulate some number of seasons for the player
        sim_pts = np.random.normal(self.points, self.points_sd, n)

        # Replace sub-replacement seasons with replacement-level player
        return np.clip(sim_pts, a_min=self.vorp_baseline, a_max=None)

    def __eq__(self, player):
        return self.name == player.name and self.pos == player.pos

    def __deepcopy__(self, memodict={}):
        return Player(self.name, self.pos, self.points, self.points_sd, self.vorp)

    def __str__(self):
        return "{0}\t{1}\t{2}\t{3}\t{4}\t{5}".format(self.name, self.pos, self.points, self.points_sd, self.vorp, self.vorp_baseline)

class Team:
    def __init__(self, league_config):
        self.league_config = league_config
        self.players = []

        # Monte-Carlo statistics to be set running simulations
        self.vorp_sd = 0
        self.start_vorp_sd = 0
        self.simulation_results = None

    @property
    def team_id(self):
        return "_".join(sorted(self.player_names))

    @property
    def size(self):
        return len(self.players)

    @property
    def player_names(self):
        return [x.name for x in self.players]

    @property
    def total_vorp(self):
        return sum([player.vorp for player in self.players])

    @property
    def total_points(self):
        return sum([player.points for player in self.players])

    @property
    def starters(self):
        remaining_players = sorted(self.players,
                                   key=lambda player: player.points,
                                   reverse=True)
        # Get positional starters
        starters = []
        for pos in self.league_config["global"]["pos"]:
            # fnd the best players with this position
            num_to_start = self.league_config["draft"]["start"][pos]
            num_starting = 0
            for player in remaining_players:
                if player.pos == pos and player not in starters:
                    starters.append(player)
                    num_starting += 1
                    if num_starting == num_to_start:
                        break

        # Get flex starters
        num_starting = 0
        num_to_start = self.league_config["draft"]["start"]["Flex"]
        flex_pos = self.league_config["draft"]["flex_pos"]
        for player in remaining_players:
            if player.pos in flex_pos and player not in starters:
                starters.append(player)
                num_starting += 1
                if num_starting == num_to_start:
                    break
        return starters

    @property
    def startable_points(self):
        return sum([player.points for player in self.starters])

    @property
    def startable_vorp(self):
        return sum([player.vorp for player in self.starters])

    def get_players(self, positions):
        if not isinstance(positions, list):
            positions = [positions]
        return [player for player in self.players if player.pos in positions]

    def draft_player(self, player):
        if player in self.players:
            raise IOError("Attempt to add duplicate player to team: {0}".format(player.name))
        # Add to list of players
        self.players.append(player)

    def can_add_player(self, player):
        # Check if adding player exceeds team size
        if self.size + 1 > self.league_config["draft"]["team_size"]:
            return False

        # Check to see if player already on team
        if player in self.players:
            return False

        # Check if adding player exceeds team positional limit
        num_at_pos = len(self.get_players(player.pos))
        max_at_pos = self.league_config["draft"]["max"][player.pos]

        # Return false if adding player will exceed position limit
        if num_at_pos + 1 > max_at_pos:
            return False

        # Check if adding player would mean other minimum position limits don't get met
        num_needed = 0
        for need_pos in self.league_config["global"]["pos"]:
            if need_pos != player.pos:
                # Number needed is the difference between positional min and the number you currently have at position
                num_at_pos = len(self.get_players(need_pos))
                num_needed += max(0, self.league_config["draft"]["min"][need_pos] - num_at_pos)

        # Return false if adding player of current position would prevent other positions getting filled
        if num_needed > self.league_config["draft"]["team_size"] - (self.size+1):
            return False

        return True

    def simulate_n_seasons(self, n):
        # Simulate N number of seasons for each player on team

        # Just return if no players exist
        if not self.players:
            return

        sim_results_points = np.array([player.simulate_n_seasons(n) for player in self.players])

        # Get simulated VORPs for each player
        sim_results_vorp = np.array([sim_results_points[i,:] - self.players[i].vorp_baseline for i in range(len(self.players))])

        # Calculate starter values of each simulated season
        def get_start_value(season_index):
            # Sort in descending order of expected points scored that season
            starters = []
            remaining_players = np.argsort(sim_results_points[:, season_index])[::-1].tolist()
            for pos in self.league_config["global"]["pos"]:
                num_to_start = self.league_config["draft"]["start"][pos]
                num_starting = 0
                for player_index in remaining_players:
                    if self.players[player_index].pos == pos and player_index not in starters:
                        num_starting += 1
                        starters.append(player_index)
                        if num_starting == num_to_start:
                            break

            # Add points for flex players
            num_starting = 0
            num_to_start = self.league_config["draft"]["start"]["Flex"]
            flex_pos = self.league_config["draft"]["flex_pos"]
            for player_index in remaining_players:
                if self.players[player_index].pos in flex_pos and player_index not in starters:
                    starters.append(player_index)
                    num_starting += 1
                    if num_starting == num_to_start:
                        break

            # Return total points the starters scored that season
            return sim_results_points[starters, season_index].sum()

        # Get total number of points from starters for each simulated season
        sim_points_starters = np.array([get_start_value(i) for i in range(n)])

        # Get total number of points from whole team for each simulated season
        sim_vorp_team       = sim_results_vorp.sum(axis=0)

        # Determine indices to use for 5th and 95th percentiles
        percentile_indices  = [int(math.ceil(n * 0.05))-1, int(math.ceil(n * 0.95))-1]

        # Calculate total team value of each simulated season
        self.simulation_results = {"sim_team_vorp_avg": sim_vorp_team.mean(),
                                   "sim_team_vorp_sd": sim_vorp_team.std(),
                                   "sim_team_vorp_5pct": np.sort(sim_vorp_team)[percentile_indices[0]],
                                   "sim_team_vorp_95pct": np.sort(sim_vorp_team)[percentile_indices[1]],
                                   "sim_starters_pts_avg": sim_points_starters.mean(),
                                   "sim_starters_pts_sd": sim_points_starters.std(),
                                   "sim_starters_pts_5pct": np.sort(sim_points_starters)[percentile_indices[0]],
                                   "sim_starters_pts_95pct": np.sort(sim_points_starters)[percentile_indices[1]]}

    def get_summary_dict(self):
        team_dict = {}
        team_dict["players"] = ", ".join([player.name for player in self.players])
        team_dict["starters"] = ", ".join([player.name for player in self.starters])
        team_dict["player_points"] = ", ".join([str(int(player.points)) for player in self.players])
        team_dict["player_vorp"] = ", ".join(str(int(player.vorp)) for player in self.players)
        team_dict["total_team_vorp"] = self.total_vorp
        team_dict["total_starter_points"] = self.startable_points
        if self.simulation_results is not None:
            for stat in self.simulation_results:
                team_dict[stat] = self.simulation_results[stat]
        return team_dict

    def __eq__(self, team):
        return team.team_id == self.team_id

    def __deepcopy__(self, memodict={}):
        copy_object = Team(self.league_config)
        copy_object.players = [deepcopy(player) for player in self.players]
        return copy_object


class DraftBoard:
    required_cols = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD, cols.POINTS_SD_FIELD,
                     cols.VORP_FIELD, cols.ADP_FIELD, cols.ADP_SD_FIELD,
                     cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT]

    def __init__(self, draft_df, league_config):

        self.league_config = league_config
        self.draft_df = draft_df

        # Check required cols
        utils.check_df_columns(self.draft_df,
                               self.required_cols,
                               err_msg="Draft board missing one or more required columns!")

        #  Check all and only positions in draft config are included on draftboard
        self._check_pos_column()

        # Deduplicate players if necessary
        self.draft_df["DupID"] = self.draft_df[cols.NAME_FIELD] + self.draft_df[cols.POS_FIELD]
        utils.drop_duplicates_and_warn(self.draft_df,
                                       id_col="DupID",
                                       warn_msg="Draft board contains duplicate players!")

        # Remove players missing required data
        self._remove_players_missing_data()

        # Validate draft status
        self._validate_draft_status()

        # Sort by average value
        self.draft_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

        # Add column for value-based draft rank
        self.draft_df[cols.VORP_RANK_FIELD] = self.draft_df.reset_index().index + 1

        # Generate auto-draft slots
        self._generate_autodraft_slots()

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
    def undrafted_players(self):
        return self.draft_df[pd.isnull(self.draft_df[cols.DRAFT_STATUS])]

    @property
    def potential_picks(self):
        return self.draft_df[~pd.isnull(self.draft_df[cols.RUN_SIM_DRAFT])]

    @property
    def num_drafted(self):
        return len(self.drafted_players)

    @property
    def curr_round(self):
        return ((self.num_drafted - 1) // self.league_config["draft"]["league_size"]) + 1

    @property
    def next_draft_slot_up(self):
        return utils.get_draft_slot(self.num_drafted+1, self.league_config["draft"]["league_size"])

    def _check_pos_column(self):
        # Check required columns present
        errors = False

        # Convert position column to uppercase
        self.draft_df[cols.POS_FIELD] = self.draft_df[cols.POS_FIELD].str.upper()

        # Check to make sure POS column contains all RB, WR, TE, QB
        pos_available = [x.upper() for x in set(list(self.draft_df[cols.POS_FIELD]))]

        for pos in self.league_config["global"]["pos"]:
            if pos not in pos_available:
                logging.error("Missing players of position type: {0}".format(pos))
                errors = True

        # Check to make sure Pos column contains only RB, WR, TE, QB
        for pos in pos_available:
            if pos not in self.league_config["global"]["pos"]:
                logging.error("One or more players contains invalid position: {0}".format(pos))
                errors = True

        if errors:
            raise IOError("Improperly formatted draft board! See above errors")

    def _remove_players_missing_data(self):
        # Remove players with missing data in any required columns (usually few/no projections for weird players)
        to_remove = []
        col_to_check = [col for col in self.required_cols if col not in [cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT]]
        for col in col_to_check:
            to_remove += self.draft_df[pd.isnull(self.draft_df[col])][cols.NAME_FIELD].tolist()

        # Log warning and remove players missing data
        logging.warning("Following players have missing "
                        "data and will"
                        " be removed:\n{0}".format(self.draft_df[self.draft_df[cols.NAME_FIELD].isin(to_remove)]))
        self.draft_df = self.draft_df[~self.draft_df[cols.NAME_FIELD].isin(to_remove)].copy()

    def _validate_draft_status(self):

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
                                                                 league_size=self.league_config["draft"]["league_size"],
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

    def _generate_autodraft_slots(self):
        # Generate autodraft slots based on current draft status and ADP

        # Reset rank of drafted players to 1 so all undrafted players will have higher draft rank
        self.draft_df.loc[~pd.isnull(self.draft_df[cols.DRAFT_STATUS]), cols.ADP_FIELD] = 1

        # Sort based on expected draft order
        self.draft_df.sort_values(by=cols.ADP_FIELD, inplace=True)

        # Create column for draft rank
        self.draft_df[cols.DRAFT_RANK_FIELD] = self.draft_df.reset_index().index + 1

        # Generate expected autodraft slots for remaining undrafted players
        draft_slots = [-1] * self.num_drafted + utils.generate_autodraft_slots(num_players=len(self.draft_df) - self.num_drafted,
                                                                               league_size=self.league_config["draft"]["league_size"],
                                                                               curr_pick=self.num_drafted+1)
        self.draft_df[cols.DRAFT_SLOT_FIELD] = pd.Series(draft_slots).values

    def check_player_exists(self, player_name):
        # Check to see if player actually exists
        if player_name not in self.draft_df[cols.NAME_FIELD].tolist():
            logging.error("Cannot get {0}! Player does not exist on draft board!".format(player_name))
            raise IOError("Cannot get {0}! Player does not exist on draft board!".format(player_name))

    def get_player_info_dict(self, player_name):
        # Check player exists
        self.check_player_exists(player_name)
        return self.draft_df[self.draft_df[cols.NAME_FIELD] == player_name].to_dict(orient="records")[0]

    def get_current_team(self):
        # Return a team object of my team based on current draft
        my_team = Team(self.league_config)
        player_names = self.my_players.sort_values(by=cols.VORP_RANK_FIELD)[cols.NAME_FIELD].tolist()
        for player_name in player_names:
            my_team.draft_player(self.get_player(player_name))
        return my_team

    def get_player(self, player_name):
        player = self.get_player_info_dict(player_name)
        return Player(player[cols.NAME_FIELD],
                      player[cols.POS_FIELD],
                      player[cols.POINTS_FIELD],
                      player[cols.POINTS_SD_FIELD],
                      player[cols.VORP_FIELD])

    def draft_player(self, player_name, on_my_team=True):
        # Check player exists
        self.check_player_exists(player_name)

        # Check to see if player has already been drafted
        if player_name in self.drafted_players[cols.NAME_FIELD].tolist():
            logging.error("Cannot draft {0}! Player has already been drafted!".format(player_name))
            raise IOError("Cannot draft player '{0}'! See above for details.".format(player_name))

        # Otherwise set player to drafted
        self.draft_df.loc[self.draft_df[cols.NAME_FIELD] == player_name, cols.DRAFT_STATUS] = "T"
        # Optionally specify whether the player is drafted to your own team
        if on_my_team:
            self.draft_df.loc[self.draft_df[cols.NAME_FIELD] == player_name, cols.MY_PICKS] = "T"

        # Update autodraft slots to reflect player being drafted
        self._generate_autodraft_slots()

    def get_next_draft_pick_pos(self, draft_slot, curr_pick=1):
        # Return pick numbers for a current draft slot
        max_pick = self.draft_df[cols.DRAFT_RANK_FIELD].max()

        if curr_pick > max_pick:
            raise IOError("Invalid draft position! Current position ({0}) "
                          "is larger than number of players in draft ({1})! ".format(curr_pick, max_pick))

        # Get next pick available to draft slot given current pick
        picks = list(self.draft_df[self.draft_df[cols.DRAFT_SLOT_FIELD] == draft_slot][cols.DRAFT_RANK_FIELD])
        return [x for x in sorted(picks) if x >= curr_pick][0]

    def get_player_draft_prob(self, player_name, pick_num):
        # Return probability that a player will already be drafted by a given pick
        player_info = self.get_player_info_dict(player_name)
        mean        = player_info[cols.ADP_FIELD]
        sd          = player_info[cols.ADP_SD_FIELD]
        # Get probability if pick_num is > mean
        return 1 - sp.norm.cdf(pick_num, loc=mean, scale=sd)

    def get_player_draft_ci(self, player_name, confidence=0.95):
        # Return confidence interval of where a player will be drafted
        player_info = self.get_player_info_dict(player_name)
        mean        = player_info[cols.ADP_FIELD]
        sd          = player_info[cols.ADP_SD_FIELD]
        return sp.norm.interval(confidence, loc=mean, scale=sd)

    def get_probable_players_in_draft_window(self, start_pick, end_pick, prob_thresh=0.05):
        # Return a list of players that will be available at the current pick

        draftable_players = []

        # Sort undrafted players by ADP
        draft_df = self.undrafted_players.sort_values(by=cols.ADP_FIELD)
        for player_name in draft_df[cols.NAME_FIELD]:
            earliest_pick, latest_pick = self.get_player_draft_ci(player_name, confidence=1-prob_thresh)
            if latest_pick < start_pick:
                # Don't consider players we can't reasonably expect to draft at this position
                continue
            elif earliest_pick > end_pick:
                # Exit loop when you've reached first pick that we could reasonably expect to be there the next round
                continue
            elif latest_pick > start_pick or earliest_pick < end_pick:
                # Consider players who have at least
                player_info = self.get_player_info_dict(player_name)
                player_info["Draft Prob"] = self.get_player_draft_prob(player_name, start_pick)
                player_info["CI"] = "({0:.2f}-{1:.2f})".format(earliest_pick, latest_pick)
                draftable_players.append(player_info)

        # Calculate expected returns for each player
        return pd.DataFrame(draftable_players).sort_values(by=cols.VORP_FIELD, ascending=False)

    def get_best_available_players_in_window(self, pick_start, pick_end=None, pos_group_size=8):
        # Return list of next best available players at each position given current draft status

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
        fields = [cols.NAME_FIELD, cols.POS_FIELD, cols.POINTS_FIELD, cols.VORP_FIELD, cols.POINTS_SD_FIELD]
        for pos in self.league_config["global"]["pos"]:
            pos_df = draftable_df[draftable_df[cols.POS_FIELD] == pos][fields]
            if len(pos_df) > 0:
                num_avail = min(len(pos_df), pos_group_size)
                pos_df = pos_df[0:num_avail]
                players[pos] = pos_df.values.tolist()
        return players

    def get_autopick_pos_window(self, draft_slot, start_pick=1, end_pick=None):
        # Return pick numbers for a current draft slot
        if cols.DRAFT_RANK_FIELD not in self.draft_df.columns:
            self._generate_autodraft_slots()

        # Set end pick to last draft slot if no end specified
        end_pick = end_pick if end_pick is not None else self.draft_df[cols.DRAFT_RANK_FIELD].max()

        if end_pick > start_pick:
            raise IOError("Invalid autodraft window specificied: end_pick {0} "
                          "cannot be greater than start_pick {1}".format(end_pick,
                                                                         start_pick))

        picks = list(self.draft_df[self.draft_df[cols.DRAFT_SLOT_FIELD] == draft_slot][cols.DRAFT_RANK_FIELD])
        return [x for x in picks if x >= start_pick and x <= end_pick]

    def clone(self):
        return DraftBoard(self.draft_df.copy(deep=True), self.league_config)

    def sort_by_VORP(self):
        self.draft_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

    def sort_by_ADP(self):
        self.draft_df.sort_values(by=cols.ADP_FIELD, inplace=True)


class MCMCDraftTree:
    # Compare the expected outcomes from one or more players
    def __init__(self, players_to_compare, draft_board, min_player_draft_probability=0.05, max_draft_node_size=None):

        # Draft board to use for simulation
        self.draft_board = draft_board

        # Players for comparing draft outcomes if you draft them with next pick
        self.players_to_compare = players_to_compare
        if not isinstance(players_to_compare, list):
            self.players_to_compare = [self.players_to_compare]

        # Check to make sure players actually exist
        for player in players_to_compare:
            self.draft_board.check_player_exists(player)

        # Current team from which to simulate draft
        self.curr_team = self.draft_board.get_current_team()

        # Draft slot to simulate draft from
        self.draft_slot = self.draft_board.next_draft_slot_up

        # Current draft round from which to simulate
        self.start_round = self.curr_team.size + 1

        # Max draft rounds
        self.max_draft_rounds = draft_board.league_config["draft"]["team_size"]

        # Min probablity for considering whether to draft a player within a given window
        self.min_player_draft_probability = min_player_draft_probability

        # Maximum number of players to consider in at each draft node
        self.max_draft_node_size = max_draft_node_size

        # Initialize board of
        self.draft_tree = self.init_mcmc_tree()

    def init_mcmc_tree(self):
        logging.info("Building MCMC draft tree...")
        draft_slot_board = {}
        curr_round = self.start_round
        curr_pick  = len(self.draft_board.drafted_players)
        first_node = True
        while curr_round <= self.max_draft_rounds and curr_pick < self.draft_board.total_players:

            # Get players that will likely be drafted within current window
            curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick)
            next_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick+1)
            possible_players = self.draft_board.get_probable_players_in_draft_window(curr_pick,
                                                                                     next_pick-1,
                                                                                     prob_thresh=self.min_player_draft_probability)

            # If the first node, select only the players you're looking to compare
            if first_node:
                possible_players = possible_players[possible_players[cols.NAME_FIELD].isin(self.players_to_compare)]

            # Create MCMC decisions tree node for sampling players for this round
            node = MCMCDraftNode(possible_players, use_flat_priors=first_node, max_size=self.max_draft_node_size)
            draft_slot_board[str(curr_round)] = node

            logging.debug("Round {0} player distribution ({1}-{2}) :\n{3}".format(curr_round,
                                                                                  curr_pick,
                                                                                  next_pick-1,
                                                                                  node))

            print("Building round {0} player distribution ({1}-{2})...".format(curr_round,
                                                                               curr_pick,
                                                                               next_pick-1))

            # Move onto next round and increment current pick
            curr_round += 1
            curr_pick = next_pick
            first_node = False

        return draft_slot_board

    def sample(self):
        # Sample a team from the draft tree
        team = deepcopy(self.curr_team)
        for draft_round in range(self.start_round, self.max_draft_rounds+1):
            # Keep drawing from player distribution at current round until you find a player
            # you can add to the current team
            player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample())
            while not team.can_add_player(player):
                player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample())
            # Add player to current team
            team.draft_player(player)

        # Sample from teams score probability distribution as well
        team.simulate_n_seasons(1)

        # Return team after all rounds completed
        return team


class MCMCDraftNode:
    # Node for generating random selections for a round of drafting
    def __init__(self, players_to_evaluate, use_flat_priors=False, max_size=None):
        self.players_to_evaluate = players_to_evaluate
        self.use_flat_priors = use_flat_priors
        self.max_size = max_size

        # Initialize prior probablities for drafting players in the round
        self.init_posterior_dist()

        # Get names of players and cutoffs
        self.player_names = self.players_to_evaluate[cols.NAME_FIELD].tolist()
        self.posterior = self.players_to_evaluate["PostProb"]

    def init_posterior_dist(self):
        # Generate draft posterior probabilities of selecting players based on expected returns

        # Shift VORPs to be > 0 so we don't mess with probabilities
        if self.players_to_evaluate[cols.VORP_FIELD].min() <= 0:
            shift_dist = 0 - self.players_to_evaluate[cols.VORP_FIELD].min() + 1
            self.players_to_evaluate[cols.VORP_FIELD] = self.players_to_evaluate[cols.VORP_FIELD] + shift_dist

        # Set priors either to expected return or 1 if using flat priors when comparing players equally at a position
        self.players_to_evaluate["Prior"] = self.players_to_evaluate[cols.VORP_FIELD] * self.players_to_evaluate["Draft Prob"] if not self.use_flat_priors else 1
        self.players_to_evaluate.sort_values(by="Prior", ascending=False, inplace=True)

        # Subset to include only top choices if max size set
        if self.max_size is not None:
            self.players_to_evaluate = self.players_to_evaluate.iloc[0:self.max_size].copy()

        # Generate probability each player should be drafted given expected return
        self.players_to_evaluate["PostProb"] = self.players_to_evaluate["Prior"] / self.players_to_evaluate["Prior"].sum()

    def sample(self):
        # Sample player from posterior distribution and return name
        return np.random.choice(self.player_names, p=self.posterior)

    def __str__(self):
        return str(self.players_to_evaluate[[cols.NAME_FIELD, "Draft Prob", "CI", cols.ADP_FIELD, cols.VORP_FIELD, "PostProb"]])


class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


def sample_teams(mcmc_draft_tree, num_teams, counter, output_queue):
    # Reset numpy random seed so all threads don't return the same teams
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    teams = []
    for i in range(num_teams):
        if counter.value() % 1000 == 0 and counter.value() != 0:
            logging.info("Sampled {0} teams...".format(counter.value()))
            print("Sampled {0} teams...".format(counter.value()))
        teams.append(mcmc_draft_tree.sample())
        counter.increment()

    # Add teams to shared queue
    output_queue.put(teams)


def do_mcmc_sample_parallel(mcmc_draft_tree, num_samples, num_threads):
    # Perform MCMC sampling in parallel
    counter = Counter(0)
    results_q = Queue()
    samples_per_thread = int(math.ceil(num_samples/float(num_threads)))
    procs = []
    teams = []

    # Kick off thread workers to sample teams from distribution
    for i in range(num_threads):
        procs.append(Process(target=sample_teams,
                             args=(mcmc_draft_tree,
                                   samples_per_thread,
                                   counter,
                                   results_q)))
        procs[i].start()

    # Add results as they finish
    for p in procs: teams.extend(results_q.get())

    # Wait for threads to complete and return teams
    for p in procs: p.join()
    return teams


draftsheet = "/Users/awaldrop/Desktop/ff/draft_cheatsheet_8-7-19.xlsx"
draft_df = pd.read_excel(draftsheet)

with open("/Users/awaldrop/PycharmProjects/fantasy_draft_tools/draft_configs/LOG.yaml", "r") as stream:
    league_config = yaml.safe_load(stream)

print(league_config)

db = DraftBoard(draft_df, league_config)


print(db.my_players)

t = db.get_current_team()
t.simulate_n_seasons(10000)
print(t.get_summary_dict())
print(db.next_draft_slot_up)
print(db.get_next_draft_pick_pos(3, curr_pick=4))

print(db.get_player_draft_ci("Melvin Gordon", 0.5))
print(db.get_player_draft_ci("Christian McCaffrey", 0.5))
print(db.get_player_draft_ci("Saquon Barkley", 0.7))

pd.set_option('display.max_columns', None)
#print(db.get_probable_players_in_draft_window(24, 48)[[cols.NAME_FIELD, "Draft Prob", "CI", cols.ADP_FIELD, cols.VORP_FIELD]])
#print(db.get_probable_players_in_draft_window(24, 25)[[cols.NAME_FIELD, "Draft Prob", "CI", cols.ADP_FIELD, cols.VORP_FIELD, "ExpReturn", "Prob", "ProbRange"]])

#print(db.get_probable_players_in_draft_window(48, 72)[[cols.NAME_FIELD, "Draft Prob", "CI", cols.ADP_FIELD, cols.VORP_FIELD]])

#x = MCMCDraftSimulator(db, 1)
#print(x.draft_slot_board)

my_players = db.potential_picks[cols.NAME_FIELD].tolist()

x = MCMCDraftTree(my_players, db, max_draft_node_size=75)
#x = MCMCDraftSimulator(my_players, db)
curr_round = x.start_round

teams = do_mcmc_sample_parallel(x, num_samples=50000, num_threads=8)
teams = [team.get_summary_dict() for team in teams]

#for i in range(1, 10000):
#    if i % 100 == 0:
#        print("Done {0} teams!".format(i))
#    team = x.sample()
#    team.simulate_n_seasons(1)
#    teams.append(team.get_summary_dict())

df = pd.DataFrame(teams)
#df.sort_values(by="sim_starters_pts_avg", ascending=False, inplace=True)
df.sort_values(by="sim_team_vorp_avg", ascending=False, inplace=True)

def player_on_team(row, player, draft_spot):
    team = row['players'].split(",")
    return team[draft_spot].strip() == player


cutoff_num = int(math.ceil(len(df)/len(my_players)))
player_prob_df = df.iloc[0:cutoff_num]
for player in my_players:
    df[player] = df.apply(player_on_team, axis=1, player=player, draft_spot=curr_round-1)
    prob = df[player][0:cutoff_num].sum()/cutoff_num
    print("{0}: {1}".format(player, prob))

df.to_excel("/Users/awaldrop/Desktop/test_round6.xlsx")
