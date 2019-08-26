import numpy as np
import math
from copy import deepcopy
import logging
import pandas as pd
import scipy.stats as sp
from collections import OrderedDict

import constants as cols
import utils


class Player:
    def __init__(self, name, pos, points, points_sd, vorp):
        self.name = name
        self.pos = pos
        self.points = points
        self.points_sd = points_sd
        self.vorp = vorp
        self.vorp_baseline = self.points - self.vorp

    def simulate_n_seasons(self, n=1000, injury_risk_model=None):
        # Simulate some number of seasons for the player
        sim_pts = np.random.normal(self.points, self.points_sd, n)

        if injury_risk_model is not None and self.vorp > 0:
            # Simulate missed games each season by sampling from risk model
            for i in range(len(sim_pts)):
                if sim_pts[i] < self.vorp_baseline:
                    #print("{0} Doesn't miss any games cuz he sucks".format(self.name))
                    # Don't sample injury if below replacement level (they'd be replaced even if not injured)
                    continue

                # Sample the number of games the player will miss from risk injury model
                games_missed = injury_risk_model.sample(self.pos)

                # Don't do anything if player misses no games that season
                if games_missed == 0:
                    continue

                #print("{0} Missing {1} games and score was {2}".format(self.name, games_missed, sim_pts[i]))
                # Recompute season score with replacement player subbing for missed games
                sim_pts[i] = (sim_pts[i]/16)*(16-games_missed) + ((self.vorp_baseline/16)*games_missed)
                #sim_pts[i] = (sim_pts[i] / 16) * (16 - games_missed)
                #print("But not its {0}".format(sim_pts[i]))

        # Replace sub-replacement seasons with replacement-level player
        return np.clip(sim_pts, a_min=self.vorp_baseline, a_max=None)

    def get_confidence_interval(self, confidence=0.95, use_vorp=True):
        mean = self.vorp if use_vorp else self.points
        return sp.norm.interval(confidence, loc=mean, scale=self.points_sd)

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
    def bench(self):
        starters = [x.name for x in self.starters]
        return [x.name for x in self.players if x.name not in starters]

    @property
    def startable_points(self):
        return sum([player.points for player in self.starters])

    @property
    def startable_vorp(self):
        return sum([player.vorp for player in self.starters])

    @property
    def bench_vorp(self):
        return sum([player.vorp for player in self.bench])

    @property
    def filled_positions(self):
        filled = []
        for pos in self.league_config["global"]["pos"]:
            if len(self.get_players(pos)) >= self.league_config["draft"]["max"][pos]:
                filled.append(pos)
        return filled

    @property
    def is_full(self):
        return self.size == self.league_config["draft"]["team_size"]

    @property
    def starters_filled(self):
        max_starters = 0
        for pos in self.league_config["draft"]["start"]:
            max_starters += self.league_config["draft"]["start"][pos]
        return len(self.starters) == max_starters

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

    def simulate_n_seasons(self, n, injury_risk_model=None, conf_interval=0.95, null_player=None):
        # Simulate N number of seasons for each player on team

        # Just return if no players exist
        if not self.players:
            return

        sim_results_points = np.array([player.simulate_n_seasons(n, injury_risk_model) for player in self.players])

        # Replace null player with replacement-level season
        if null_player:
            for i in range(len(self.players)):
                if self.players[i].name == null_player:
                    sim_results_points[i] = np.array([self.players[i].vorp_baseline]*n)

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

            bench = [player for player in remaining_players if player not in starters]

            # Return total points the starters scored that season
            return sim_results_points[starters, season_index].sum(), sim_results_vorp[bench, season_index].sum()

        # Get total number of points from starters for each simulated season
        sim_points_starters = np.array([get_start_value(i)[0] for i in range(n)])
        sim_vorp_bench = np.array([get_start_value(i)[1] for i in range(n)])

        # Determine indices to use for 5th and 95th percentiles
        percentile_indices  = [int(math.ceil(n * (1-conf_interval)))-1, int(math.ceil(n * conf_interval))-1]

        # Calculate total team value of each simulated season
        self.simulation_results = {"sim_bench_vorp_avg": sim_vorp_bench.mean(),
                                   "sim_bench_vorp_sd": sim_vorp_bench.std(),
                                   "sim_bench_vorp_low_CI": np.sort(sim_vorp_bench)[percentile_indices[0]],
                                   "sim_bench_vorp_hi_CI": np.sort(sim_vorp_bench)[percentile_indices[1]],
                                   "sim_starters_pts_avg": sim_points_starters.mean(),
                                   "sim_starters_pts_sd": sim_points_starters.std(),
                                   "sim_starters_pts_low_CI": np.sort(sim_points_starters)[percentile_indices[0]],
                                   "sim_starters_pts_hi_CI": np.sort(sim_points_starters)[percentile_indices[1]]}

    def get_summary_dict(self):
        team_dict = OrderedDict()
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
                     cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT,
                     cols.EXCLUDE_PLAYER]

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
    def exclude_players(self):
        return self.draft_df[~pd.isnull(self.draft_df[cols.EXCLUDE_PLAYER])]

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
        col_to_check = [col for col in self.required_cols if col not in [cols.DRAFT_STATUS, cols.MY_PICKS, cols.RUN_SIM_DRAFT, cols.EXCLUDE_PLAYER]]
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

    def is_player_drafted(self, player_name):
        return player_name in self.drafted_players[cols.NAME_FIELD].tolist()

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

    def get_best_available(self, pos, num_to_consider=1):
        if pos not in self.league_config["global"]["pos"]:
            err_msg = "Can't return players from invalid position '{0}'".format(pos)
            logging.error(err_msg)
            raise utils.DraftException(err_msg)
        draft_df = self.undrafted_players.sort_values(by=cols.VORP_FIELD, ascending=False)
        draft_df = draft_df[draft_df[cols.POS_FIELD] == pos]
        # Exclude players if they've been excluded
        if len(self.exclude_players) > 0:
            draft_df = draft_df[~draft_df[cols.NAME_FIELD].isin(self.exclude_players[cols.NAME_FIELD])]
        return draft_df.iloc[0:num_to_consider][cols.NAME_FIELD].tolist()

    def get_auto_draft_selections(self, num_to_consider=1, team=None, ceiling_confidence=0.1):
        # Get top-N players with highest ceiling using some statistical confidence
        team = self.get_current_team() if team is None else team
        best_players = []
        for pos in self.league_config["global"]["pos"]:
            players = self.get_best_available(pos, num_to_consider*3)
            if team.can_add_player(self.get_player(players[0])):
                best_players += [self.get_player(player) for player in players]

        best_players = sorted(best_players, key=lambda x: x.get_confidence_interval(ceiling_confidence)[1], reverse=True)[0:num_to_consider]
        return [player.name for player in best_players]

    def get_adp_prob_info(self, players, pick_num):
        # Return draft probability info for list of players
        draft_df = self.draft_df[self.draft_df[cols.NAME_FIELD].isin(players)].copy()
        draft_df["Draft Prob"] = pd.Series([self.get_player_draft_prob(player, pick_num) for player in players])
        return draft_df

    def clone(self):
        return DraftBoard(self.draft_df.copy(deep=True), self.league_config)

    def sort_by_VORP(self):
        self.draft_df.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)

    def sort_by_ADP(self):
        self.draft_df.sort_values(by=cols.ADP_FIELD, inplace=True)
