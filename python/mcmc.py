import os
import numpy as np
import math
from copy import deepcopy
import logging
import pandas as pd
from multiprocessing import Process, Value, Lock, Queue
import json
import yaml

import constants as cols
import utils

class EmpiricalInjuryModel:
    default_data_file = "/Users/awaldrop/PycharmProjects/fantasy_draft_tools/data/player_injury_data.json"
    def __init__(self, league_config, injury_data_file=None):
        self.league_config = league_config

        # Read in injury data from file
        injury_data_file = self.default_data_file if injury_data_file is None else injury_data_file
        with open(injury_data_file, "r") as fh:
            self.injury_data = json.load(fh)

        for pos in self.league_config["global"]["pos"]:
            if pos not in self.injury_data:
                err_msg = "Invalid Risk Model! Missing required position: {0}".format(pos)
                logging.error(err_msg)
                raise utils.DraftException(err_msg)
            self.injury_data[pos] = np.array(self.injury_data[pos])

    def sample(self, pos):
        return np.random.choice(self.injury_data[pos])


class TeamSampler:
    # Compare the expected outcomes from one or more players
    def __init__(self, players_to_compare, draft_board, min_player_draft_probability=0.05, max_draft_node_size=None, injury_risk_model=None):

        # Draft board to use for simulation
        self.draft_board = draft_board

        # Players for comparing draft outcomes if you draft them with next pick
        self.players_to_compare = players_to_compare
        if not isinstance(players_to_compare, list):
            self.players_to_compare = [self.players_to_compare]

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

        # Validate players you want to compare to make sure you actually can add them to your team
        # and that they aren't already drafted
        self.validate_players_to_compare()

        # Injury risk model for simulating games missed
        self.injury_risk_model = injury_risk_model

        # Initialize board of
        self.draft_tree = self.init_mcmc_tree()

    def validate_players_to_compare(self):
        # Check to make sure players actually exist and they're not already drafted
        players_to_remove = []
        for player in self.players_to_compare:
            self.draft_board.check_player_exists(player)
            if self.draft_board.is_player_drafted(player):
                logging.error("Invalid MCMCTree! Player to compare has already been drafted: {0}".format(player))
                raise utils.DraftException(
                    "Invalid MCMCTree! Player to compare has already been drafted: {0}".format(player))
            # Check to see if player position can even be added to team
            player = self.draft_board.get_player(player)
            if not self.curr_team.can_add_player(player):
                logging.warning("Dropping {0} from list of players to "
                                "consider as team can't add another {1}".format(player.name,
                                                                                player.pos))
                # Add to list of players to remove
                players_to_remove.append(player.name)

        # Remove players that don't need to be compared because they can't be added
        for player in players_to_remove: self.players_to_compare.remove(player)
        if len(self.players_to_compare) == 0:
            logging.error("Invalid MCMCTree! None of the players to compare could be added to your team!")
            raise utils.DraftException(logging.error("Invalid MCMCTree! None of the players "
                                                     "to compare could be added to your team!"))

    def init_mcmc_tree(self):
        logging.info("Building MCMC draft tree...")
        draft_slot_board = {}
        curr_round = self.start_round
        curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot,
                                                             len(self.draft_board.drafted_players))
        first_node = True
        while curr_round <= self.max_draft_rounds and curr_pick < self.draft_board.total_players:

            # Create node for sampling players for current round
            players_to_compare = self.players_to_compare if first_node else None
            node = PlayerSampler(curr_pick,
                                 self.draft_board,
                                 self.min_player_draft_probability,
                                 choose_only_players=players_to_compare,
                                 curr_team=self.curr_team,
                                 draft_slot=self.draft_slot)

            draft_slot_board[str(curr_round)] = node

            logging.debug("Round {0} player distribution ({1}) :\n{2}".format(curr_round,
                                                                                  curr_pick,
                                                                                  node))

            print("Building round {0} player distribution ({1}):".format(curr_round,
                                                                             curr_pick))

            print(node)
            # Move onto next round and increment current pick
            curr_round += 1
            curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick+1)
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
        team.simulate_n_seasons(1, self.injury_risk_model)

        # Return team after all rounds completed
        return team


class PlayerSampler:
    # Node for generating random selections for a round of drafting
    def __init__(self, curr_pick, draft_board, min_player_draft_probability=0.05, choose_only_players=None, curr_team=None, draft_slot=None, max_size=None):

        # Current round with which to select
        self.curr_pick = curr_pick
        self.draft_board = draft_board
        self.min_player_draft_probability = min_player_draft_probability
        self.max_size = max_size

        # Current team and draft slot for which to sample
        self.curr_team = self.draft_board.get_current_team()
        self.draft_slot = self.draft_board.next_draft_slot_up

        # Optionally sample from team/draft slot other than current team on draft board
        if self.curr_team is not None and self.draft_slot is not None:
            self.curr_team = curr_team
            self.draft_slot = draft_slot

        # Optionally sample uniformly from set list of players
        self.choose_only_players = choose_only_players
        if choose_only_players is not None:
            if not isinstance(choose_only_players, list):
                self.choose_only_players = [choose_only_players]
            # Make sure choose only players are actually good
            self.validate_choose_only_players()

        # Choose uniformly from player distribution when selected players are available
        # For comparing draft outcomes of selected players in an unbiased manner
        self.uniform_sample = self.choose_only_players is not None

        # Get universe of likely players that would be around and that would be drafted if available
        self.player_distribution = self.init_player_distribution()

        # Get names of players and cutoffs
        self.player_names = self.player_distribution[cols.NAME_FIELD].tolist()
        self.posterior = self.player_distribution["PostProb"]

    def validate_choose_only_players(self):
        # Check to make sure players actually exist and they're not already drafted
        players_to_remove = []
        for player in self.choose_only_players:
            self.draft_board.check_player_exists(player)
            if self.draft_board.is_player_drafted(player):
                err_msg = "Invalid PlayerSampler! Player to compare has already been drafted: {0}".format(player)
                logging.error(err_msg)
                raise utils.DraftException(err_msg)
            # Check to see if player position can even be added to team
            player = self.draft_board.get_player(player)
            if not self.curr_team.can_add_player(player):
                logging.warning("Dropping {0} from list of players to "
                                "consider as team can't add another {1}".format(player.name,
                                                                                player.pos))
                # Add to list of players to remove
                players_to_remove.append(player.name)

        # Remove players that don't need to be compared because they can't be added
        for player in players_to_remove: self.choose_only_players.remove(player)

        # Check to make any players actually remain
        if len(self.choose_only_players) == 0:
            err_msg = "Invalid PlayerSampler! None of the players to compare could be added to your team!"
            logging.error(err_msg)
            raise utils.DraftException(logging.error(err_msg))

    def init_player_distribution(self):
        # Get list of players that will likely be available
        possible_players = self.get_draftable_players()
        # Remove players you wouldn't draft even if they were available
        if not self.choose_only_players:
            possible_players = self.remove_likely_undrafted_players(possible_players)
        # Compute sampling probabilities for possible players
        return self.compute_sampling_probs(possible_players)

    def get_draftable_players(self):
        # Get players that will likely be drafted within current window
        draft_prob = self.min_player_draft_probability if self.choose_only_players is None else 0
        curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, self.curr_pick)
        next_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, self.curr_pick + 1)
        possible_players = self.draft_board.get_probable_players_in_draft_window(curr_pick,
                                                                                 next_pick - 1,
                                                                                 prob_thresh=draft_prob)
        # If the first node, select only the players you're looking to compare
        if self.choose_only_players is not None:
            possible_players = possible_players[possible_players[cols.NAME_FIELD].isin(self.choose_only_players)]

        return possible_players

    def remove_likely_undrafted_players(self, possible_players, max_players_at_pos=5, draft_prob_cutoff=0.99):
        # Remove players that wouldn't likely be drafted even if available at current position
        players_to_remove = []

        # Remove players from positions you've already filled
        for player in possible_players[cols.NAME_FIELD]:
            if not self.curr_team.can_add_player(self.draft_board.get_player(player)):
                players_to_remove.append(player)
        possible_players = possible_players[~possible_players[cols.NAME_FIELD].isin(players_to_remove)].copy()

        # Remove players if there are a bunch of better options at position that won't likely be drafted
        possible_players.sort_values(by=cols.VORP_FIELD, ascending=False, inplace=True)
        for i in range(len(possible_players)):
            player = possible_players[cols.NAME_FIELD].iloc[i]
            pos = possible_players[cols.POS_FIELD].iloc[i]
            vorp = possible_players[cols.VORP_FIELD].iloc[i]

            # Get list of players in draft window that are better than current player
            better_players = possible_players[(possible_players[cols.VORP_FIELD] > vorp) &
                                              (possible_players[cols.POS_FIELD] == pos)].copy()

            # Keep positional best players
            if len(better_players) == 0:
                print("%s was already da best"% player)
                continue

            # Remove if there are >5 better players at position and 99% chance one will be available
            better_players["ProbDrafted"] = 1 - better_players["Draft Prob"]
            if (1-better_players["ProbDrafted"].prod()) > draft_prob_cutoff and len(better_players) > max_players_at_pos:
                players_to_remove.append(player)

        # Remove players that likely wouldn't improve team if we simulated drafting
        possible_players = possible_players[~possible_players[cols.NAME_FIELD].isin(players_to_remove)].copy()
        return possible_players

    def compute_sampling_probs(self, possible_players):
        # Generate draft posterior probabilities of selecting players based on expected returns

        # Shift VORPs to be > 0 so we don't mess with probabilities
        if possible_players[cols.VORP_FIELD].min() <= 0:
            shift_dist = 0 - possible_players[cols.VORP_FIELD].min() + 1
            possible_players[cols.VORP_FIELD] = possible_players[cols.VORP_FIELD] + shift_dist

        # Set priors either to expected return or 1 if using flat priors when comparing players equally at a position
        possible_players["Prior"] = possible_players[cols.VORP_FIELD] * possible_players["Draft Prob"] if not self.uniform_sample else 1
        possible_players.sort_values(by="Prior", ascending=False, inplace=True)

        # Generate probability each player should be drafted given expected return
        possible_players["PostProb"] = possible_players["Prior"] / possible_players["Prior"].sum()
        return possible_players

    def sample(self):
        # Sample until you find a player that can be added to team
        return np.random.choice(self.player_names, p=self.posterior)

    def __str__(self):
        return str(self.player_distribution[[cols.NAME_FIELD, "Draft Prob", "CI", cols.ADP_FIELD, cols.VORP_FIELD, "PostProb"]])


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


def do_mcmc_draft(mcmc_draft_tree, num_samples, num_threads):
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

class MCMCResultAnalyzer:
    # Field on which to optimize
    optimization_fields = [cols.SIM_STARTERS_PTS, cols.SIM_TEAM_VORP, cols.SIM_COMPOSITE_SCORE, "Sharpe"]

    def __init__(self, players_to_evaluate, curr_draft_round, mcmc_teams, starter_weight=0.5, team_weight=0.5):
        # Analyze results of MCMC to determine best position
        self.players_to_evaluate = players_to_evaluate
        self.starter_weight = starter_weight
        self.team_weight = team_weight
        if not self.starter_weight + self.team_weight == 1.0:
            raise utils.DraftException("MCMCAnalyzer weights must add up to 1! "
                                 "Current weigths: {0} and {1}".format(starter_weight, team_weight))
        self.curr_round = curr_draft_round
        self.results = self.init_results_df(mcmc_teams)

    def init_results_df(self, mcmc_teams):
        # Convert results to dataframe
        results = [team.get_summary_dict() for team in mcmc_teams]
        results = pd.DataFrame(results)

        def player_on_team(row, player, draft_spot):
            team = row['players'].split(",")
            return team[draft_spot].strip() == player

        # Specify which player to evaluate was drafted by each simulated team
        for player in self.players_to_evaluate:
            results[player] = results.apply(player_on_team,
                                            axis=1,
                                            player=player,
                                            draft_spot=self.curr_round - 1)

        # Compute combined utiltiy score for each team taking a weighted average of z-scores
        def z_score(df, col):
            return (df[col] - df[col].mean())/df[col].std()

        starters_z = "{0}_z".format(cols.SIM_STARTERS_PTS)
        team_z     = "{0}_z".format(cols.SIM_TEAM_VORP)
        results[starters_z] = z_score(results, cols.SIM_STARTERS_PTS)
        results[team_z] = z_score(results, cols.SIM_TEAM_VORP)
        results[cols.SIM_COMPOSITE_SCORE] = results[starters_z]*self.starter_weight + results[team_z]*self.team_weight
        results.sort_values(by=cols.SIM_COMPOSITE_SCORE, ascending=False, inplace=True)
        return results

    def check_optimze_arg(self, opt_arg):
        # Throw error if optimization arg isn't valid
        if opt_arg not in self.optimization_fields:
            err_msg = "Invalid value type '{0}'! " \
                     "Options: {1}".format(opt_arg, ", ".join(self.optimization_fields))
            logging.error(err_msg)
            raise utils.DraftException(err_msg)

    def get_comparison_probs(self, optimize_for):
        self.check_optimze_arg(optimize_for)
        cutoff_value = int(math.ceil(len(self.results)/len(self.players_to_evaluate)))
        results = self.results.sort_values(by=optimize_for, ascending=False).copy().iloc[0:cutoff_value]
        player_results = {}
        for player in self.players_to_evaluate:
            player_results[player] = len(results[results[player]])/len(results)
        return player_results

    def get_expected_returns(self, optimize_for):
        self.check_optimze_arg(optimize_for)
        return {player: self.results[self.results[player]][optimize_for].mean() for player in self.players_to_evaluate}

    def get_sharpe_ratios(self, optimize_for):
        self.check_optimze_arg(optimize_for)
        results = {}
        # Compute std deviation for all players compared to get least risky option
        std_devs = {player: self.results[self.results[player]][optimize_for].std() for player in self.players_to_evaluate}
        replacement_player = sorted(std_devs.items(), key=lambda kv: kv[1])[0][0]
        replacement_val = self.results[self.results[replacement_player]][optimize_for].mean()
        for player in self.players_to_evaluate:
            team_val = self.results[self.results[player]][optimize_for].mean()
            results[player] = (team_val-replacement_val)/std_devs[player]
        return results

    def plot_density(self, optimize_for, filename, max_teams=10000):
        self.check_optimze_arg(optimize_for)
        player_dict = {}
        for player in self.players_to_evaluate:
            player_results  = self.results[self.results[player]][optimize_for]
            # Downsample if enough teams are present
            if len(player_results) > max_teams:
                player_results = player_results.sample(max_teams)
            player_dict[player] = player_results
        player_df = pd.DataFrame(player_dict)
        ax = player_df.plot.kde()
        fig = ax.get_figure()
        fig.savefig(filename)

    def to_excel(self, filename, **kwargs):
        self.results.to_excel(filename, **kwargs)


draftsheet = "/Users/awaldrop/Desktop/ff/projections/draftboard_8-10-19.xlsx"
draft_df = pd.read_excel(draftsheet)

with open("/Users/awaldrop/PycharmProjects/fantasy_draft_tools/draft_configs/LOG.yaml", "r") as stream:
    league_config = yaml.safe_load(stream)

from draftboard import DraftBoard

pd.set_option('display.max_columns', None)
db = DraftBoard(draft_df, league_config)
my_players = db.potential_picks[cols.NAME_FIELD].tolist()
print(my_players)
injury_risk_model = EmpiricalInjuryModel(league_config)

x = TeamSampler(my_players, db, max_draft_node_size=35, injury_risk_model=injury_risk_model)
curr_round = x.start_round
teams = do_mcmc_draft(x, num_samples=100000, num_threads=8)

derp = MCMCResultAnalyzer(my_players, curr_round, teams)
derp.to_excel("/Users/awaldrop/Desktop/ff/test7.xlsx", index=False)
derp.plot_density("sim_starters_pts_avg", "/Users/awaldrop/Desktop/test.png")
derp.plot_density("sim_team_vorp_avg", "/Users/awaldrop/Desktop/test2.png")
derp.plot_density("total_value", "/Users/awaldrop/Desktop/test3.png")

print("EXPECTED RETURNS:")
print("Starters")
print(derp.get_expected_returns(optimize_for=cols.SIM_STARTERS_PTS))
print("Team VORP")
print(derp.get_expected_returns(optimize_for=cols.SIM_TEAM_VORP))
print("Composite score")
print(derp.get_expected_returns(optimize_for=cols.SIM_COMPOSITE_SCORE))
print()
print("SHARPE")
print(derp.get_sharpe_ratios(optimize_for=cols.SIM_STARTERS_PTS))

print()
print("PROBS")
print("Starters")
print(derp.get_comparison_probs(optimize_for=cols.SIM_STARTERS_PTS))
print("Composite score")
print(derp.get_comparison_probs(optimize_for=cols.SIM_COMPOSITE_SCORE))

