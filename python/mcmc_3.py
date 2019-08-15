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


class DraftTree:
    # Compare the expected outcomes from one or more players
    def __init__(self, players_to_compare, draft_board, min_adp_prior=0.05,
                 max_draft_node_size=15, injury_risk_model=None, mcmc_exploration_constant=1):

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

        # Min probablity for considering whether to draft a player
        self.min_adp_prior = min_adp_prior

        # Maximum number of players to consider in at each draft node
        self.max_draft_node_size = max_draft_node_size

        # Validate players you want to compare to make sure you actually can add them to your team
        # and that they aren't already drafted
        self.validate_players_to_compare()

        # Injury risk model for simulating games missed
        self.injury_risk_model = injury_risk_model

        # Factor for determining tradeoff between exploration and exploitation
        self.mcmc_exploration_constant = mcmc_exploration_constant

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
        draft_round = self.start_round
        curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, len(self.draft_board.drafted_players))
        last_pick = curr_pick
        while draft_round <= self.max_draft_rounds and curr_pick < self.draft_board.total_players:

            # Get boundaries of next pick for player in current draft slot
            curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick)
            next_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick+1)

            if draft_round == self.start_round:
                # Sample root players without sampling from draft posterior distribution (we know they're available)
                possible_players = self.draft_board.get_adp_prob_info(self.players_to_compare, curr_pick)
                node = PlayerSampler(possible_players,
                                     mcmc_exploration_constant=self.mcmc_exploration_constant)

            elif draft_round == self.start_round + 1 and curr_pick-last_pick-1 < len(self.players_to_compare):
                # If one of players to compare will necessarily make it back, sample from special distribution
                # to simulate adversarial picks
                # Really more of an edge case for picks around the turn
                adversary_picks_until_turn = curr_pick-last_pick - 1
                possible_players = self.draft_board.get_adp_prob_info(self.players_to_compare, curr_pick)
                node = DraftSimSampler(possible_players,
                                       num_adv_picks_to_sim=adversary_picks_until_turn,
                                       mcmc_exploration_constant=self.mcmc_exploration_constant)

            else:
                # Otherwise get players whose 95% ADP confidence interval overlaps with current draft window

                # If in last 3 rounds just open it up and try to find the best players
                last_pick = 1000 if self.max_draft_rounds - draft_round <= 3 else next_pick - 1

                possible_players = self.draft_board.get_probable_players_in_draft_window(curr_pick,
                                                                                         last_pick,
                                                                                         prob_thresh=self.min_adp_prior)
                # Remove players from filled team positions
                possible_players = possible_players[~possible_players[cols.POS_FIELD].isin(self.curr_team.filled_positions)]
                #possible_players = possible_players.sort_values(by=cols.VORP_FIELD, ascending=False)

                # Remove all but top-2 TEs and QBs

                # Get top-N by VORP with at least one from each position
                #best_players = possible_players[0:self.max_draft_node_size][cols.NAME_FIELD].tolist()

                # Check to see if any positions are missing
                #pos_in_best = possible_players[possible_players[cols.NAME_FIELD].isin(best_players)][cols.POS_FIELD].unique().tolist()
                #pos_to_fill = [pos for pos in self.draft_board.league_config["global"]["pos"] if pos not in self.curr_team.filled_positions + pos_in_best]
                #for pos in pos_to_fill:
                #    best_pos_player = possible_players[possible_players[cols.POS_FIELD] == pos][cols.NAME_FIELD].iloc[0]
                #    best_players.append(best_pos_player)

                # Create node for sampling from top-25
                #best_players = possible_players[possible_players[cols.NAME_FIELD].isin(best_players)]
                best_players = self.subset_draftable_players(possible_players)
                node = DraftSimSampler(best_players, mcmc_exploration_constant=self.mcmc_exploration_constant)

            # Add node to mcmc tree
            draft_slot_board[str(draft_round)] = node

            logging.debug("Round {0} player distribution ({1}) :\n{2}".format(draft_round,
                                                                                  curr_pick,
                                                                                  node))

            print("Building round {0} player distribution ({1}):".format(draft_round,
                                                                             curr_pick))
            print(node)

            # Move onto next round and increment current pick
            draft_round += 1
            last_pick = curr_pick
            curr_pick = next_pick

        # Set draft tree to be tree just created
        return draft_slot_board

    def subset_draftable_players(self, possible_players):
        # Goal is to reduce each node down to smallest number of players where each position still has a chance to
        # be drafted

        # Sort possible players by value
        best_players = []
        possible_players = possible_players.sort_values(by=cols.VORP_FIELD, ascending=False)
        pos_prob_missing = {pos: 1 for pos in self.draft_board.league_config["global"]["pos"] if pos not in self.curr_team.filled_positions}
        for i in range(len(possible_players)):
            name = possible_players[cols.NAME_FIELD].iloc[i]
            pos  = possible_players[cols.POS_FIELD].iloc[i]
            draft_prob = possible_players["Draft Prob"].iloc[i]

            # Add player to draftable list until prob of no players from this group
            # at this position will be left falls below 5%
            if pos_prob_missing[pos] > 0.01:
                best_players.append(name)
                pos_prob_missing[pos] *= (1-draft_prob)

        # Fill up any remaining slots with WR/RB
        if len(best_players) < self.max_draft_node_size:
            num_to_get = self.max_draft_node_size - len(best_players)
            best_wr_rb = possible_players[possible_players[cols.POS_FIELD].isin(["RB", "WR"])]
            best_wr_rb = best_wr_rb[~best_wr_rb[cols.NAME_FIELD].isin(best_players)].iloc[0:num_to_get]
            best_players.append(best_wr_rb[cols.NAME_FIELD])

        # Return subsetted data frame
        return possible_players[possible_players[cols.NAME_FIELD].isin(best_players)]

    def get_draftable_players(self, curr_pick, include_only_players=None):
        # Get list of likely players that will be drafted that you'd actually want to draft

        # Get players that will be available with > this probability
        draft_prob = self.min_adp_prior if include_only_players is None else 0

        # Define draft window as the next two picks for the current draft slot
        curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick)
        next_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick + 1)

        # Get list of players that could possibly be drafted
        possible_players = self.draft_board.get_probable_players_in_draft_window(curr_pick,
                                                                                 next_pick - 1,
                                                                                 prob_thresh=draft_prob)
        # If the first node, select only the players you're looking to compare
        if include_only_players is not None:
            # Subset to include only the players included
            return possible_players[possible_players[cols.NAME_FIELD].isin(include_only_players)]

        # Remove players from filled positions
        # Remove players from positions you've already filled
        players_to_remove = []
        for player in possible_players[cols.NAME_FIELD]:
            if not self.curr_team.can_add_player(self.draft_board.get_player(player)):
                players_to_remove.append(player)
        possible_players = possible_players[~possible_players[cols.NAME_FIELD].isin(players_to_remove)].copy()

        # Otherwise remove players you're not actually interested in drafting
        return possible_players

    def sample(self):
        # Sample a team from the draft tree
        team = deepcopy(self.curr_team)
        player_map = {}
        full_positions = []
        for draft_round in range(self.start_round, self.max_draft_rounds+1):

            # Sample until you find a player you can add to the current team
            exclude_players = team.player_names if team.size > 0 else []
            try:
                player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample(exclude_players=exclude_players,
                                                                                              exclude_pos=full_positions))
            except IOError:
                print("THIS WHERE WE FAILED")
                print("ROUND: %s" % draft_round)
                print(", ".join(team.player_names))
                print(self.draft_tree[str(draft_round)])
                raise

            while not team.can_add_player(player):
                full_positions.append(player.pos)
                player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample(exclude_players=exclude_players,
                                                                                              exclude_pos=full_positions))
            # Add player to current team
            player_map[str(draft_round)] = player.name
            team.draft_player(player)

        # Sample from teams score probability distribution as well
        team.simulate_n_seasons(1, self.injury_risk_model)

        # Compute value as average of starter and bench points
        sim_results = team.get_summary_dict()
        team_value = (sim_results[cols.SIM_STARTERS_PTS] + sim_results[cols.SIM_TEAM_VORP])/2

        # Update weights on player nodes
        for round in player_map: self.draft_tree[round].record_visit(player_map[round], team_value)

        # Return team after all rounds completed
        return team

    def merge_branch(self, branch):
        # Merge data from another draft tree
        for node in self.draft_tree:
            # If node doesn't exist merge with other node
            if node not in branch.draft_tree:
                raise utils.DraftException("Cannot merge draft trees! Node not found in both tree: {0}".format(node))
            self.draft_tree[node].merge_branch(branch.draft_tree[node])

    def get_node(self, round):
        if round == "root":
            return self.draft_tree[str(self.start_round)]
        return self.draft_tree[str(round)]

    def set_exploration_constant(self, exploration_constant):
        self.mcmc_exploration_constant = exploration_constant
        for node in self.draft_tree:
            self.draft_tree[node].set_exploration_constant(exploration_constant)


class PlayerSampler:
    # Special case of draft sampler that picks players uniformly from distribution
    # Assumes all players will be undrafted at time of pick
    def __init__(self, players, mcmc_exploration_constant=1):
        # Compute the absolute probability of sampling each player in distribution
        self.players = self.init_node_priors(players)
        self.players["Visits"] = 0
        self.players["Q"] = 0
        self.players["Weights"] = 0
        self.players = self.players.set_index(cols.NAME_FIELD, drop=False)
        self.total_visits = 0
        self.weight_order = None
        self.exploration_constant = mcmc_exploration_constant

    def init_node_priors(self, players):
        players = players.sort_values(by="Draft Prob").copy().reset_index()
        players["Prior"] = 1
        return players

    def sample(self, exclude_players=None, exclude_pos=None):
        # Choose randomly if not all have been visited

        # Remove player names and positions that would be invalid
        players = self.players[~(self.players[cols.NAME_FIELD].isin(exclude_players)) &
                               ~(self.players[cols.POS_FIELD].isin(exclude_pos))]

        # Sample from unselected players first
        if len(players[players["Visits"] == 0]) != 0:
            return np.random.choice(players[players["Visits"] == 0][cols.NAME_FIELD])

        # Otherwise find player with highest weight that appears in list of valid players to sample
        self.update_weights()
        for player in self.weight_order:
            if player in players[cols.NAME_FIELD]:
                return player

    def record_visit(self, player_name, team_value):
        # Update total number of visits and average value of all nodes
        self.total_visits += 1
        curr_visits = self.players.loc[player_name, "Visits"]
        curr_value = self.players.loc[player_name, "Q"]
        new_visits = curr_visits + 1
        new_value = (curr_visits*curr_value + team_value)/(curr_visits + 1)
        self.players.loc[self.players[cols.NAME_FIELD] == player_name, "Visits"] = new_visits
        self.players.loc[self.players[cols.NAME_FIELD] == player_name, "Q"] = new_value

    def update_weights(self):
        self.players["U"] = np.sqrt(2*math.log(self.total_visits)/(self.players["Visits"])) * self.players["Prior"] * self.exploration_constant
        self.players["Weights"] = (self.players["Q"]/self.players["Q"].max()) + self.players["U"]
        self.weight_order = self.players.sort_values(by="Weights", ascending=False)[cols.NAME_FIELD]

    def merge_branch(self, node):
        self.players["Visits"] = self.players["Visits"] + node.players["Visits"]
        self.players["Q"] = ((self.players["Q"]*self.total_visits) + (node.players["Q"]*node.total_visits))/(self.total_visits+node.total_visits)
        self.total_visits = self.total_visits + node.total_visits

    def set_exploration_constant(self, exploration_constant):
        self.exploration_constant = exploration_constant

    def __str__(self):
        return "Total visits: {0}\n{1}".format(self.total_visits,
                                               self.players[[cols.NAME_FIELD, "Draft Prob", cols.ADP_FIELD,
                                                             cols.VORP_FIELD, "Prior", "Visits", "Q", "Weights"]])


class DraftSimSampler(PlayerSampler):
    # Draft aware sampler. Samples from draft posterior distribution using ADP priors
    # Also handles special case for turns when only a finite number of players will be removed
    #   Useful for turn picks (e.g. 2, 11) when you know the exact number of players to remove from pool
    def __init__(self, players, num_adv_picks_to_sim=None, mcmc_exploration_constant=1):
        PlayerSampler.__init__(self, players, mcmc_exploration_constant)

        # Number of players to remove from potential pool of players
        # If None, simluate available player pool each round based on draft probabilities
        self.num_adv_picks_to_sim = num_adv_picks_to_sim

        if self.num_adv_picks_to_sim is None:
            # If not simulating adversary picking one of the desired players
            # Just set probability of being drafted = 1 - probability they're still here
            draft_probs = 1 - self.players["Draft Prob"]
        else:
            # Otherwise set draft probabilities equal to relative player would drafted considering other players were available
            draft_probs = (1-self.players["Draft Prob"])/(1-self.players["Draft Prob"]).sum()

        self.draft_probs = list(zip(self.players[cols.NAME_FIELD], draft_probs))

    def sample(self, exclude_players=None, exclude_pos=None):
        drafted_players = self.simulate_draft()
        exclude_players += drafted_players
        player = super().sample(exclude_players, exclude_pos)
        while player is None:
            player = super().sample(exclude_players, exclude_pos)
        return player

    def simulate_draft(self):
        if self.num_adv_picks_to_sim == 0:
            # No players drafted if around turn (e.g. pick 1 or pick 12)
            return []

        elif self.num_adv_picks_to_sim is None:
            # 'Draft' players independently
            # Remove players based on probability of them being drafted at current position given ADP distribution
            probs = np.random.uniform(size=len(self.players))
            return [self.draft_probs[i] for i in range(len(self.players)) if self.draft_probs[i][1] < probs[i]]

        # Otherwise simulate N adversary picks based on relative ADP probabilities
        return np.random.choice(self.players[cols.NAME_FIELD],
                                p=self.draft_probs,
                                size=self.num_adv_picks_to_sim)


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
        if counter.value() % 100 == 0 and counter.value() != 0:
            logging.info("Sampled {0} teams...".format(counter.value()))
            print("Sampled {0} teams...".format(counter.value()))
        teams.append(mcmc_draft_tree.sample())
        counter.increment()

    # Add teams to shared queue
    output_queue.put((teams, mcmc_draft_tree))


def do_mcmc_draft_chain(mcmc_draft_tree, num_samples, num_chains):
    # Perform MCMC sampling in parallel
    counter = Counter(0)
    results_q = Queue()
    samples_per_thread = int(math.ceil(num_samples / float(num_chains)))
    chains = []
    teams = []
    chain_trees = []
    # Kick off thread workers to sample teams from distribution
    for i in range(num_chains):
        chains.append(Process(target=sample_teams,
                             args=(mcmc_draft_tree,
                                   samples_per_thread,
                                   counter,
                                   results_q)))
        chains[i].start()

    # Add results as they finish
    for chain in chains:
        results = results_q.get()
        # Add to list of teams created by tree
        teams.extend(results[0])
        # Merge chain back into original tree
        mcmc_draft_tree.merge_branch(results[1])

    # Wait for threads to complete and return teams
    for chain in chains:
        chain.join()

    # Mix Chains into single draft tree
    return teams, mcmc_draft_tree


def do_mcmc_tree_search(mcmc_draft_tree, num_samples, num_chains=8, mix_chains_every=2000):
    # Perform MCMC sampling in parallel
    # Number of times independent chains will be mixed
    teams = []
    curr_samples = 0
    while curr_samples < num_samples:

        batch_samples = num_chains * mix_chains_every
        batch_chains = num_chains
        if batch_samples > num_samples-curr_samples:
            batch_samples = int(math.ceil((num_samples-curr_samples)/float(mix_chains_every)))
            batch_chains = int(math.ceil(batch_samples/float(mix_chains_every)))
            print("leftover samples so now we're only running {0} chains with {1} samples".format(batch_chains,
                                                                                                  batch_samples))

        # Run a set of MCMC chains off current mcmc tree and merge results
        print("Draft tree before mixins:\n%s" % mcmc_draft_tree.get_node("root"))
        teams, mcmc_draft_tree = do_mcmc_draft_chain(mcmc_draft_tree, batch_samples, batch_chains)
        print("Merged draft tree after mixins:\n%s" % mcmc_draft_tree.get_node("root"))
        teams += teams

        curr_samples = len(teams)
        print("Number teams so far: {0}".format(len(teams)))

    return teams, mcmc_draft_tree



draftsheet = "/Users/awaldrop/Desktop/ff/projections/draftboard_8-14-19.xlsx"
draft_df = pd.read_excel(draftsheet)

with open("/Users/awaldrop/PycharmProjects/fantasy_draft_tools/draft_configs/LOG.yaml", "r") as stream:
    league_config = yaml.safe_load(stream)

from draftboard import DraftBoard

pd.set_option('display.max_columns', None)
db = DraftBoard(draft_df, league_config)
my_players = db.get_auto_draft_selections(num_to_consider=25, ceiling_confidence=0.2)
#my_players = ["Eric Ebron", "Robby Anderson"]
#exit(0)
my_players = db.potential_picks[cols.NAME_FIELD].tolist()
print(my_players)
injury_risk_model = EmpiricalInjuryModel(league_config)

x = DraftTree(my_players, db, injury_risk_model=injury_risk_model, min_adp_prior=0.05, max_draft_node_size=20)
curr_round = x.start_round

#teams, mcmc_tree = do_mcmc_tree_search(x, num_samples=30000, num_chains=4)
teams, mcmc_tree = do_mcmc_draft_chain(x, num_samples=100000, num_chains=8)
for i in range(curr_round, 15):
    print("ROUND %s:\n%s" % (i, mcmc_tree.get_node(i)))

 # Convert results to dataframe
results = [team.get_summary_dict() for team in teams]
results = pd.DataFrame(results)

def player_on_team(row, player, draft_spot):
    team = row['players'].split(",")
    return team[draft_spot].strip() == player

# Specify which player to evaluate was drafted by each simulated team
for player in my_players:
    results[player] = results.apply(player_on_team,
                                    axis=1,
                                    player=player,
                                    draft_spot=curr_round - 1)
#for i in range(1000):
#    if i % 100 == 0:
#        print("Done this  many: %s" % i)
#    x.sample()

#print(x.draft_tree["1"])
#print(x.draft_tree["2"])
#print(x.draft_tree["3"])
exit()

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

