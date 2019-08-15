import os
import numpy as np
import math
from copy import deepcopy
import logging
import pandas as pd
from multiprocessing import Process, Value, Lock, Queue, cpu_count
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
                 max_draft_node_size=15, injury_risk_model=None, mcmc_exploration_constant=1,
                 positional_priors=None, n_rollouts=1):

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

        # Priors to attach globally to certain positions
        self.positional_priors = positional_priors if positional_priors is not None else {pos: 1.0 for pos in self.draft_board.league_config["global"]["pos"]}
        self.validate_positional_priors()

        # Number of rollouts before using wieghts
        self.n_rollouts = n_rollouts

        # Initialize board of
        self.draft_tree = self.init_mcmc_tree()

    def validate_positional_priors(self):
        if self.positional_priors is None:
            return

        # Set any missing positional priors to 1
        for pos in self.draft_board.league_config["global"]["pos"]:
            if pos not in self.positional_priors:
                self.positional_priors[pos] = 1.0

        errors = False
        for pos in self.positional_priors:
            if pos not in self.draft_board.league_config["global"]["pos"]:
                logging.error("Invalid DraftTree! Can't set positional prior on non-existing position: {0}".format(pos))
                errors = True

            elif not isinstance(self.positional_priors[pos], float):
                logging.error("Invalid DraftTree! {0} Positional prior is not a float!".format(pos))
                errors = True

            elif self.positional_priors[pos] > 1 or self.positional_priors[pos] < 0:
                logging.error("Invalid DraftTree! {0} Positional prior must be between [0,1]".format(pos))
                errors = True

        # Raise error if invalid
        if errors:
            raise utils.DraftException("Invalid DraftTree positional priors! See error messages above for details.")

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
                                     mcmc_exploration_constant=self.mcmc_exploration_constant,
                                     n_rollouts=self.n_rollouts)

            elif draft_round == self.start_round + 1 and curr_pick-last_pick-1 < len(self.players_to_compare):
                # If one of players to compare will necessarily make it back, sample from special distribution
                # to simulate adversarial picks
                # Really more of an edge case for picks around the turn
                adversary_picks_until_turn = curr_pick-last_pick - 1
                possible_players = self.draft_board.get_adp_prob_info(self.players_to_compare, curr_pick)
                node = DraftSimSampler(possible_players,
                                       num_adv_picks_to_sim=adversary_picks_until_turn,
                                       mcmc_exploration_constant=self.mcmc_exploration_constant,
                                       n_rollouts=self.n_rollouts)
            else:
                # Otherwise get players whose ADP confidence interval overlaps with current draft window

                # If in last 3 rounds just open it up and try to find the best players
                last_pick = 1000 if self.max_draft_rounds - draft_round <= 3 else next_pick - 1

                possible_players = self.draft_board.get_probable_players_in_draft_window(curr_pick,
                                                                                         last_pick,
                                                                                         prob_thresh=self.min_adp_prior)
                # Remove players from filled team positions
                possible_players = possible_players[~possible_players[cols.POS_FIELD].isin(self.curr_team.filled_positions)]

                # Reduce to a smaller set of players I'd actually draft
                best_players = self.subset_draftable_players(possible_players)
                node = DraftSimSampler(best_players,
                                       positional_priors=self.positional_priors,
                                       mcmc_exploration_constant=self.mcmc_exploration_constant,
                                       n_rollouts=self.n_rollouts)

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

    def sample(self):
        # Sample a team from the draft tree
        team = deepcopy(self.curr_team)
        player_map = {}
        full_positions = []
        for draft_round in range(self.start_round, self.max_draft_rounds+1):

            # Sample until you find a player you can add to the current team
            exclude_players = team.player_names if team.size > 0 else []
            player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample(exclude_players=exclude_players,
                                                                                          exclude_pos=full_positions))
            while not team.can_add_player(player):
                # Only reaosn we can't add player to team would be full position so add to list of filled positions
                full_positions.append(player.pos)
                player = self.draft_board.get_player(self.draft_tree[str(draft_round)].sample(exclude_players=exclude_players,
                                                                                              exclude_pos=full_positions))
            # Add player to current team
            player_map[str(draft_round)] = player.name
            team.draft_player(player)

        # Sample from teams score probability distribution as well
        #print(", ".join([x.name for x in team.starters]))
        team.simulate_n_seasons(1, self.injury_risk_model)

        # Compute value as average of starter and bench points
        sim_results = team.get_summary_dict()
        team_value = (sim_results[cols.SIM_STARTERS_PTS] + sim_results[cols.SIM_TEAM_VORP])/2

        # Update weights on player nodes
        for round in player_map: self.draft_tree[round].record_visit(player_map[round], team_value)

        # Return team after all rounds completed
        return team

    def make_branch(self):
        # Make a copy of self and return branch versions of all current nodes
        branch = deepcopy(self)
        for draft_round in branch.draft_tree:
            branch.draft_tree[draft_round] = branch.draft_tree[draft_round].make_branch()
        return branch

    def merge_branch(self, branch):
        # Merge data from a branch split from draft tree
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

    def set_positional_priors(self, positional_priors=None):
        # Update positional priors and broadcast to all nodes
        if positional_priors is None:
            self.positional_priors = {pos: 1 for pos in self.draft_board.league_config["global"]["pos"]}
        else:
            self.positional_priors = positional_priors
        self.validate_positional_priors()

        # Set new positional priors for any round that's sampling players other than players to compare
        for draft_round in self.draft_tree:
            if isinstance(self.draft_tree[draft_round], DraftSimSampler) and self.draft_tree[draft_round].num_adv_picks_to_sim is None:
                self.draft_tree[draft_round].set_positional_priors(positional_priors)


class PlayerSampler:
    # Special case of draft sampler that picks players uniformly from distribution
    # Assumes all players will be undrafted at time of pick
    def __init__(self, players, mcmc_exploration_constant=1, n_rollouts=1):
        # Compute the absolute probability of sampling each player in distribution
        self.players = self.init_node_priors(players)
        self.players["Visits"] = 0
        self.players["Q"] = 0
        self.players["Weights"] = 0
        self.players = self.players.set_index(cols.NAME_FIELD, drop=False)
        self.total_visits = 0
        self.weight_order = None
        self.exploration_constant = mcmc_exploration_constant
        self.n_rollouts = n_rollouts

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
        if len(players[players["Visits"] < self.n_rollouts]) != 0:
            return np.random.choice(players[players["Visits"] < self.n_rollouts][cols.NAME_FIELD])

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

        # Update number of vists to current branch
        curr_visits = self.players.loc[player_name, "BranchVisits"]
        curr_value = self.players.loc[player_name, "BranchQ"]
        new_visits = curr_visits + 1
        new_value = (curr_visits * curr_value + team_value) / (curr_visits + 1)
        self.players.loc[self.players[cols.NAME_FIELD] == player_name, "BranchVisits"] = new_visits
        self.players.loc[self.players[cols.NAME_FIELD] == player_name, "BranchQ"] = new_value

    def update_weights(self):
        self.players["U"] = np.sqrt(2*math.log(self.total_visits)/(self.players["Visits"])) * self.players["Prior"] * self.exploration_constant
        self.players["Weights"] = (self.players["Q"]/self.players["Q"].max()) + self.players["U"]
        self.weight_order = self.players.sort_values(by="Weights", ascending=False)[cols.NAME_FIELD]

    def make_branch(self):
        # Make branched copy of self for parallel processing
        branch = deepcopy(self)
        branch.players["BranchVisits"] = 0
        branch.players["BranchQ"] = 0
        return branch

    def merge_branch(self, node):
        # Update Q to be weighted average score based on number of visits
        self.players["Q"] = ((self.players["Q"]*self.players["Visits"]) +
                             (node.players["BranchQ"]*node.players["BranchVisits"]))/(self.players["Visits"]+node.players["BranchVisits"])

        # Repace NaNs caused by dividing by 0
        self.players["Q"] = self.players["Q"].fillna(0)

        # Update number of node visits and total visits
        self.players["Visits"] = self.players["Visits"] + node.players["BranchVisits"]
        self.total_visits = self.total_visits + node.players["BranchVisits"].sum()

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
    def __init__(self, players, num_adv_picks_to_sim=None, positional_priors=None, mcmc_exploration_constant=1, n_rollouts=1):
        PlayerSampler.__init__(self, players, mcmc_exploration_constant, n_rollouts)

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

        # Priors to apply by position
        self.positional_priors = positional_priors
        self.players["Prior"] = self.players[cols.POS_FIELD].map(positional_priors)

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

    def set_positional_priors(self, positional_priors):
        self.positional_priors = positional_priors
        # Update positional priors in players table
        self.players["Priors"] = self.players[cols.POS_FIELD].map(positional_priors)


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
        if counter.value() % 200 == 0 and counter.value() != 0:
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
    # Kick off thread workers to sample teams from distribution
    for i in range(num_chains):
        chains.append(Process(target=sample_teams,
                             args=(mcmc_draft_tree.make_branch(),
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


def do_mcmc_tree_search(mcmc_draft_tree, num_samples, num_chains=6, mix_chains_every=5000):
    # Perform MCMC sampling in parallel
    # Number of times independent chains will be mixed
    teams = []
    curr_samples = 0
    while curr_samples < num_samples:

        if mix_chains_every > num_samples-curr_samples:
            # Decrease batch size for last batch if not exactly == mix_chains_every
            mix_chains_every = num_samples-curr_samples
            print("leftover samples so now we're only doing a batch of {0} samples".format(mix_chains_every))

        # Run a set of MCMC chains off current mcmc tree and merge results
        new_teams, mcmc_draft_tree = do_mcmc_draft_chain(mcmc_draft_tree, mix_chains_every, num_chains)
        print("Merged draft tree after mixins:")
        for draft_round in mcmc_draft_tree.draft_tree:
            print(mcmc_draft_tree.get_node(str(draft_round)))
            if int(draft_round) > 6:
                break

        teams += new_teams
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

x = DraftTree(my_players, db, injury_risk_model=injury_risk_model,
              min_adp_prior=0.01,
              max_draft_node_size=20,
              n_rollouts=3)

curr_round = x.start_round

#teams, mcmc_tree = do_mcmc_tree_search(x, num_samples=30000, num_chains=4)
import time
t0 = time.time()
chains = cpu_count()
#teams, mcmc_tree = do_mcmc_draft_chain(x, num_samples=100000, num_chains=chains)
teams, mcmc_tree = do_mcmc_tree_search(x, num_samples=100000)
#teams, mcmc_tree = do_mcmc_draft_chain(x, num_samples=20000, num_chains=6)
print("Took {0} seconds with {1} cores".format(time.time()-t0, chains))

#for i in range(1, 2000):
#    if i % 100 == 0:
#        print("Done %s samples" % i)
#    x.sample()

#mcmc_tree = x
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
results.to_excel("/Users/awaldrop/Desktop/test8.xlsx")
#for i in range(1000):
#    if i % 100 == 0:
#        print("Done this  many: %s" % i)
#    x.sample()

#print(x.draft_tree["1"])
#print(x.draft_tree["2"])
#print(x.draft_tree["3"])
exit()

derp = MCMCResultAnalyzer(my_players, curr_round, teams)
derp.to_excel("/Users/awaldrop/Desktop/ff/test11.xlsx", index=False)
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

