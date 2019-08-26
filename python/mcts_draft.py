import numpy as np
from copy import deepcopy
import logging
import json

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


class DraftAction:
    def __init__(self, player, prob):
        self.name = player.name
        self.player = player
        self.prob = prob


class TeamState:
    def __init__(self, team):
        self.team = team

    def is_terminal(self):
        return self.team.is_full

    def take_action(self, action):
        new_team = deepcopy(self.team)
        new_team.draft_player(action.player)
        return TeamState(new_team)

    def __str__(self):
        return ", ".join(self.team.player_names)


class DraftTreeHelper:
    # Compare the expected outcomes from one or more players
    def __init__(self, players_to_compare, draft_board, min_adp_prior=0.05,
                 max_draft_node_size=15, injury_model=None, bench_weight=0.5):

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
        self.min_draft_node_size = None

        # Injury model to use for simulating team season scenarios
        self.injury_model = injury_model

        # Initialize board of
        self.draft_tree = self.init_draft_player_tree()

        # Starter and total value weight to consider for team "payoff"
        self.bench_weight = bench_weight
        self.starter_weight = 0 if self.curr_team.starters_filled else 1

        # Dict for holding draft actions
        self.draft_actions_dict = {}

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

    def init_draft_player_tree(self):
        logging.info("Building universe of likely draft picks at each round...")
        draft_slot_board = {}
        draft_round = self.start_round
        curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, len(self.draft_board.drafted_players))
        last_pick = curr_pick
        while draft_round <= self.max_draft_rounds and curr_pick < self.draft_board.total_players:

            # Get boundaries of next pick for player in current draft slot
            curr_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick)
            next_pick = self.draft_board.get_next_draft_pick_pos(self.draft_slot, curr_pick+1)

            logging.info("Round {0} player distribution (Pick {1})...".format(draft_round,
                                                                              curr_pick))

            if draft_round == self.start_round:
                # Sample root players without sampling from draft posterior distribution (we know they're available)
                possible_players = self.draft_board.get_adp_prob_info(self.players_to_compare, curr_pick)
                possible_players["Draft Prob"] = 1.0


            elif draft_round == self.start_round + 1 and curr_pick-last_pick-1 < len(self.players_to_compare):
                # If one of players to compare will necessarily make it back, sample from special distribution
                # to simulate adversarial picks
                # Really more of an edge case for picks around the turn
                adversary_picks_until_turn = curr_pick-last_pick - 1
                possible_players = self.draft_board.get_adp_prob_info(self.players_to_compare, curr_pick)

                # Probability that player will be taken by one adversary pick
                prob_drafted_once = 1.0/len(self.players_to_compare)

                # Probability that player not taken by any adversary picks
                prob_not_drafted = (1.0 - prob_drafted_once) ** adversary_picks_until_turn
                possible_players["Draft Prob"] = prob_not_drafted

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
                possible_players = self.subset_draftable_players(possible_players)

            # Order by expected return
            #possible_players["ExpReturn"] = possible_players["Draft Prob"]*possible_players[cols.VORP_FIELD]
            #possible_players = possible_players.sort_values(by="ExpReturn", ascending=False)

            # Add node to mcmc tree
            draft_slot_board[str(draft_round)] = possible_players
            logging.info(possible_players[[cols.NAME_FIELD, "Draft Prob", cols.ADP_FIELD, cols.VORP_FIELD]])

            # Move onto next round and increment current pick
            draft_round += 1
            last_pick = curr_pick
            curr_pick = next_pick

        # Set draft tree to be tree just created
        return draft_slot_board

    def subset_draftable_players(self, possible_players, pos_miss_prob=0.01):
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
            if pos_prob_missing[pos] > pos_miss_prob:
                best_players.append(name)
                pos_prob_missing[pos] *= (1-draft_prob)

        # Fill up any remaining slots with WR/RB
        if len(best_players) < self.max_draft_node_size:
            num_to_get = self.max_draft_node_size - len(best_players)
            best_wr_rb = possible_players[possible_players[cols.POS_FIELD].isin(["RB", "WR"])]
            best_wr_rb = best_wr_rb[~best_wr_rb[cols.NAME_FIELD].isin(best_players)].iloc[0:num_to_get]
            best_players.extend(best_wr_rb[cols.NAME_FIELD])

        # Return subsetted data frame
        return possible_players[possible_players[cols.NAME_FIELD].isin(best_players)]

    def get_possible_actions(self, team_state):
        exclude_players = team_state.team.player_names
        exclude_pos = team_state.team.filled_positions
        draft_round = team_state.team.size + 1

        possible_player_id = "{0}_{1}_{2}".format("-".join(sorted(exclude_players)), "-".join(sorted(exclude_pos)), draft_round)
        if possible_player_id in self.draft_actions_dict:
            return self.draft_actions_dict[possible_player_id]

        # Select list of possible players in round
        possible_players = self.draft_tree[str(draft_round)]

        # Remove players and positions that are invalid
        possible_players = possible_players[~(possible_players[cols.NAME_FIELD].isin(exclude_players)) &
                                            ~(possible_players[cols.POS_FIELD].isin(exclude_pos))]
        # Convert each to an action
        draft_actions = list(zip(possible_players[cols.NAME_FIELD], possible_players["Draft Prob"]))
        draft_actions = [DraftAction(player=self.draft_board.get_player(data[0]), prob=data[1]) for data in draft_actions]
        self.draft_actions_dict[possible_player_id] = draft_actions
        return draft_actions

    def get_root(self):
        return TeamState(self.curr_team)

    def rollout(self, team_state):
        team = deepcopy(team_state.team)
        draft_round = team.size + 1
        while draft_round <= self.draft_board.league_config["draft"]["team_size"]:
            # Randomly select player from player pool, simulate whether they're available, and choose best available

            # Get list of available players in this round
            players = self.draft_tree[str(draft_round)][cols.NAME_FIELD].tolist()
            probs = self.draft_tree[str(draft_round)]["Draft Prob"].tolist()

            # Simulate which ones are available
            available_players = [self.draft_board.get_player(players[i]) for i in range(len(players)) if probs[i] > np.random.uniform()]
            # Draft the best available player that can play on the current team
            for available_player in available_players:
                if team.can_add_player(available_player):
                    team.draft_player(available_player)
                    break

            draft_round += 1

        # Simulate a season and return simulated points of starters
        team.simulate_n_seasons(1, self.injury_model)
        return team.get_summary_dict()[cols.SIM_STARTERS_PTS]
