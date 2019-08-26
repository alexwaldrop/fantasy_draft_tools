import argparse
import logging
import os
import pandas as pd
import yaml

import utils
import constants as cols
import mcts
import mcts_draft
import draftboard

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
    argparser_obj.add_argument("--draftboard",
                               action="store",
                               type=file_type,
                               dest="db_file",
                               required=True,
                               help="Path to draft board")

    # Path to league config
    argparser_obj.add_argument("--league-config",
                               action="store",
                               type=file_type,
                               dest="league_config",
                               required=True,
                               help="Path to league_config file")

    argparser_obj.add_argument("--time",
                               action="store",
                               type=int,
                               dest="time",
                               required=False,
                               default=10,
                               help="Time in minutes to run")

    argparser_obj.add_argument("--rollouts",
                               action="store",
                               type=int,
                               dest="n_rollouts",
                               required=False,
                               default=3,
                               help="Humber of rollouts during expansion")

    argparser_obj.add_argument("--explore-const",
                               action="store",
                               type=float,
                               dest="exp_constant",
                               required=False,
                               default=0.25,
                               help="Exploration constat")

    argparser_obj.add_argument("--sim-injury",
                               action="store",
                               type=bool,
                               dest="sim_injury",
                               required=False,
                               default=False,
                               help="Exploration constat")

    argparser_obj.add_argument("--bench-weight",
                               action="store",
                               type=float,
                               dest="bench_weight",
                               required=False,
                               default=0.5,
                               help="Scaling factor for how much you value a deep bench")

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

def main():
    # Configure argparser
    argparser = argparse.ArgumentParser(prog="do_mcmc_draft_search")
    configure_argparser(argparser)

    # Parse the arguments
    args = argparser.parse_args()

    # Configure logging
    utils.configure_logging(args.verbosity_level)

    # Get names of input/output files
    draftboard_file  = args.db_file
    league_config_file = args.league_config
    time_to_run = args.time
    exploration_const = args.exp_constant
    bench_weight = args.bench_weight
    n_rollouts = args.n_rollouts
    sim_injury = args.sim_injury

    # Read config file
    with open(league_config_file, "r") as stream:
        league_config = yaml.safe_load(stream)

    # Read draft sheet
    draft_df = pd.read_excel(draftboard_file)

    # Initialize and validate draft board
    db = draftboard.DraftBoard(draft_df, league_config)


    # Get my potential picks
    my_players = db.potential_picks[cols.NAME_FIELD].tolist()
    if not my_players:
        my_players = db.get_auto_draft_selections()
    logging.info("Players to compare: {0}".format(", ".join(my_players)))

    injury_risk_model = mcts_draft.EmpiricalInjuryModel(league_config) if sim_injury else None

    draft_tree_helper = mcts_draft.DraftTreeHelper(my_players,
                                                   db,
                                                   min_adp_prior=0.01,
                                                   max_draft_node_size=25,
                                                   injury_model=injury_risk_model,
                                                   bench_weight=bench_weight)

    # Initialize MCTS for mcmc tree search
    mcmc_tree = mcts.MCTS(root_state=draft_tree_helper.get_root(),
                          tree_helper=draft_tree_helper,
                          time_limit=time_to_run*1000*60,
                          num_rollouts=n_rollouts,
                          exploration_constant=exploration_const)

    # Do MCTS search and output best player
    best_action = mcmc_tree.search()
    logging.info("THIS THE BEST PLAYER:\n"
                 "**********************************************\n\n{0}\n\n"
                 "**********************************************".format(best_action.upper()))

    # Also output best player for next round from best player
    best_node = mcmc_tree.root.children[best_action]
    logging.info("Next round best players: ")
    for child in best_node.children:
        logging.info(best_node.children[child])

if __name__ == "__main__":
    main()