import re
import logging
import requests
from fuzzywuzzy import fuzz


from utils import DraftException

def normalize_string(value):
    clean_val = re.sub('[^\w\s-]', '', value).strip().lower()
    return clean_val

def fetch_api_data(request_url):
    logging.info("Fetching data URL: {0}".format(request_url))
    response = requests.get(request_url)
    if response.status_code != 200:
        logging.error("API call unsuccessful! Received following error:\n{0}".format(response.text))
        response.raise_for_status()

    logging.info("API call was success!")
    return response.text

def match_reference_player_name(name, pos, reference_names, reference_pos, match_threshold=90, min_match_threshold=75):
    # Find reference name for a player to harmonize names across two datasets

    # Check to see if reference names, pos are same length
    if len(reference_names) != len(reference_pos):
        err_msg = "Different number of reference names ({0}) " \
                  "and positions ({1})!".format(len(reference_names), len(reference_pos))
        logging.error(err_msg)
        raise DraftException(err_msg)

    # Normalize player names to remove differences in cases and punctuation
    norm_name = normalize_string(name)
    norm_reference_names = {normalize_string(reference_names[i]): (reference_names[i], reference_pos[i]) for i in range(len(reference_names))}

    if norm_name in norm_reference_names and pos == norm_reference_names[norm_name][1]:
        # Return name if name matches and same position
        if name not in reference_names:
            print(norm_name)
            logging.debug("Normalizing resolved names: {0} | {1}".format(norm_name,
                                                                         norm_reference_names[norm_name][0]))
        return norm_reference_names[norm_name][0]

    elif norm_name not in norm_reference_names:
        # Do fuzzy matching to see if name closely matches another name
        # Store results in case none exceed fuzzy match threshold and human input needed
        match_results = []
        for norm_ref_name in norm_reference_names:
            match_ratio = fuzz.partial_ratio(norm_name, norm_ref_name)

            # Return name if fuzzy match is over threshold
            if match_ratio > match_threshold and norm_reference_names[norm_ref_name][1] == pos:
                print(match_ratio)
                logging.warning("Fuzzy match resolved names: {0} | {1}".format(norm_name,
                                                                               norm_reference_names[norm_ref_name][0]))
                return norm_reference_names[norm_ref_name][0]

            # Otherwise add match results to list of names
            match_results.append((norm_ref_name, match_ratio))

    elif norm_name in norm_reference_names and pos != norm_reference_names[norm_name][1]:
        # Raise error if there's a player that matches but for different position. That shouldn't happen.
        logging.error("Player name matched but we disagree on position:\n"
                      "Name: {0} ({1})\n"
                      "Ref Name: {2} ({3})".format(name,
                                                   pos,
                                                   norm_reference_names[norm_name][0],
                                                   norm_reference_names[norm_name][1]))
        raise DraftException("Player name matches but positional disagreement between "
                             "projections/ADP. See log for details.")

    # If no matches found > match threshold, ask user if any matches are correct
    match_results = sorted(match_results, key=lambda x: x[1], reverse=True)
    for match_result in match_results:
        # Break loop and return if the next closest match is below minimum match threshold
        if match_result[1] < min_match_threshold:
            return None

        # Ask for user input to determine whether match is correct
        ref_player = norm_reference_names[match_result[0]][0]
        ref_pos    = norm_reference_names[match_result[0]][1]
        is_match = None
        while is_match not in ["0", "1"]:
            is_match = input("Is this the same player (match score: {0})? "
                             "{0} ({1}) and {2} ({3}) [0=No, 1=Yes]: ".format(match_result[1],
                                                                              name,
                                                                              pos,
                                                                              ref_player,
                                                                              ref_pos))
        # Return player name if user thinks it's a match
        if is_match == "1":
            return ref_player

    # If user loops through all potential matches and doesn't agree, return None
    return None
