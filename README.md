# Fantasy Football Draft Tools 

## Description
Library of utilities for generating and customizing Value-Based Drafting spreadsheets from existing player projections

### Dependencies
* Python3.7+
* Pandas

### Install 
    git clone https://github.com/alexwaldrop/fantasy_draft_tools.git
    
### make_value_cheatsheet.py
Command line tools for producing VBD spreadsheet from player projections

    python3 make_value_cheatsheet.py  --input INPUT_FILE --output OUTPUT_FILE
                                      [--qb NUM_QB] [--rb NUM_RB] [--wr NUM_WR]
                                      [--te NUM_TE] [--league-size] [-v]
    optional arguments:
    -h, --help                  show this help message and exit
    --input INPUT_FILE          Path to excel spreadsheet
    --output OUTPUT_FILE        Path to value cheatsheet output file.
    --qb NUM_QB                 Expected total number of QBs drafted
    --rb NUM_RB                 Expected total number of RBs drafted
    --wr NUM_WR                 Expected total number of WRs drafted
    --te NUM_TE                 Expected total number of TEs drafted
    --league-size LEAGUE_SIZE   League Size
    -v                          Increase verbosity of the program.Multiple -v's
                                increase the verbosity level: 0 = Errors 1 = Errors +
                                Warnings 2 = Errors + Warnings + Info 3 = Errors +
                                Warnings + Info + Debug

#### Expected input file
Excel spreadsheet with single sheet (.xlsx) and required columns:
* Name: player name
* Pos: position (must include RB, WR, TE, QB)
* ADP: Average draft position or player rank. Numerical rank of players as they are expected to be drafted. 
* Points: Projected points

#### Suggested parameters for 12 team league 
Suggested from this [footballguys](https://www.footballguys.com/05vbdrevisited.htm) post
* 15 QB
* 36 RB
* 38 WR
* 8 TE

This is an updated and additionally [helpful post from football guys](https://www.fantasypros.com/2017/06/what-is-value-based-drafting/)

#### Additional considerations
Count number of players taken in previous draft at each position after top 100 players. Some say 180.

#### Output statistics
* **Replacement value**: value of best waiver replacement at a player's position
* **VORP**: Value above replacement player. Points above best best waiver replacement at position. 
* **Draft Rank**: Expected position player will be drafted based on ADP.
* **Draft slot**: Expected draft slot that will have the first opportunity to draft a player. 
* **ADP inefficiency**: Difference between VBD and ADP ranks (negative = bad value at current ADP, positive = good value at current ADP)
* **VONA**: value over next available. How much better player is than best available at position next time you draft. Also interpreted as the cost of not drafting that player at their current ADP.  


