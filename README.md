# Fantasy Football Draft Tools 

## Description
Utilities for Markov-Chain Monte Carlo Tree Search for Snake Draft Optimization

### Dependencies
* Python3.7+
* pip
* virtualenv

### Download repo
    git clone https://github.com/alexwaldrop/fantasy_draft_tools.git
    
### Environment setup

Install [python 3.7+](https://www.python.org/downloads/) if you don't already have it
    
### Creating virtual envioronment (do this once) 
Create virtual environment directory inside fantasy_draft_tools directory
    
    cd fantasy_draft_tools
    /usr/local/bin/python3 -m venv ./venv
    
* This should create a venv/ directory in your fantasy_draft_tools repo
    
Install dependencies
    
    source venv/bin/activate
    pip install -r requirements.txt
    
### Running program
Change into directory

    cd fantasy_draft_tools
    
Activate virtual environment

    source venv/bin/activate
  
Run programs inside your virtual environment by just typing 'python' in front on exectuable

    python python/do_mcmc_draft_search.py --help
    
    


