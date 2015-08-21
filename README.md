# p4l
prize4life als challenge tests and working examples, for use in PayPal datathon

# supplied code

A working prototype (reference implementation) pipeline. There's a lot to improve in each stage = add rainbows and unicorns where appropriate

1. vectorizer
2. clusterer
3. feature_selector
4. predictor

# code used by datathon's organizers

1. train_test_splitter
2. scorer


# Installation instructions

    git clone git@github.com:ihadanny/p4l-als-z-team.git
    cd p4l-als-z-team
    sudo pip install pip virtualenv --upgrade
    virtualenv venv
    venv/bin/pip install -r requirements.txt
    echo "alias ipy=\"venv/bin/python -c 'import IPython; IPython.terminal.ipapp.launch_new_instance()'\"" >> ~/.bashrc
    
Run `ipython notebook` with:
  
    ipy notebook
    
from within the project directory.
    
