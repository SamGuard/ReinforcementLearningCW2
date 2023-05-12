# ReinforcementLearningCW2
To run this code, install the necessary libraries (pytorch, numpy, gym, matplotlib) and then enter the Jupyter Notebook. Run the first two cells, and then configure your agent (either DDPG or TD3 - one is commented out, while the other isn't). Run this cell, and then the next one. The specified agent will run against the BipedalWalker environment and will begin learning.

"agents.py" contains the implementation for TD3 and DDPG, and "replay.py" contains the ReplayMemory class. 

This code can be found at https://github.com/SamGuard/ReinforcementLearningCW2 - there are various branches with various experiments, but this is a copy of the "cleanup" branch. In the GitHub repository are also our implementations of DQN and SARSA(λ), which did not perform well against the chosen problem.

