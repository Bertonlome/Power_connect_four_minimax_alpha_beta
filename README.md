README for Connect-4 Game
Overview

The file power_connect_four.py implements a Connect-4-like game locally between two agents or remotely for one agent.


Main Functions
connect_to_server: Establishes a connection to the game server.
send_initial_setup: Sends the initial setup message to the server with the game ID and player's color.
receive_message: Receives messages from the server.
send_move: Sends the player's move to the server.
create_board: Initializes the game board.
display_board: Displays the current state of the board in the console.
apply_gravity: Ensures gravity is applied to the pieces after each move.
drop_piece: Drops a piece in the selected column.
slide_pieces: Slides a run of pieces either left or right.
is_board_full: Checks if the board is full, resulting in a draw.
check_win: Checks if a player has won the game.
heuristic_v3: Evaluates the board state using the new heuristic.
heuristic_old: Evaluates the board state using the old heuristic.
count_runs_v3: Counts the number of runs of 2 or 3 pieces for heuristic evaluation.
minimax: Implements the minimax algorithm with alpha-beta pruning for move selection.


How to Run the Game
1. Running Locally (AI vs. AI)

To launch a game between two AI agents locally:

Uncomment the following lines in the script:
Lines 627-745 (the local game loop).
Line 980 (AI game start).
Comment out line 981 (remote game connection).
Run the script locally. The game will randomly assign the first turn between white (using the old heuristic) and black (using the improved heuristic).
2. Running Remotely (AI vs. Server)

To launch a game remotely:

Keep the original script as-is.
The script connects to the remote McGill server (156trlinux-1.ece.mcgill.ca) using port 12345.
Modify the parameters on line 981 to define the color played (either BLACK or WHITE in full caps) and the game ID (e.g., 1).
Run the script to play against the server. Moves will be echoed back to both players after each turn.
Requirements

The following libraries are required to run the script:

numpy
matplotlib
socket
logging
threading
