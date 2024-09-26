import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
import socket
import logging
import threading

WHITE = 0
BLACK = 1
PLAYER = 1
AI = 0
DEPTH_BEGIN = 4
DEPTH = 5
ROW_COUNT = 8
COLUMN_COUNT= 8
NB_MOVE_PLOT = 10
totTurn = 0
node_visited = 0
# Ensure the times and nodes dictionaries are initialized for totTurn
timesTurn = {totTurn: [] for totTurn in range (0, 50)}
nodesTurn = {totTurn: [] for totTurn in range (0, 50)}
timesTurnWhite = {totTurn: [] for totTurn in range (0, 50)}
nodesTurnWhite = {totTurn: [] for totTurn in range (0, 50)}
times = {depth: [] for depth in range(DEPTH_BEGIN, DEPTH)}
nodes = {depth: [] for depth in range(DEPTH_BEGIN, DEPTH)}

# Set up logging configuration
logging.basicConfig(filename='game_log.txt', level=logging.INFO, format='%(message)s')

# Function to connect to the server
def connect_to_server(host, port, timeout=60):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Try to connect to the server
        s.connect((host, port))
        s.settimeout(timeout)
        print(f"Connected to {host}:{port}")
    except socket.timeout:
        print(f"Connection to {host}:{port} timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error connecting to the server: {e}")
        return None
    return s

def send_initial_setup(s, game_id, color):
    initial_message = f"{game_id} {color}"
    try:
        s.sendall(f"{initial_message}\n".encode('utf-8'))
        print(f"Sent: {initial_message}")
    except Exception as e:
        print(f"Error sending message: {e}")
    success = receive_message(s)
    if success:
        return True

def receive_message(s):
        try:
            response = s.recv(1024).decode().strip()
            return True, response
        except Exception as e:
            print(f"Error receiving message: {e}")
            return True, None

def send_move(s, move):
    message = ''
    if move[0] == 'D':
        # If the move is a drop move
        message =  f'D {move[1] + 1}'
    elif move[0] == 'L':
        # If the move is a slide move
        message =  f'L {move[1] +1} {move[2]+1}'
    elif move[0] == 'R':
        message = f'R {move[1]+1} {move[2]+1}'
    else:
        raise ValueError("Invalid move type. Must be 'D' or 'L'.")
    try:
        s.sendall(f"{message}\n".encode())
        print(f"Sent: {message}")
        return True
    except Exception as e:
            print(f"Error sending message: {e}")
            return False        



#Create the board
def create_board():
    board = np.full((ROW_COUNT,COLUMN_COUNT),' ', dtype=str)
    return board



#print the board
def display_board(board):
    for row in board:
        print(",".join(row))
        logging.info(",".join(row))
    print("###############")#floor of the board
    logging.info("###############")#floor


def apply_gravity(board):
    for col in range(8):  # Iterate through each column
        # Start from the bottom of the column and work upwards
        for row in range(7, -1, -1):
            if board[row][col] == ' ':  # If the current position is empty
                # Look for the next non-empty piece above it
                for above_row in range(row - 1, -1, -1):
                    if board[above_row][col] != ' ':
                        # Move the piece down to fill the empty space
                        board[row][col] = board[above_row][col]
                        board[above_row][col] = ' '  # Set the old position to empty
                        break  # Exit the loop since we made a move


# Drop a piece into a column
def drop_piece(board, column, player):
    column = int(column)
    column_index = column - 1  # To match 1-based indexing from input
    for row in range(7, -1, -1):  # Start from the bottom row
        if board[row][column_index] == ' ':
            board[row][column_index] = str(player)
            return row, column_index  # Return position where the piece was placed
    return -1, -1  # Invalid move if the column is full

def slide_pieces(board, x, y, direction):
    """
    Slide a run of pieces either left (L) or right (R), starting at (x, y).
    
    :param board: 2D list representing the game board.
    :param x: Column number of the starting piece.
    :param y: Row number of the starting piece.
    :param direction: 'L' for left slide, 'R' for right slide.
    :return: Updated board after the slide or False if invalid move.
    """
    try:
        player_piece = board[y][x]
        if player_piece == ' ':
            return False
        opponent_piece = 1 if int(player_piece) == 0 else 0
    except ValueError:
        print("Error detected")
        return False

    
    # Check if there is a valid run of 2 or 3 pieces starting at (x, y)
    if direction == 'L':
        if x - 2 >= 0 and board[y][x-1] == player_piece and board[y][x-2] == player_piece:
            run_length = 3
        elif x - 1 >= 0 and board[y][x-1] == player_piece:
            run_length = 2
        else:
            return False  # Invalid move
        
        opponent_piece_x = x - run_length
        if opponent_piece_x >= 0 and board[y][opponent_piece_x] != player_piece and board[y][opponent_piece_x] != ' ':
            # Check if the opponent's piece is valid for the slide
            if opponent_piece_x - 1 < 0:  # Opponent piece at the left edge
                # Perform slide left, and push opponent piece out of the board
                for i in range(run_length):
                    board[y][x - i - 1] = player_piece  # Move player pieces left
                board[y][x] = ' '  # Empty the original position
                # Opponent piece is effectively removed from the game
                #print("Slide piece " + str(x + 1) + " " + str(y + 1) + " in direction " + direction + "\n")
                return True
            elif opponent_piece_x - 1 >= 0 and board[y][opponent_piece_x - 1] == ' ':
                # Perform slide left
                for i in range(run_length):
                    board[y][x - i - 1] = player_piece  # Move player pieces left
                board[y][x] = ' '  # Empty the original position
                board[y][opponent_piece_x - 1] = opponent_piece  # Move opponent piece
                # Call the gravity function
                #print("Slide piece " + str(x + 1) + " " + str(y + 1) + " in direction " + direction + "\n")
                return True
            else:
                return False
    elif direction == 'R':
        if x + 2 < len(board[0]) and board[y][x+1] == player_piece and board[y][x+2] == player_piece:
            run_length = 3
        elif x + 1 < len(board[0]) and board[y][x+1] == player_piece:
            run_length = 2
        else:
            return False  # Invalid move
        
        opponent_piece_x = x + run_length
        if opponent_piece_x < len(board[0]) and board[y][opponent_piece_x] != player_piece and board[y][opponent_piece_x] != ' ':
            # Check if the opponent's piece is valid for the slide
            if opponent_piece_x + 1 >= len(board[0]):  # Opponent piece at the right edge
                # Perform slide right, and push opponent piece out of the board
                for i in range(run_length):
                    board[y][x + i + 1] = player_piece  # Move player pieces right
                    board[y][x] = ' '  # Empty the original position
                # Opponent piece is effectively removed from the game
                #print("Slide piece " + str(x + 1) + " " + str(y + 1) + " in direction " + direction + "\n")
                return True
            elif opponent_piece_x + 1 < len(board[0]) and board[y][opponent_piece_x + 1] == ' ':
                # Perform slide right
                for i in range(run_length):
                    board[y][x + i + 1] = player_piece  # Move player pieces right
                board[y][x] = ' '  # Empty the original position
                board[y][opponent_piece_x + 1] = opponent_piece  # Move opponent piece
                # Call the gravity function
                #print("Slide piece " + str(x + 1) + " " + str(y + 1) + " in direction " + direction + "\n")
                return True
            else:
                return False
    return False


# Check if the board is full (a draw)
def is_board_full(board):
    return all(board[0][col] != ' ' for col in range(8))

# Check for win after the move
def check_win(board):
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Down, Right, Diagonal Down-Right, Diagonal Down-Left

    for r in range(8):
        for c in range(8):
            piece = board[r][c]
            if piece in ['0', '1']:  # Check only for players' pieces
                # Check in all directions from this piece
                for direction in directions:
                    count = 1  # Start with the current piece
                    for d in [-1, 1]:  # Check both directions
                        rr, cc = r, c
                        while True:
                            rr += direction[0] * d
                            cc += direction[1] * d
                            if 0 <= rr < 8 and 0 <= cc < 8 and board[rr][cc] == piece:
                                count += 1
                            else:
                                break
                        if count >= 4:  # If 4 or more pieces are connected, return the winning player
                            return piece

    return None

def heuristic_evaluation(board):
    white_runs2, white_runs3 = count_runs(board, '1')  # Assuming '1' represents white
    black_runs2, black_runs3 = count_runs(board, '0')  # Assuming '0' represents black
    score = (5 * white_runs3) + (3 * white_runs2) - (4 * black_runs3) - (3 * black_runs2) + center_control(board, AI)
    return score

def heuristic_v3(board, player):
    white_runs2, white_runs3, white_open_square_near_2, _white_open_square_near_3 = count_runs_v3(board, str(WHITE))  # Assuming '1' represents white
    black_runs2, black_runs3, black_open_square_near_2, black_open_square_near_3 = count_runs_v3(board, str(BLACK))  # Assuming '0' represents black
    if player == WHITE:
        score = (5 * white_runs3) + (3 * white_runs2) + (1* white_open_square_near_2) + (2* _white_open_square_near_3)
        score -= 1.5*((4 * black_runs3) + (3 * black_runs2) + (1* black_open_square_near_2) + (2* black_open_square_near_3))
        score += center_control(board, player)
        # Add score for ejectable opponent pieces
        ejectable_black_pieces = check_ejectable_pieces(board, player)
        score += 2 * ejectable_black_pieces  #weight of 5
    else:
        score = (5 * black_runs3) + (3 * black_runs2) + (2* black_open_square_near_2) + (3* black_open_square_near_3)
        score -= 1.5*((4 * white_runs3) + (3 * white_runs2) + (2* white_open_square_near_2) + (3* _white_open_square_near_3))
        score += center_control(board, player)
        # Add score for ejectable opponent pieces
        ejectable_white_pieces = check_ejectable_pieces(board, player)
        score += 2 * ejectable_white_pieces  # weight of 5
    return score

def heuristic_old(board):
    white_runs = count_runs_old(board, '1')  # Assuming '1' represents white
    black_runs = count_runs_old(board, '0')  # Assuming '0' represents black
    return black_runs - white_runs

def check_ejectable_pieces(board, player):
    opponent = str(WHITE) if player == BLACK else str(BLACK)
    ejectable_pieces = 0
    
    # Check the left and right edges for potential ejections
    for row in range(len(board)):
        # Check left edge
        if board[row][0] == opponent:
            if board[row][1] == player and (row - 1 >= 0 and board[row - 1][1] == player or row + 1 < len(board) and board[row + 1][1] == player):
                ejectable_pieces += 1
                
        # Check right edge
        if board[row][-1] == opponent:
            if board[row][-2] == player and (row - 1 >= 0 and board[row - 1][-2] == player or row + 1 < len(board) and board[row + 1][-2] == player):
                ejectable_pieces += 1
                
    return ejectable_pieces

def center_control(board, piece):
    score = 0
    opponent_piece = WHITE if piece == BLACK else BLACK

    # Evaluation board for scoring positions on the game board.

    evaluation_board = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 3, 3, 3, 3, 3, 3, 0],
                                 [0, 3, 9, 9, 9, 9, 9, 0],
                                 [0, 3, 9, 13, 13, 9, 3, 0],
                                 [0, 3, 9, 13, 13, 9, 3, 0],
                                 [0, 3, 9, 9, 9, 9, 3, 0],
                                 [0, 3, 5, 7, 7, 5, 3, 0],
                                 [0, 0, 0, 5, 5, 0, 0, 0]])
    
        # Loop through the board to find pieces and calculate score.
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            value = board[row,col]
            if board[row, col] == str(piece):  # Check if the current position has the player's piece
                score += evaluation_board[row, col]  # Add the corresponding score from evaluation_board

    # You can also calculate the opponent's score if needed
    opponent_score = 0
    for row in range(board.shape[0]):
        for col in range(board.shape[1]):
            if board[row, col] == str(opponent_piece):  # Check if the current position has the opponent's piece
                opponent_score += evaluation_board[row, col]  # Add the corresponding score from evaluation_board

    # Final score can still consider the opponent's score if needed
    final_score = score - opponent_score
    return final_score

    # Calculate scores for the given player's and opponent's pieces on the board.
    piece_score = np.sum(evaluation_board[board == piece])
    opponent_score = np.sum(evaluation_board[board == opponent_piece])

    # Calculate the final score by subtracting the opponent's score from the player's score.
    score = piece_score - opponent_score
    return score

def count_runs_old(board, piece):
    # Count all horizontal, vertical, and diagonal runs of 2 or more
    runs = 0
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == piece:
                for direction in directions:
                    if check_run(board, x, y, direction, piece):
                        runs += 1
    return runs

def count_runs_v3(board, player):
    """Count runs of 2 and 3 for the given player on the board and check for open squares adjacent to runs."""
    runs_of_2 = 0
    runs_of_3 = 0
    open_squares_near_2 = 0
    open_squares_near_3 = 0
    
    # Check horizontal, vertical, and diagonal runs
    for row in range(len(board)):
        for col in range(len(board[0])):
            # Check horizontal
            if col + 2 < len(board[0]) and all(board[row][col + i] == player for i in range(3)):
                runs_of_3 += 1
                # Check open squares adjacent to the run
                if col - 1 >= 0 and board[row][col - 1] == ' ':
                    open_squares_near_3 += 1  # left side
                if col + 3 < len(board[0]) and board[row][col + 3] == ' ':
                    open_squares_near_3 += 1  # right side
            if col + 1 < len(board[0]) and all(board[row][col + i] == player for i in range(2)):
                runs_of_2 += 1
                # Check open squares adjacent to the run
                if col - 1 >= 0 and board[row][col - 1] == ' ':
                    open_squares_near_2 += 1  # left side
                if col + 2 < len(board[0]) and board[row][col + 2] == ' ':
                    open_squares_near_2 += 1  # right side

            # Check vertical
            if row + 2 < len(board) and all(board[row + i][col] == player for i in range(3)):
                runs_of_3 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and board[row - 1][col] == ' ':
                    open_squares_near_3 += 1  # top side
                if row + 3 < len(board) and board[row + 3][col] == ' ':
                    open_squares_near_3 += 1  # bottom side
            if row + 1 < len(board) and all(board[row + i][col] == player for i in range(2)):
                runs_of_2 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and board[row - 1][col] == ' ':
                    open_squares_near_2 += 1  # top side
                if row + 2 < len(board) and board[row + 2][col] == ' ':
                    open_squares_near_2 += 1  # bottom side

            # Check diagonal (bottom-right)
            if row + 2 < len(board) and col + 2 < len(board[0]) and all(board[row + i][col + i] == player for i in range(3)):
                runs_of_3 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == ' ':
                    open_squares_near_3 += 1  # top-left side
                if row + 3 < len(board) and col + 3 < len(board[0]) and board[row + 3][col + 3] == ' ':
                    open_squares_near_3 += 1  # bottom-right side
            if row + 1 < len(board) and col + 1 < len(board[0]) and all(board[row + i][col + i] == player for i in range(2)):
                runs_of_2 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and col - 1 >= 0 and board[row - 1][col - 1] == ' ':
                    open_squares_near_2 += 1  # top-left side
                if row + 2 < len(board) and col + 2 < len(board[0]) and board[row + 2][col + 2] == ' ':
                    open_squares_near_2 += 1  # bottom-right side

            # Check diagonal (bottom-left)
            if row + 2 < len(board) and col - 2 >= 0 and all(board[row + i][col - i] == player for i in range(3)):
                runs_of_3 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and col + 1 < len(board[0]) and board[row - 1][col + 1] == ' ':
                    open_squares_near_3 += 1  # top-right side
                if row + 3 < len(board) and col - 3 >= 0 and board[row + 3][col - 3] == ' ':
                    open_squares_near_3 += 1  # bottom-left side
            if row + 1 < len(board) and col - 1 >= 0 and all(board[row + i][col - i] == player for i in range(2)):
                runs_of_2 += 1
                # Check open squares adjacent to the run
                if row - 1 >= 0 and col + 1 < len(board[0]) and board[row - 1][col + 1] == ' ':
                    open_squares_near_2 += 1  # top-right side
                if row + 2 < len(board) and col - 2 >= 0 and board[row + 2][col - 2] == ' ':
                    open_squares_near_2 += 1  # bottom-left side

    return runs_of_2, runs_of_3, open_squares_near_2, 2* open_squares_near_3

def count_runs(board, player):
    """Count runs of 2 and 3 for the given player on the board."""
    runs_of_2 = 0
    runs_of_3 = 0
    open_square_near_2 = 0
    open_square_near_3 = 0
    
    # Check horizontal, vertical, and diagonal runs
    for row in range(len(board)):
        for col in range(len(board[0])):
            # Check horizontal
            if col + 2 < len(board[0]) and all(board[row][col + i] == player for i in range(3)):
                runs_of_3 += 1
            if col + 1 < len(board[0]) and all(board[row][col + i] == player for i in range(2)):
                runs_of_2 += 1

            # Check vertical
            if row + 2 < len(board) and all(board[row + i][col] == player for i in range(3)):
                runs_of_3 += 1
            if row + 1 < len(board) and all(board[row + i][col] == player for i in range(2)):
                runs_of_2 += 1

            # Check diagonal (bottom-right)
            if row + 2 < len(board) and col + 2 < len(board[0]) and all(board[row + i][col + i] == player for i in range(3)):
                runs_of_3 += 1
            if row + 1 < len(board) and col + 1 < len(board[0]) and all(board[row + i][col + i] == player for i in range(2)):
                runs_of_2 += 1

            # Check diagonal (bottom-left)
            if row + 2 < len(board) and col - 2 >= 0 and all(board[row + i][col - i] == player for i in range(3)):
                runs_of_3 += 1
            if row + 1 < len(board) and col - 1 >= 0 and all(board[row + i][col - i] == player for i in range(2)):
                runs_of_2 += 1

    return runs_of_2, runs_of_3

def check_run(board, x, y, direction, piece):
    run_length = 0
    dx, dy = direction
    for i in range(2):  # Check at least 2 pieces in a row
        new_x = x + i * dx
        new_y = y + i * dy
        if 0 <= new_x < len(board[0]) and 0 <= new_y < len(board) and board[new_y][new_x] == piece:
            run_length += 1
        else:
            break
    return run_length >= 2

def get_all_possible_moves(board, player):
    '''    if player == True:
        myPlayer = 0
    else:
        myPlayer = 1'''
    moves = []
    # Add drop moves (D <column>)
    for col in range(len(board[0])):  # Check all columns
        if board[0][col] == ' ':  # If the top of the column is empty
            moves.append(('D', col))  # 'D' for drop, and the column number
    
    # Add slide moves (L <x y> or R <x y>)
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == str(player):  # If it's the player's piece
                board_copy = board.copy()
                if slide_pieces(board_copy,col,row,'L') == True:  # Space to slide left
                    moves.append(('L', col, row))
                # Check if sliding right is valid
                board_copy = board.copy()
                if slide_pieces(board_copy,col,row,'R') == True:  # Space to slide right
                    moves.append(('R', col, row))
    return moves


def get_next_open_row(board, col):
    for r in range(ROW_COUNT - 1, 0, -1):
        if board[r][col] == ' ':  # Check for an empty spot
            return r
    return None  # No open row

def apply_move(board, move, player):

    if move[0] == 'D':  # Drop move
        col = move[1] +1
        # Drop the piece into the selected column
        drop_piece(board, col, player)

    elif move[0] == 'L':  # Slide left move
        x, y = move[1], move[2]
        # Slide a run of pieces left
        slide_pieces(board, x, y, 'L')

    elif move[0] == 'R':  # Slide right move
        x, y = move[1], move[2]
        # Slide a run of pieces right
        slide_pieces(board, x, y, 'R')

    return board

def apply_move_for_heuristic(board, move, player):
    board_copy = board.copy()
    if move[0] == 'D':  # Drop move
        col = move[1] +1
        # Drop the piece into the selected column
        drop_piece(board_copy, col, player)

    elif move[0] == 'L':  # Slide left move
        x, y = move[1], move[2]
        # Slide a run of pieces left
        slide_pieces(board_copy, x, y, 'L')

    elif move[0] == 'R':  # Slide right move
        x, y = move[1], move[2]
        # Slide a run of pieces right
        slide_pieces(board_copy, x, y, 'R')

    return board_copy

def random_move(board, player):
    move = []
    # Generate a random integer between 1 and 6
    x = random.randint(1, 6)
    
    # Create the string "D x"
    move.append(('D', {x}))
    
    # Return or print the result (to simulate sending)
    return move

def minimax_with_timeout(board,player, depth, results):
    start_time = time.time()
    move, minimax_score = minimax(board, player, depth, True, -math.inf, math.inf, "v3")  # naive heuristic
    elapsed_time = time.time() - start_time
    # Store the result in the shared variable
    results[0] = (move, minimax_score, elapsed_time)


def minimax(board, player, depth, maximizing_player, alpha, beta, heuristic):
    global node_visited
    node_visited += 1
    moves = get_all_possible_moves(board,player)
    
    random.shuffle(moves)
    # Create a list of (move, heuristic_value) tuples
    '''evaluated_moves = [(move, heuristic_v3(apply_move_for_heuristic(board, move, player),player)) for move in moves]
    
    # Sort moves based on heuristic values (descending for maximizing player, ascending for minimizing player)
    if maximizing_player:
        evaluated_moves.sort(key=lambda x: x[1], reverse=True)  # Sort by heuristic value for maximizing player
    else:
        evaluated_moves.sort(key=lambda x: x[1])  # Sort by heuristic value for minimizing player'''

    # Check if we reached terminal state (win/loss) or depth cutoff
    if player == WHITE:
        opponent = BLACK
        if check_win(board) == str(WHITE):#0 = AI
            return (None, 10000)
        elif check_win(board) == str(PLAYER):#0 = player
            return (None, -10000)
    elif player == BLACK:
        opponent = WHITE
        if check_win(board) == str(WHITE):#0 = AI
            return (None, -10000)
        elif check_win(board) == str(PLAYER):#0 = player
            return (None, 10000)
    if is_board_full(board):
        return (None, 0)
    if depth == 0:
        if heuristic =="new":
            return (None, heuristic_evaluation(board))
        elif heuristic =="old":
            return(None, heuristic_old(board))
        elif heuristic == "v3":
            return(None, heuristic_v3(board, player))
    
    if maximizing_player: #maximize
        value = -math.inf
        finalmove = random.choice(moves)
        for move in moves:
            new_board = board.copy()
            apply_move(new_board, move, player)
            eval = minimax(new_board, player, depth - 1, False, alpha, beta, heuristic)[1]
            
            #update the best move and alpha value
            if eval > value:
                value = eval
                finalmove = move
            alpha = max(alpha, eval)

            if alpha >= beta:
                break  # Alpha-beta pruning '''
        return finalmove, value
    
    else:   # minimize
        value = math.inf
        finalmove = random.choice(moves)
        for move in moves:
            new_board = board.copy()
            apply_move(new_board, move, opponent)
            eval = minimax(new_board, player, depth - 1, True, alpha, beta, heuristic)[1]
            
            if eval < value:
                value = eval
                finalmove = move
            beta = min(beta, value)
            if alpha >= beta:
                break  # Alpha-beta pruning'''
        return finalmove, value

'''def play_game_local():
    print("Starting a new game!\n")
    logging.info("Starting a new game!\n")

    board = create_board()

    game_over = False
    turn = random.randint(0,1)
    print(f"Player {turn} begins!")
    logging.info(f"Player {turn} begins!")
    global totTurn
    totTurn = 0

    # Game loop
    while not game_over:
        print(f"Turn number : {totTurn}")
        logging.info(f"Turn number : {totTurn}")

        if turn == WHITE:
            # Use the minimax algorithm to find the best move for the AI.
            for depth in range(DEPTH_BEGIN, DEPTH):
                global node_visited
                node_visited = 0
                start_time = time.time()
                move, minimax_score = minimax(board, WHITE, depth, True, -math.inf, math.inf,"old") #naive heuristic
                end_time = time.time()

                elapsed_time = end_time - start_time
                #times[depth].append(elapsed_time)
                #nodes[depth].append(node_visited)
                print(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")
                logging.info(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")

            #timesTurnImproved[totTurn].append(elapsed_time)
            #nodesTurnImproved[totTurn].append(node_visited)
            print(f"Player White makes move: {move}")
            logging.info(f"Player White makes move: {move}")
            if move != None:
                apply_move(board,move,WHITE)
            apply_gravity(board)

            # Check if Player WHITE has a winning move.
            result = check_win(board)
            if result == str(WHITE):
                game_over = True
                display_board(board)
                print("\n")
                print("WHITE wins!")
                logging.info("\n")
                logging.info("WHITE wins!")
            elif result == str(BLACK):
                game_over = True
                display_board(board)
                print("\n")
                print("BLACK has won!")
                logging.info("\n")
                logging.info("BLACK has won!")
                break

            # Print the current game board and switch to the next turn.
            display_board(board)
            print("\n")
            logging.info("\n")
            turn += 1
            turn= turn % 2
            #if totTurn >= 3:
                #plot_avg_time_results(times[depth])

        if turn == BLACK and not game_over:
            print("\n")
            logging.info("\n")
            # Use the minimax algorithm to find the best move for the AI.
            for depth in range(DEPTH_BEGIN, DEPTH):
                node_visited = 0
                start_time = time.time()
                move, minimax_score = minimax(board, BLACK, depth, True, -math.inf, math.inf, "v3")#new heuristic
                end_time = time.time()

                elapsed_time = end_time - start_time
                #times[depth].append(elapsed_time)
                #nodes[depth].append(node_visited)
                print(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")
                logging.info(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")
            timesTurn[totTurn].append(elapsed_time)
            nodesTurn[totTurn].append(node_visited)
            print(f"Player Black makes move: {move}")
            logging.info(f"Player Black makes move: {move}")

            if move != None:
                apply_move(board,move,BLACK)
            apply_gravity(board)

            # Check if Player 2 (AI) has a winning move.
            result = check_win(board)
            if result == str(BLACK):
                game_over = True
                display_board(board)
                print("\n")
                print("BLACK has won!")
                logging.info("\n")
                logging.info("BLACK has won!")
            elif result == str(WHITE):
                game_over = True
                display_board(board)
                print("\n")
                print("WHITE has won!")
                logging.info("\n")
                logging.info("WHITE has won!")
                break

            # Print the current game board and switch to the next turn.
            display_board(board)
            print("\n")
            logging.info("\n")
            turn += 1
            turn= turn % 2
            totTurn +=1
    #plot_avg_time_nodes_per_turn()
    #plot_avg_time_results()'''

def play_remote(player, game_number):
    print("Starting a new game!\n")
    #s = connect_to_server('127.0.0.1', 65432)
    #nc command : nc 156trlinux-1.ece.mcgill.ca 12345
    s = connect_to_server('156trlinux-1.ece.mcgill.ca', 12345 )
    time.sleep(2)
    send_initial_setup(s, f"game{game_number}", "white" if player == WHITE else 'black')

    board = create_board()

    game_over = False
    turn = WHITE
    opponent = BLACK if player == WHITE else WHITE
    #turn = random.randint(0,1)
    totTurn = 0

    # Game loop
    while not game_over:
        print(f"Turn number : {totTurn}")
        logging.info(f"Turn number : {totTurn}")

        if turn == player:
            move = None
            timeout = 9
            time_limit_exceeded = False
            # Use the minimax algorithm to find the best move for the AI.
            for depth in range(DEPTH_BEGIN, DEPTH):
                global node_visited
                results = [None]
                node_visited = 0
                thread = threading.Thread(target=minimax_with_timeout, args=(board, player, depth, results))
                thread.start()
                thread.join(timeout)  # Wait for the thread to complete, with a timeout
                
                if thread.is_alive():  # Check if the thread is still running
                    print("Time limit exceeded, sending random move.")
                    time_limit_exceeded = True
                    thread.join()  # Join the thread to clean up
                    move = random_move(board, player)  # Get a random move
                    print(f"Player {player} makes move: {move}")
                    logging.info(f"Player {player} makes move: {move}")
                    if move != None:
                        apply_move(board,move,player)
                        send_move(s,move)
                    apply_gravity(board)
                    break
                else:
                    move, minimax_score, elapsed_time = results[0]
                    print(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")
                    logging.info(f"Depth = {depth}, Time taken: {elapsed_time:.4f} seconds \n Nodes visited: {node_visited}")

            if not time_limit_exceeded:
                print(f"Player {player} makes move: {move}")
                logging.info(f"Player {player} makes move: {move}")
                if move != None:
                    apply_move(board,move,player)
                    send_move(s,move)
                apply_gravity(board)
            else:
                print("Random move chosen instead")

            flag, msg = receive_message(s)
            print(f"received echo of my own move : {msg}")

            # Check if Player has a winning move.
            result = check_win(board)
            if result == str(player):
                game_over = True
                display_board(board)
                print("\n")
                print(f"{player} player wins!")
                logging.info("\n")
                logging.info(f"{player} wins!")
            elif result == str(opponent):
                game_over = True
                display_board(board)
                print("\n")
                print(f"{opponent} has won!")
                logging.info("\n")
                logging.info(f"{opponent} has won!")
                break

            # Print the current game board and switch to the next turn.
            display_board(board)
            print("\n")
            logging.info("\n")
            turn += 1
            turn= turn % 2
            #if totTurn >= 3:
                #plot_avg_time_results(times[depth])

        if turn == opponent and not game_over:
            print("\n")
            logging.info("\n")

            # Receive data from the client
            echoed_move = s.recv(1024).decode()

            # Check if the received string is longer than 5 characters
            while len(echoed_move) > 8:
                print(f"Received move is too long, {echoed_move}.")
                return
            
            move = parse_move(echoed_move)
            
            #timesTurn[totTurn].append(elapsed_time)
            #nodesTurn[totTurn].append(node_visited)
            print(f"Player {opponent} makes move: {move}")
            logging.info(f"Player {opponent} makes move: {move}")

            if move != None:
                apply_move(board,move,opponent)
            apply_gravity(board)

            # Check if opponent has a winning move.
            result = check_win(board)
            if result == str(opponent):
                game_over = True
                display_board(board)
                print("\n")
                print(f"{opponent} has won!")
                logging.info("\n")
                logging.info(f"{opponent} has won!")
            elif result == str(player):
                game_over = True
                display_board(board)
                print("\n")
                print(f"{player} has won!")
                logging.info("\n")
                logging.info(f"{player} has won!")
                break

            # Print the current game board and switch to the next turn.
            display_board(board)
            print("\n")
            logging.info("\n")
            turn += 1
            turn= turn % 2
            totTurn +=1
    s.close()

def parse_move(echoed_string):
    print(echoed_string + "from parse move")
    parts = echoed_string.split()
    move_type = parts[0]

    if move_type == 'D':
        # If the move is a drop move
        column = int(parts[1])
        return ('D', column -1)
    elif move_type == 'L':
        # If the move is a slide move
        column = int(parts[1])
        row = int(parts[2])
        return ('L', column -1, row-1)
    elif move_type == 'R':
        # If the move is a slide move
        column = int(parts[1])
        row = int(parts[2])
        return ('R', column-1, row-1)
    else:
        return None

# Function to plot the average time results
def plot_avg_time_results():
        # After several moves, compute the average time for each depth
    avg_times = {depth: sum(times[depth]) / len(times[depth]) for depth in times}
    avg_nodes = {depth: sum(nodes[depth]) / len(nodes[depth]) for depth in nodes}

    # Plot time
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list(avg_times.keys()), list(avg_times.values()), marker='o')
    plt.title('Average Time vs Depth')
    plt.xlabel('Search Depth')
    plt.ylabel('Time (seconds)')

    # Plot nodes visited
    plt.subplot(1, 2, 2)
    plt.plot(list(avg_nodes.keys()), list(avg_nodes.values()), marker='o', color='orange')
    plt.title('Average Nodes Visited vs Depth')
    plt.xlabel('Search Depth')
    plt.ylabel('Nodes Visited')

    plt.tight_layout()
    plt.show()

# Function to plot the average time and nodes visited results per turn
'''def plot_avg_time_nodes_per_turn():

    turn_numbers = []
    turn_times = []
    turn_times_imp = []
    turn_nodes = []
    turn_nodes_imp = []

   # Iterate through the times and nodes dictionaries
    for turn in sorted(timesTurn.keys()):
        if all(time == 0 for time in timesTurn[turn]) and all(node == 0 for node in nodesTurn[turn]):
            break
        
        turn_numbers.append(turn)
        turn_times.append(sum(timesTurn[turn]))  # Assuming you want total time for that turn
        turn_times_imp.append(sum(timesTurnImproved[turn]))
        turn_nodes.append(sum(nodesTurn[turn]))  # Assuming you want total nodes for that turn
        turn_nodes_imp.append(sum(nodesTurnImproved[turn]))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Plotting time taken
    ax1.plot(turn_numbers, turn_times, marker='o', linestyle='-', color='blue', label='WHITE Time (Naive)')
    ax1.plot(turn_numbers, turn_times_imp, marker='x', linestyle='--', color='orange', label='BLACK Time (Improved)')
    
    ax1.set_xlabel('Turn Number')
    ax1.set_ylabel('Time Taken (seconds)', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Create a second y-axis for nodes searched
    ax2 = ax1.twinx()
    ax2.plot(turn_numbers, turn_nodes, marker='o', linestyle='-', color='green', label='WHITE Nodes (Naive)')
    ax2.plot(turn_numbers, turn_nodes_imp, marker='x', linestyle='--', color='red', label='BLACK Nodes (Improved)')
    
    ax2.set_ylabel('Nodes Visited', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title('Time Taken and Nodes Visited per Turn for Both Players')
    plt.show(block=True)  # Keep the plot window open'''


#play_game_local()
play_remote(WHITE,43)
