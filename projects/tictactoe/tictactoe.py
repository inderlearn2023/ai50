"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = 0
    o_count = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == 'X':
                x_count += 1
            elif board[i][j] == 'O':
                o_count += 1

    if x_count > o_count:
        return 'O'

    return 'X'

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))

    return possible_actions

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """

    if action[0] < 0 or action[1] > 9:
        raise IndexError("out of bounds")

    # copy the board to update
    updated_board = copy.deepcopy(board)

    # if the action slot is available, then let the player play their move
    if updated_board[action[0]][action[1]] == EMPTY:
        updated_board[action[0]][action[1]] = player(board)

    return updated_board

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        # check the row
        if (board[i][0] in (X, O)) and (board[i][0] == board[i][1] == board[i][2]):
            return board[i][0]
        # check column
        if (board[0][i] in (X, O)) and (board[0][i] == board[1][i] == board[2][i]):
            return board[0][i]

    # check diagonals
    if (board[1][1] in (X, O)) and ((board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0])):
        return board[1][1]

    # no winner
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    result = utility(board)

    return result != 0 or len(actions(board)) == 0

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """

    result = winner(board)

    if result == 'X':
        return 1

    if result == 'O':
        return -1

    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    moves = []

    p = player(board)

    for action in actions(board):
        score = min_value(result(board, action)) if p == X else max_value(result(board, action))
        # Store options in list
        moves.append([score, action])
        # Return highest value action
    return sorted(moves, reverse=(p == X))[0][1]

def max_value(board):
    # Check for terminal state
    if terminal(board):
        return utility(board)

    # Loop through all possible actions
    value = -math.inf
    for action in actions(board):
        value = max(value, min_value(result(board, action)))
    return value

def min_value(board):
    # Check for terminal state
    if terminal(board):
        return utility(board)

    # Loop through all possible actions
    value = math.inf
    for action in actions(board):
        value = min(value, max_value(result(board, action)))
    return value
