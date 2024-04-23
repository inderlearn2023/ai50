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
        if (board[i][0] != EMPTY) and (board[i][0] == board[i][1] == board[i][2]):
            return board[i][0]
        # check column
        if (board[0][i] != EMPTY) and (board[0][i] == board[1][i] == board[2][i]):
            return board[0][i]

    # check diagonals
    if (board[1][1] != EMPTY) and ((board[0][0] == board[1][1] == board[2][2]) or (board[0][2] == board[1][1] == board[2][0])):
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

    val, move = min_value(board) if player(board) == X else max_value(board)
    return move


def max_value(board, alpha=-math.inf, beta=math.inf):
    if terminal(board):
        return utility(board), None

    val = -math.inf
    move = None
    for action in actions(board):
        # Calculate the value of the resulting board
        result_value, _ = min_value(result(board, action), alpha, beta)

        # Update alpha if necessary
        if result_value > val:
            val = result_value
            move = action
            if val == 1:  # Found winning move
                return val, move
            alpha = max(alpha, val)
            if alpha >= beta:  # Beta cutoff
                break

    return val, move

def min_value(board, alpha=-math.inf, beta=math.inf):
    if terminal(board):
        return utility(board), None

    val = math.inf
    move = None
    for action in actions(board):
        # Calculate the value of the resulting board
        result_value, _ = max_value(result(board, action), alpha, beta)

        # Update beta if necessary
        if result_value < val:
            val = result_value
            move = action
            if val == -1:  # Found winning move
                return val, move
            beta = min(beta, val)
            if alpha >= beta:  # Alpha cutoff
                break

    return val, move
