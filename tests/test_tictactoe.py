from tic_tac_toe_minimax.main import TicTacToeAgent


def test_tictactoe_winner_and_terminal_logic():
    agent = TicTacToeAgent()

    state = [
        ["x", "x", "x"],
        ["o", ".", "o"],
        [".", ".", "."],
    ]

    assert agent.winner(state) == "x"
    assert agent.terminal(state)


def test_tictactoe_draw_terminal():
    agent = TicTacToeAgent()

    draw_state = [
        ["x", "o", "x"],
        ["x", "o", "o"],
        ["o", "x", "x"],
    ]

    assert agent.winner(draw_state) is None
    assert agent.terminal(draw_state)
