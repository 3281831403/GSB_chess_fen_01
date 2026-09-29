'''Tests for the candidate-move check optimization in rlm.py.

Covers single/multi-step candidate checks, long search sequences, illegal
moves, state before/after simulated checks, and multiple game instances.
Existing tests in rlm.TestRLM are also exercised here (unmodified).
'''
import copy

import pytest

import rlm
from rlm import Board, Game, King, Queen, Rook, Bishop, Knight, Pawn, Move


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def standardize_square(square):
    '''Normalize a square representation (None / 'e4' / (4, 3)) to a tuple.'''
    if square is None:
        return None
    if isinstance(square, str):
        return tuple(Board.square_name_to_array_idxs(square))
    return tuple(square)


def canonical_state(game):
    '''A deterministic, field-by-field snapshot of everything make_move can touch.'''
    return {
        'board': tuple(game.board.board_array.ravel().tolist()),
        'ep_square': standardize_square(game.ep_square),
        'castling_state': tuple(game.castling_state),
        'side_to_move': game.side_to_move,
        'move_counter': game.move_counter,
        'half_moves_since': game.half_moves_since,
        'white_pieces': sorted((p.char, standardize_square(p.current_square))
                               for p in game.white_pieces),
        'black_pieces': sorted((p.char, standardize_square(p.current_square))
                               for p in game.black_pieces),
        'move_history': tuple(m.to_long_algebraic() for m in game.move_history),
    }


def reference_is_in_check_after_move(king, move):
    '''The original copy-the-whole-game algorithm, kept as ground truth.'''
    temp_game = king.game.copy()
    temp_game.make_move(move)
    other_side_moves = temp_game.get_moves_for(allow_own_king_checked=True)
    return any(temp_game.board[m.destination_square] == king.char
               for m in other_side_moves)


def king_for_side_to_move(game):
    color = 'w' if game.side_to_move == 'w' else 'b'
    square = game.board.find_king_square(color)
    return King(color, square, game)


def make_game(piece_list, side='w', ep_square=None):
    game = Game()
    game.set_board(Board(piece_list=piece_list))
    if side == 'w':
        game.set_white_to_move()
    else:
        game.set_black_to_move()
    game.ep_square = ep_square
    return game


POSITIONS = {
    'startpos': lambda: make_game_from_board(Board(), 'w'),
    'kings_only': lambda: make_game([King('w', 'e1'), King('b', 'e8')], 'w'),
    'castling': lambda: make_game(
        [King('w', 'e1'), Rook('w', 'a1'), Rook('w', 'h1'),
         King('b', 'e8'), Rook('b', 'a8'), Rook('b', 'h8')], 'w'),
    'pawns_ep': lambda: make_game(
        [King('w', 'e1'), King('b', 'h3'), Pawn('w', 'b5'), Pawn('b', 'c5')],
        'w', ep_square='c6'),
    'promotion': lambda: make_game(
        [King('w', 'e1'), King('b', 'e8'), Pawn('w', 'f7'), Rook('b', 'g8')], 'w'),
    'stalemate': lambda: make_game(
        [King('w', 'b6'), Queen('w', 'c7'), King('b', 'a8')], 'b'),
    'single_move': lambda: make_game(
        [King('w', 'f6'), Rook('w', 'g1'), King('b', 'h8')], 'b'),
}


def make_game_from_board(board, side):
    game = Game()
    game.set_board(board)
    if side == 'w':
        game.set_white_to_move()
    else:
        game.set_black_to_move()
    return game


# ---------------------------------------------------------------------------
# Existing test suite (unmodified) still passes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('test_method', [
    'test_board_generation_from_piece_list_1',
    'test_move_generation_1',
    'test_move_generation_2',
    'test_pawn_moves_1',
])
def test_existing_rlm_tests_still_pass(test_method):
    assert getattr(rlm.TestRLM(), test_method)() is True


# ---------------------------------------------------------------------------
# Candidate checks must not copy the whole game per candidate
# ---------------------------------------------------------------------------

def test_candidate_checks_do_not_copy_game_or_board(monkeypatch):
    game = POSITIONS['startpos']()
    copy_calls = []
    orig_game_copy = Game.copy
    orig_board_copy = Board.copy

    def spy_game_copy(self):
        copy_calls.append('game')
        return orig_game_copy(self)

    def spy_board_copy(self):
        copy_calls.append('board')
        return orig_board_copy(self)

    monkeypatch.setattr(Game, 'copy', spy_game_copy)
    monkeypatch.setattr(Board, 'copy', spy_board_copy)
    moves = game.get_moves_for()
    assert len(moves) == 20
    assert copy_calls == []


def test_candidate_checks_survive_copy_resource_failure(monkeypatch):
    '''Simulated checks must not depend on whole-game copying, so a resource
    failure inside Game.copy cannot break legal move generation.'''
    def boom(self):
        raise MemoryError('simulated resource exhaustion')

    monkeypatch.setattr(Game, 'copy', boom)
    game = POSITIONS['startpos']()
    moves = game.get_moves_for()
    assert len(moves) == 20


# ---------------------------------------------------------------------------
# Single candidate: result parity with reference and zero state change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('position_name', sorted(POSITIONS))
def test_single_candidate_matches_reference_and_leaves_state_unchanged(position_name):
    game = POSITIONS[position_name]()
    before = copy.deepcopy(game)
    king = king_for_side_to_move(game)
    candidates = game.get_moves_for()
    for move in candidates:
        expected = reference_is_in_check_after_move(king, move)
        actual = king.is_in_check_after_move(move)
        assert actual == expected
        assert canonical_state(game) == canonical_state(before)


# ---------------------------------------------------------------------------
# Multi-step candidates: full legal move lists match reference filtering
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('position_name,expected_count', [
    ('startpos', 20),
    ('kings_only', 5),
    ('castling', 26),
    ('stalemate', 0),      # empty candidate set
    ('single_move', 1),    # single-element candidate set
])
def test_legal_move_counts_and_contents(position_name, expected_count):
    game = POSITIONS[position_name]()
    moves = game.get_moves_for()
    assert len(moves) == expected_count
    # Every surviving move must be one the reference check also clears
    king = king_for_side_to_move(game)
    for move in moves:
        assert reference_is_in_check_after_move(king, move) is False


def test_multi_step_candidates_each_leave_state_unchanged():
    game = POSITIONS['castling']()
    before = copy.deepcopy(game)
    king = king_for_side_to_move(game)
    provisional = [m for m in game.get_moves_for()]
    assert len(provisional) > 1
    for move in provisional:
        king.is_in_check_after_move(move)
        assert canonical_state(game) == canonical_state(before)


# ---------------------------------------------------------------------------
# Illegal / rejected moves: rollback, then success unpolluted
# ---------------------------------------------------------------------------

def test_illegal_move_empty_source_raises_and_rolls_back():
    game = POSITIONS['startpos']()
    before = copy.deepcopy(game)
    king = king_for_side_to_move(game)
    illegal = Move('P', 'a3', 'a4')  # a3 is empty
    with pytest.raises(Exception):
        king.is_in_check_after_move(illegal)
    assert canonical_state(game) == canonical_state(before)
    # A failure must not pollute subsequent successful calls
    fresh = POSITIONS['startpos']()
    assert ([m.to_long_algebraic() for m in game.get_moves_for()]
            == [m.to_long_algebraic() for m in fresh.get_moves_for()])


def test_out_of_range_destination_rolls_back_completely():
    game = POSITIONS['startpos']()
    before = copy.deepcopy(game)
    king = king_for_side_to_move(game)
    out_of_range = Move('P', 'a2', (0, 9))  # rank 9 is off the board
    try:
        king.is_in_check_after_move(out_of_range)
    except Exception:
        pass  # raising is acceptable; corrupting state is not
    assert canonical_state(game) == canonical_state(before)
    assert len(game.get_moves_for()) == 20


def test_repeated_calls_are_deterministic_and_do_not_grow_state():
    game = POSITIONS['startpos']()
    king = king_for_side_to_move(game)
    move = Move('P', 'e2', 'e4')
    first = king.is_in_check_after_move(move)
    history_len = len(game.move_history)
    dict_keys = set(game.__dict__)
    for _ in range(50):
        assert king.is_in_check_after_move(move) == first
    assert len(game.move_history) == history_len
    assert set(game.__dict__) == dict_keys


# ---------------------------------------------------------------------------
# Long search sequences: parity, determinism, consecutive rounds
# ---------------------------------------------------------------------------

def play_sequence(seed, plies):
    import random
    rng = random.Random(seed)
    game = POSITIONS['startpos']()
    transcript = []
    for _ in range(plies):
        moves = game.get_moves_for()
        if not moves:
            break
        move_strs = sorted(m.to_long_algebraic() for m in moves)
        transcript.append(tuple(move_strs))
        game.make_move(moves[rng.randrange(len(moves))])
    return transcript, game


def test_long_search_sequence_matches_reference_every_ply():
    import random
    rng = random.Random(11)
    game = POSITIONS['startpos']()
    for _ in range(40):
        before = copy.deepcopy(game)
        king = king_for_side_to_move(game)
        candidates = game.get_moves_for()
        if not candidates:
            break
        for move in candidates:
            assert (king.is_in_check_after_move(move)
                    == reference_is_in_check_after_move(king, move))
        assert canonical_state(game) == canonical_state(before)
        game.make_move(candidates[rng.randrange(len(candidates))])


def test_same_input_sequence_replays_identically():
    transcript_1, game_1 = play_sequence(seed=7, plies=30)
    transcript_2, game_2 = play_sequence(seed=7, plies=30)
    assert transcript_1 == transcript_2
    assert canonical_state(game_1) == canonical_state(game_2)


def test_consecutive_rounds_track_state_correctly():
    game = POSITIONS['startpos']()
    for expected_ply in range(1, 6):
        moves = game.get_moves_for()
        assert len(moves) > 0
        game.make_move(moves[0])
        assert len(game.move_history) == expected_ply
        assert game.side_to_move == ('b' if expected_ply % 2 == 1 else 'w')


# ---------------------------------------------------------------------------
# Multiple game instances stay independent
# ---------------------------------------------------------------------------

def test_multiple_game_instances_are_independent():
    game_a = POSITIONS['startpos']()
    game_b = POSITIONS['castling']()
    snapshot_a = copy.deepcopy(game_a)
    snapshot_b = copy.deepcopy(game_b)
    # Interleave simulated checks and real moves across instances
    king_b = king_for_side_to_move(game_b)
    for move in game_b.get_moves_for():
        king_b.is_in_check_after_move(move)
    game_a.make_move(game_a.get_moves_for()[0])
    king_a = king_for_side_to_move(game_a)
    for move in game_a.get_moves_for():
        king_a.is_in_check_after_move(move)
    # game_b must be exactly as it was; game_a differs only by its real move
    assert canonical_state(game_b) == canonical_state(snapshot_b)
    after_a = canonical_state(game_a)
    assert after_a['move_history'] != canonical_state(snapshot_a)['move_history']
    assert len(after_a['move_history']) == 1


# ---------------------------------------------------------------------------
# Deep-copy snapshot equality after rejected calls (field by field)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('position_name', ['startpos', 'castling', 'pawns_ep', 'promotion'])
def test_deepcopy_snapshot_field_equality_after_simulations(position_name):
    game = POSITIONS[position_name]()
    snapshot = copy.deepcopy(game)
    king = king_for_side_to_move(game)
    for move in game.get_moves_for():
        king.is_in_check_after_move(move)
    before_fields = canonical_state(snapshot)
    after_fields = canonical_state(game)
    for field in before_fields:
        assert after_fields[field] == before_fields[field], field
