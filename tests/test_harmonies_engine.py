import unittest
from harmonies_engine import HarmoniesGameState, PILE_SIZE


class TestHarmoniesEngineStateChanges(unittest.TestCase):

    def setUp(self):
        """Set up a basic game state for tests."""
        self.initial_state = HarmoniesGameState()
        # Ensure initial state is somewhat predictable for easier testing
        # Let's force the initial piles if possible (modify __init__ or test specific state)
        # For now, we'll work with the random initial state
        self.initial_state_copy = self.initial_state.clone() # Keep original copy

    def test_apply_move_choose_pile_returns_new_object(self):
        """Verify apply_move returns a new object, not modifying in place."""
        if not self.initial_state.available_piles: self.skipTest("Initial state has no piles to choose")
        
        move = 0 # Choose the first pile
        new_state = self.initial_state.apply_move(move)
        
        self.assertIsNot(self.initial_state, new_state, "apply_move should return a new object.")
        # Also check original wasn't modified (comparing relevant attributes)
        self.assertEqual(self.initial_state.current_player, self.initial_state_copy.current_player)
        self.assertEqual(self.initial_state.turn_phase, self.initial_state_copy.turn_phase)
        self.assertEqual(self.initial_state.tiles_in_hand, self.initial_state_copy.tiles_in_hand)
        self.assertEqual(len(self.initial_state.available_piles), len(self.initial_state_copy.available_piles))


    def test_apply_move_choose_pile_updates_player_and_phase(self):
        """Check if player and phase change correctly after choosing a pile."""
        if not self.initial_state.available_piles: self.skipTest("Initial state has no piles to choose")
        
        move = 0 # Choose the first pile
        new_state = self.initial_state.apply_move(move)
        
        self.assertEqual(new_state.turn_phase, "place_tile_1", "Phase should change to place_tile_1.")
        # Check original state was not modified
        self.assertEqual(self.initial_state.current_player, 0, "Original state player should remain 0.")
        self.assertEqual(self.initial_state.turn_phase, "choose_pile", "Original state phase should remain choose_pile.")

    def test_apply_move_choose_pile_updates_piles_and_hand(self):
        """Check if piles are removed and hand is populated correctly."""
        if len(self.initial_state.available_piles) < 1: self.skipTest("Not enough piles for test")

        initial_pile_count = len(self.initial_state.available_piles)
        # Important: store the content of the chosen pile BEFORE the move
        chosen_pile_content = list(self.initial_state.available_piles[0]) 
        
        move = 0 # Choose the first pile
        new_state = self.initial_state.apply_move(move)

        self.assertEqual(len(new_state.available_piles), initial_pile_count - 1, "One pile should be removed.")
        self.assertEqual(len(new_state.tiles_in_hand), PILE_SIZE, f"Hand should contain {PILE_SIZE} tiles.")
        # Use assertCountEqual for lists where order might not be guaranteed or doesn't matter
        self.assertCountEqual(new_state.tiles_in_hand, chosen_pile_content, "Hand should contain tiles from the chosen pile.")
        # Check original state
        self.assertEqual(len(self.initial_state.available_piles), initial_pile_count, "Original state pile count shouldn't change.")
        self.assertEqual(len(self.initial_state.tiles_in_hand), 0, "Original state hand should remain empty.")

    def test_apply_move_choose_pile_does_not_change_bag_or_boards(self):
        """Check if bag and boards are unaffected by choosing a pile."""
        if not self.initial_state.available_piles: self.skipTest("Initial state has no piles to choose")

        initial_bag = self.initial_state.tile_bag.copy()
        initial_board0 = self.initial_state.player_boards[0].copy()
        initial_board1 = self.initial_state.player_boards[1].copy()

        move = 0
        new_state = self.initial_state.apply_move(move)

        self.assertEqual(new_state.tile_bag, initial_bag, "Tile bag should not change when only choosing a pile.")
        self.assertEqual(new_state.player_boards[0], initial_board0, "Player 0 board should not change.")
        self.assertEqual(new_state.player_boards[1], initial_board1, "Player 1 board should not change.")

class TestHarmoniesEngineHashing(unittest.TestCase):

    def setUp(self):
        # Check if __hash__ is implemented, skip if not
        if not hasattr(HarmoniesGameState, '__hash__') or HarmoniesGameState.__hash__ is object.__hash__:
             self.skipTest("Custom __hash__ method not implemented on HarmoniesGameState.")
             
    def test_identical_states_have_same_hash(self):
        """Test that two identical state objects produce the same hash."""
        state1 = HarmoniesGameState()
        # Need to make piles deterministic for reliable comparison
        state1.available_piles = [['a','b','c'], ['d','e','f'], ['g','h','i'], ['j','k','l'], ['m','n','o']] 
        state1.tile_bag = {'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1, 'g':1, 'h':1, 'i':1, 'j':1, 'k':1, 'l':1, 'm':1, 'n':1, 'o':1}
        
        state2 = state1.clone() # Create an identical copy

        self.assertEqual(state1, state2, "Cloned identical states should be equal.")
        self.assertEqual(hash(state1), hash(state2), "Cloned identical states should have the same hash.")

    def test_different_player_different_hash(self):
        """Test that changing only the current player results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [['a','b','c']] # Simplify
        state1.tile_bag = {'a':1, 'b':1, 'c':1}
        
        state2 = state1.clone()
        state2.current_player = 1 # Change only the player

        self.assertNotEqual(state1, state2, "States differing only by player should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by player should have different hashes.")

    def test_different_phase_different_hash(self):
        """Test that changing only the turn phase results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [['a','b','c']]
        state1.tile_bag = {'a':1, 'b':1, 'c':1}
        state1.turn_phase = "choose_pile"
        
        state2 = state1.clone()
        state2.turn_phase = "place_tile_1" # Change only the phase

        self.assertNotEqual(state1, state2, "States differing only by phase should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by phase should have different hashes.")

    def test_different_hand_different_hash(self):
        """Test that changing only the tiles in hand results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [] 
        state1.tile_bag = {'a':1, 'b':1, 'c':1}
        state1.turn_phase = "place_tile_1"
        state1.tiles_in_hand = ['a', 'b']
        
        state2 = state1.clone()
        state2.tiles_in_hand = ['a', 'c'] # Change hand content

        self.assertNotEqual(state1, state2, "States differing only by hand should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by hand should have different hashes.")

    def test_different_piles_different_hash(self):
        """Test that changing only the available piles results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [['a','b','c'], ['d','e','f']]
        state1.tile_bag = {'a':1, 'b':1, 'c':1, 'd':1, 'e':1, 'f':1}
        
        state2 = state1.clone()
        state2.available_piles = [['a','b','c'], ['X','Y','Z']] # Change one pile

        self.assertNotEqual(state1, state2, "States differing only by piles should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by piles should have different hashes.")
        
    def test_different_bag_different_hash(self):
        """Test that changing only the tile bag results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [['a','b','c']]
        state1.tile_bag = {'a':2, 'b':1, 'c':1}
        
        state2 = state1.clone()
        state2.tile_bag = {'a':1, 'b':1, 'c':1} # Change bag count

        self.assertNotEqual(state1, state2, "States differing only by bag should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by bag should have different hashes.")

    def test_different_board_different_hash(self):
        """Test that changing only a player board results in a different hash."""
        state1 = HarmoniesGameState()
        state1.available_piles = [['a','b','c']]
        state1.tile_bag = {'a':1, 'b':1, 'c':1}
        state1.player_boards[0] = { (0,0): ['a'] } # Give P0 a tile
        
        state2 = state1.clone()
        state2.player_boards[0] = { (1,0): ['a'] } # Change board content

        self.assertNotEqual(state1, state2, "States differing only by board should not be equal.")
        self.assertNotEqual(hash(state1), hash(state2), "States differing only by board should have different hashes.")


# Boilerplate to run tests
if __name__ == '__main__':
    # First, remind user to implement __hash__ and __eq__ if they haven't
    if not hasattr(HarmoniesGameState, '__hash__') or HarmoniesGameState.__hash__ is object.__hash__:
         print("="*70)
         print("WARNING: Custom __hash__ and __eq__ methods are recommended for HarmoniesGameState.")
         print("Add them using the get_canonical_tuple() approach before running hashing tests.")
         print("Skipping Hashing Tests.")
         print("="*70)
         # Run only the state change tests
         suite = unittest.TestSuite()
         suite.addTest(unittest.makeSuite(TestHarmoniesEngineStateChanges))
         runner = unittest.TextTestRunner()
         runner.run(suite)
    else:
        # Run all tests
        unittest.main()