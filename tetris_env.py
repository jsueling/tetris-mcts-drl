"""
Tetris environment for reinforcement learning agents.
Adapted from: https://github.com/corentinpla/Learning-Tetris-Using-the-Noisy-Cross-Entropy-Method
"""

import random
from typing import Optional

import numpy as np

class Tetromino:
    """Represents a Tetromino piece with its position, type, and rotation."""

    figures = [
        # Each figure is represented by a flattened 4x4 dimension
        # with a list of each 4 rotations it can take
        [[0, 4, 8, 12], [0, 1, 2, 3]],                            # I
        [[0, 1, 5, 6], [1, 4, 5, 8]],                             # Z
        [[4, 5, 1, 2], [0, 4, 5, 9]],                             # S
        [[1, 0, 4, 8], [0, 4, 5, 6], [1, 5, 9, 8], [0, 1, 2, 6]], # J
        [[0, 1, 5, 9], [4, 0, 1, 2], [0, 4, 8, 9], [4, 5, 6, 2]], # L
        [[1, 4, 5, 6], [1, 4, 5, 9], [0, 1, 2, 5], [0, 4, 8, 5]], # T
        [[0, 1, 4, 5]],                                           # O
    ]

    default_spawns = [
        # Default spawns are horizontally centered along the top row
        # format: column, rotation
        [3, 1],  # I
        [3, 0],  # Z
        [3, 0],  # S
        [3, 3],  # J
        [3, 1],  # L
        [3, 2],  # T
        [4, 0],  # O
    ]

    # Each Tetromino's position is represented by
    # The following 4x4 grid:

    #  0  1  2  3
    #  4  5  6  7
    #  8  9 10 11
    # 12 13 14 15

    # A Tetromino's (x, y) coordinates reference the position
    # of where cell 0 in the 4x4 grid is located in the 10x20 grid.

    def __init__(self, tetromino_type: int):
        # Instantiated with no position or rotation
        self.x = None
        self.y = None
        self.type = tetromino_type # type of the Tetromino (0-6)
        self.rotation = None

    def spawn(self, x: int, rotation: int):
        """
        Spawns the Tetromino in the grid at row 0 and the specified column (x)
        and rotation.
        """
        self.x = x
        self.y = 0
        self.rotation = rotation # rotation of the Tetromino (0-3)

    def despawn(self):
        """Despawn the Tetromino by resetting its position and rotation."""
        self.x = None
        self.y = None
        self.rotation = None

    def image(self):
        """Returns the current 4x4 image of the Tetromino based on its type and rotation."""
        return self.figures[self.type][self.rotation]

class Tetris:
    """
    Represents the Tetris game state,
    including the grid, current tetromino, score, and game state.
    """
    def __init__(
        self,
        height: int=20,
        width: int=10,
        tetromino_randomisation_scheme: str="uniform"
    ):

        self.current_tetromino = None
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=np.float32)
        self.score = 0
        self.done = False

        # scheme can be type "uniform" or "bag"
        self.tetromino_randomisation_scheme = tetromino_randomisation_scheme

        if self.tetromino_randomisation_scheme == "bag":
            # Initialise a bag of random Tetrominoes
            self.bag = list(range(len(Tetromino.figures)))
            random.shuffle(self.bag)

    def __repr__(self):
        grid_repr = []
        for row in range(self.height):
            grid_repr.append("".join(str(self.grid[row][col]) for col in range(self.width)))
        return "\n".join(grid_repr)

    def create_tetromino(self, tetromino_type: int) -> None:
        """Creates a new unspawned Tetromino of tetromino_type attached to this game state."""
        self.current_tetromino = Tetromino(tetromino_type)

    def get_current_tetromino_type(self) -> Optional[int]:
        """
        Returns the type of the current Tetromino.
        If no Tetromino is currently active, returns None.
        """
        if self.current_tetromino:
            return self.current_tetromino.type
        return None

    def generate_next_tetromino_type(self, is_first_tetromino=False) -> int:
        """
        Generates the next Tetromino type based on the randomisation scheme.
        For "uniform", it returns a random Tetromino type.
        For "bag", it returns a Tetromino type from a bag containing each
        Tetromino in a random order, refilling and reshuffling it when empty.
        """

        if self.tetromino_randomisation_scheme == "uniform":
            # Randomly select a Tetromino type uniformly
            return random.randint(0, 6)

        if self.tetromino_randomisation_scheme == "bag":

            # If this is the first Tetromino of the episode
            # or the bag is empty, refill and shuffle it
            if is_first_tetromino or not self.bag:
                self.bag = list(range(len(Tetromino.figures)))
                random.shuffle(self.bag)
            return self.bag.pop()

        raise ValueError(
            f"Invalid tetromino randomisation scheme: {self.tetromino_randomisation_scheme}"
        )

    def intersects(self):
        """
        Returns True if the current Tetromino placement is invalid or False otherwise
        (OOB and collision checks).
        """
        x, y = self.current_tetromino.x, self.current_tetromino.y
        for cell_index in self.current_tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            if ( # OOB checks and collision check
                tetromino_row < 0 or
                tetromino_row >= self.height or
                tetromino_col >= self.width or
                tetromino_col < 0 or
                self.grid[tetromino_row][tetromino_col] > 0
            ):
                return True
        return False

    def break_lines(self):
        """Break lines that are completely filled by Tetrominoes."""

        filled_lines = np.all(self.grid > 0, axis=1)
        # All filled lines now become broken lines
        broken_lines = np.count_nonzero(filled_lines)
        if broken_lines > 0:
            self.grid = np.vstack((
                # Add empty rows at the top to replace the broken lines
                np.zeros((broken_lines, self.width), dtype=np.float32),
                # Keep only rows that were not filled (maintains ordering)
                self.grid[~filled_lines]
            ))
            self.score += broken_lines

    def hard_drop(self, colour=1):
        """
        Move the current Tetromino directly down to the bottom of the grid.
        After the hard drop, the Tetromino is removed from the game state.
        """
        while not self.intersects():
            self.current_tetromino.y += 1
        self.current_tetromino.y -= 1
        self.freeze(colour)
        self.current_tetromino = None

    def freeze(self, colour):
        """Freeze the current Tetromino, it now becomes part of the grid."""
        x, y = self.current_tetromino.x, self.current_tetromino.y
        for cell_index in self.current_tetromino.image():
            tetromino_row = y + (cell_index // 4)
            tetromino_col = x + (cell_index % 4)
            self.grid[tetromino_row][tetromino_col] = colour
        self.break_lines()

    def path_exists_to_col(self, target_column):
        """
        Check if there is a valid path across the top row to the desired column
        for the Tetromino hard drop placement. This assumes that
        all shifts and rotations are possible during lock delay.
        """
        current_column = self.current_tetromino.x
        if target_column < current_column:
            for col in range(current_column, target_column - 1, -1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        elif target_column > current_column:
            for col in range(current_column, target_column + 1):
                if self.grid[0][col] > 0 or self.grid[1][col] > 0:
                    return False
        return True

    def get_legal_actions(self):
        """
        Returns a list of legal actions for the current Tetromino.
        Each action is represented as 2 digits in base 10:
        - The first digit is the rotation (0-3).
        - The second digit is the column (0-9).

        Legal actions here are defined using the simplified version of Tetris
        which assumes that the Tetromino can be spawned in any column.
        Redundant rotations are also masked to avoid unnecessary actions.

        Returns:
            np.array: A boolean array of shape (40,) where each index corresponds
            to a possible action (rotation * width + column).
            True indicates that the action is legal, False indicates it is illegal.
        """

        max_rotations = 4
        max_columns = self.width

        # All actions are initially illegal. Actions are represented
        # as indices in base max_columns as rotation * max_columns + column
        # where rotation is in [0, 3] and column is in [0, 9]
        legal_actions = np.array([False] * (max_columns * max_rotations))

        # Get the number of unique rotations for the current Tetromino
        # since Tetrominoes have symmetric rotations.
        unique_rotations = len(Tetromino.figures[self.current_tetromino.type])

        for rotation in range(unique_rotations):
            for col in range(max_columns):
                # Try to spawn the Tetromino in the specified column and rotation
                self.current_tetromino.spawn(x=col, rotation=rotation)
                if not self.intersects():
                    legal_actions[rotation * max_columns + col] = True

        # Despawn the Tetromino after checking all legal actions
        # to ensure the state is clean for the next action
        self.current_tetromino.despawn()

        return legal_actions

    def step(self, action, colour=1):
        """
        Perform a step in the game by applying the given action.

        Each action is represented as 2 digits in base 10:
        - The first digit is the rotation (0-3).
        - The second digit is the column (0-9).
        """
        rotation, col = divmod(action, self.width)
        self.current_tetromino.spawn(x=col, rotation=rotation)
        if self.intersects():
            self.done = True
        else:
            # Current Tetromino is placed and removed from the game state
            self.hard_drop(colour)
        return self.done, self.score

    def reset(self):
        """
        Reset the game state to the initial conditions.
        """
        self.grid.fill(0)
        self.score = 0
        self.done = False
        # Generate the first Tetromino of the new episode
        self.create_tetromino(
            self.generate_next_tetromino_type(is_first_tetromino=True)
        )

    def copy(self):
        """Create a deep copy of the current game state."""
        new_env = Tetris(self.height, self.width, self.tetromino_randomisation_scheme)
        np.copyto(new_env.grid, self.grid)
        new_env.score = self.score
        new_env.done = self.done
        if self.current_tetromino:
            new_env.create_tetromino(self.get_current_tetromino_type())
        if self.tetromino_randomisation_scheme == "bag":
            new_env.bag = self.bag.copy()
        return new_env

    def get_state(self):
        """
        Returns a representation of the current game state used by the neural network.
        The state is a tuple containing:
        - The current grid representation as a 4D tensor with shape (1, height, width)
        - The current Tetromino type as a one-hot encoded vector.
        """
        grid_copy = np.zeros((self.height, self.width), dtype=np.float32)
        np.copyto(grid_copy, self.grid)
        tetromino_type = self.get_current_tetromino_type()
        tetromino_one_hot = np.zeros((len(Tetromino.figures),), dtype=np.float32)
        tetromino_one_hot[tetromino_type] = 1.0

        return np.expand_dims(grid_copy, axis=0), tetromino_one_hot
