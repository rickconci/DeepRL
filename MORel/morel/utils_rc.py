SEED = 42


Mazes = {
    'PointMaze_UMaze-v3': [[1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]


}



def find_zero_positions(maze):
    """
    Find the positions of zeros in a maze.

    Parameters:
    maze (list of lists): The maze to search.

    Returns:
    list of tuples: Positions of zeros in the maze.
    """
    zero_positions = [(i, j) for i, row in enumerate(maze) for j, value in enumerate(row) if value == 0]
    return zero_positions
