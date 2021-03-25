# -*- coding: utf-8 -*-
"""This module is the main module.
"""
import logging
import pathlib
import shutil
import argparse
import random
import coloredlogs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

################################################################################
# Constant definitions
################################################################################

# Logging
LOG = logging.getLogger(__name__)

# File/directory locations
PROJ_DIR = pathlib.Path(__file__).parent.absolute()
IMAGE_DIR = PROJ_DIR.joinpath('img')
IMAGE_DIR.mkdir(mode=0o775, exist_ok=True)
GIF_TEMP_DIR = IMAGE_DIR.joinpath('gif_frames')

# Simulation constants
MAX_SEARCHES = 100  # The parameter Q, used in the random policy
RED = np.array([1, 0, 0])
BLUE = np.array([0, 0, 1])
EMPTY = np.array([0, 0, 0])

################################################################################
# Simulation functions
################################################################################


def main(
    grid_size: int,
    min_neighbors: int,
    num_agents: int,
    max_epochs: int,
    iterations: int,
) -> None:
    """Main function

    Parameters
    ---------
    grid_size : int
        Size of the environment grid (the length)
    min_neighbors : int
        Minimum neighbors of the same type to be happy
    num_agents : int
        Number of agents to populate the grid
    max_epochs : int
        Maximum epochs for each iteration
        One epoch is one time through the population of agents
    iterations : int
        Number of iterations to run the simulation
    """
    LOG.debug("Running simulation with parameters:\n"
              f"grid_size (L) = {grid_size},\n"
              f"min_neighbors (k) = {min_neighbors},\n"
              f"num_agents (N) = {num_agents},\n"
              f"max_epochs={max_epochs}")
    for i in range(iterations):
        LOG.info(f"Begin iteration {i+1}")
        shutil.rmtree(GIF_TEMP_DIR, ignore_errors=True)  # Remove old steps
        GIF_TEMP_DIR.mkdir(mode=0o775, exist_ok=True)
        env = init_env(grid_size, num_agents)
        step = 0
        save_env(
            env,
            f"Random Policy\nepoch 0",
            GIF_TEMP_DIR.joinpath(f"step_{step:05d}.png"),
        )
        for epoch in range(max_epochs):
            LOG.info(f"Begin epoch {epoch+1}")
            agent_cells = np.concatenate(
                (get_matching_coords(env, grid_size, RED),
                 get_matching_coords(env, grid_size, BLUE)))
            np.random.shuffle(agent_cells)
            for agent in agent_cells:
                if random_policy(env, grid_size, min_neighbors, tuple(agent)):
                    step += 1
                    save_env(
                        env,
                        f"Random Policy - epoch {epoch}\nstep {step:06d}",
                        GIF_TEMP_DIR.joinpath(f"step_{step:06d}.png"),
                    )
        save_gif(
            GIF_TEMP_DIR,
            IMAGE_DIR.joinpath(f"random_{iterations:02d}.gif"),
        )


def random_policy(env: np.ndarray, grid_size: int, min_neighbors: int,
                  agent_coord: tuple[int]) -> bool:
    """Randomly choose a tile that makes the agent happy

    If the agent cannot be happy in MAX_SEARCHES, it chooses the one it
    saw which has the most neighbors of the same type

    Parameters
    ----------
    env : np.ndarray
        The environment grid which contains agents
    grid_size : int
        Size of the environment grid (the length)
    min_neighbors : int
        Minimum neighbors of the same type to be happy
    agent_coord : np.ndarray
        The coordinates of the agent to evaluate in the form [x, y]

    Returns
    -------
    bool
        True if the agent moved, false if not
    """
    agent_type = env[agent_coord]
    # Check if the agent is happy first
    happy, _ = happiness(env, grid_size, min_neighbors, agent_coord,
                         agent_type)
    if happy:
        return False
    # Get randomized list of empty cells
    empty_cells = get_matching_coords(env, grid_size, EMPTY)
    np.random.shuffle(empty_cells)
    best_cell = tuple(empty_cells[0])
    _, best_cell_matching_neighbors = happiness(env, grid_size, min_neighbors,
                                                agent_coord, agent_type)
    for i, cell in enumerate(empty_cells):
        cell = tuple(cell)
        happy, matching_neighbors = happiness(env, grid_size, min_neighbors,
                                              agent_coord, agent_type)
        if matching_neighbors > best_cell_matching_neighbors:
            best_cell = cell
            best_cell_matching_neighbors = matching_neighbors
        if happy:
            break
        if i >= MAX_SEARCHES:
            break
    swap_cells(env, agent_coord, best_cell)
    return True


def happiness(
    env: np.ndarray,
    grid_size: int,
    min_neighbors: int,
    agent_coord: tuple[int],
    agent_type: np.ndarray,
) -> tuple[bool, int]:
    """Check if the given coordinate is happy

    Check all 8 neighbors (adjacent + diagonals) and see if number of
    alike neighbors is at least min_neighbors
    """
    agent_type = env[agent_coord]
    matching_neighbors = 0
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            if (env[(agent_coord[0] + i) % grid_size,
                    (agent_coord[1] + j) % grid_size] == agent_type).all():
                matching_neighbors += 1
    return ((matching_neighbors >= min_neighbors), matching_neighbors)


def swap_cells(env: np.ndarray, coord1: tuple[int],
               coord2: tuple[int]) -> None:
    """Swap 2 cells in place in env

    Parameters
    ----------
    env : np.ndarray
        The environment grid which contains agents
    coord1, coord2 : np.ndarray
        The coordinates of the cells to swap in the form [x, y]
    """
    temp_cell = np.copy(env[coord1])
    env[coord1] = np.copy(env[coord2])
    env[coord2] = temp_cell


def get_matching_coords(env: np.ndarray, grid_size: int,
                        agent_type: np.ndarray) -> np.ndarray:
    """Get a list of coordinates matching agent_type

    Parameters
    ----------
    env : np.ndarray
        The environment grid which contains agents
    grid_size : int
        The length of the env grid
    agent_type : np.ndarray
        The agent type, either RED, BLUE, or EMPTY
        These are in the form of RGB values and have length 3

    Returns
    -------
    np.ndarray
        Array of coordinates of cells containing agent_type
        In the form [(x0, y0), (x1, y1), ...]
    """
    return np.array([(i, j) for i in range(grid_size) for j in range(grid_size)
                     if (env[i, j] == agent_type).all()])


def init_env(grid_size: int, num_agents: int) -> np.ndarray:
    """Return the initialized env"""
    env = np.zeros((grid_size, grid_size, 3))
    coords = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(coords)
    for i, coord in enumerate(coords):
        if i < (num_agents // 2):
            env[coord] = BLUE
        elif i < num_agents:
            env[coord] = RED
        else:
            env[coord] = EMPTY
    return env


def save_env(
    env: np.ndarray,
    title: str,
    path: pathlib.Path,
) -> None:
    """Save a single frame image

    Parameters
    ----------
    TODO
    """
    fig, axis = plt.subplots()
    axis.imshow(env)
    axis.axis("off")
    axis.set_title(title)
    fig.savefig(path)
    fig.clf()
    plt.close()


def save_gif(img_dir: pathlib.Path, gif_path: pathlib.Path):
    """Save the images into a gif"""
    LOG.info(f"Generating gif {gif_path} from {img_dir}")
    images = []
    for f in sorted(list(img_dir.iterdir())):
        image = Image.open(f)
        images.append(image.copy())
        image.close()
        f.unlink()
    img_dir.rmdir()
    images[0].save(
        gif_path,
        save_all=True,
        duration=25,
        append_images=images[1:],
    )
    LOG.info("Done saving gif")


################################################################################
# CLI handler functions
################################################################################


def parse_args(arg_list: list[str] = None) -> argparse.Namespace:
    """Parse the arguments

    Parameters
    ----------
    arg_list : list[str]

    Returns
    -------
    argparse.Namespace
        A namespace of parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="A penguin swarm simulator",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        help="""set the logging level:
        1 = DEBUG
        2 = INFO
        3 = WARNING
        4 = ERROR
        5 = CRITICAL""",
        type=int,
        choices=range(1, 6),
        default=2,
    )
    parser.add_argument(
        "-L",
        "--grid_size",
        help="size of the grid (LxL), L >= 3",
        type=int,
        default=40,
    )
    parser.add_argument(
        "-k",
        "--min_neighbors",
        help="matching neighbors required to be happy",
        type=int,
        choices=range(0, 9),
        default=3,
    )
    parser.add_argument(
        "-N",
        "--num_agents",
        help="number of agents in the simulation, 2 <= N < L*L",
        type=int,
        default=1440,
    )
    parser.add_argument(
        "-e",
        "--max_epochs",
        help="Maximum number of simulation epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        help="Number of times to run the simulation",
        type=int,
        default=30,
    )
    return parser.parse_args(args=arg_list)


if __name__ == "__main__":
    import sys
    args = parse_args()
    print(type(args))
    coloredlogs.install(level=args.log_level * 10,
                        logger=LOG,
                        milliseconds=True)
    # Check boundaries on arguments
    if args.grid_size <= 2:
        LOG.error("GRID_SIZE must be >= 3!")
        sys.exit(1)
    if args.num_agents < 2 or args.num_agents >= args.grid_size**2:
        LOG.error("NUM_AGENTS must be in the interval [2, L*L)")
        sys.exit(1)
    # Run main
    main(
        args.grid_size,
        args.min_neighbors,
        args.num_agents,
        args.max_epochs,
        1,
        # TODO: Fix this to do multiple iterations
        # args.iterations,
    )
    logging.shutdown()
