# -*- coding: utf-8 -*-
"""This module is the main module.
"""
import logging
import pathlib
import shutil
import argparse
import random
import coloredlogs
from abc import ABC, abstractmethod
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

# Simulation constants
MAX_SEARCHES = 100  # The parameter Q, used in the random policy
RED = np.array([1, 0, 0])
BLUE = np.array([0, 0, 1])
EMPTY = np.array([0, 0, 0])

################################################################################
# Simulation classes
################################################################################


class SegregationModel(ABC):
    """Abstract class for the segregation model interface.

    Parameters
    ----------
    grid_size : int
        Size of the environment grid (the length).
    min_neighbors : int
        Minimum neighbors of the same type to be happy.
    num_agents : int
        Number of agents to populate the grid.
    max_epochs : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    iterations : int
        Number of iterations to run the simulation.
    make_gif: bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    """
    def __init__(
        self,
        grid_size: int,
        min_neighbors: int,
        num_agents: int,
        max_epochs: int,
        iterations: int,
        make_gif: bool,
    ):
        self.grid_size = grid_size
        self.min_neighbors = min_neighbors
        self.num_agents = num_agents
        self.max_epochs = max_epochs
        self.iterations = iterations
        self.make_gif = make_gif
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        self.happiness = np.zeros((iterations, max_epochs + 1))
        self.init_env()
        self.file_prefix = "segregation_model"
        # This gets initialized in a function but the linter doesn't like that
        self.temp_gif_dir = IMAGE_DIR.joinpath(self.file_prefix)

    def run_sim(self):
        """Run the simulation with the parameters of the object."""
        LOG.info(f"Starting simulation for {self.file_prefix}")
        # Clear old gifs from this policy
        # TODO: (#3) Should we do this? Might be a bit aggressive
        for path in list(IMAGE_DIR.glob(f"{self.file_prefix}*")):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for self.iteration in range(self.iterations):
            # TODO: (#4) This could be better with progress bars
            LOG.info(f"Begin iteration {self.iteration+1} of "
                     f"{self.iterations}")
            self.init_env()
            self.step = 0
            self.epoch = 0
            self.make_temp_gif_dir()
            self.save_env()
            self.log_happiness()
            for self.epoch in range(1, self.max_epochs + 1):
                # TODO: (#4) Again, could be better with progress bars
                LOG.info(f"Begin epoch {self.epoch} of {self.max_epochs}")
                agent_cells = np.concatenate((self.get_matching_coords(RED),
                                              self.get_matching_coords(BLUE)))
                np.random.shuffle(agent_cells)
                for agent in agent_cells:
                    agent = tuple(agent)
                    if ((not self.cell_stats(agent, self.env[agent])[0])
                            and self.move_policy(agent)):
                        self.step += 1
                        self.save_env()
                self.log_happiness()
            self.save_gif()
            shutil.rmtree(self.temp_gif_dir, ignore_errors=True)
        self.save_happiness()
        LOG.info(f"Completed simulation for {self.file_prefix}")

    @abstractmethod
    def move_policy(self, agent_coord: tuple[int]) -> bool:
        """Find a place to move for the given agent.

        This must be implemented for each policy.

        Parameters
        ----------
        agent_coord : tuple[int]
            Coordinates of the agent to move.

        Returns
        -------
        bool
            True if the agent moved, else False.
        """
        ...

    def init_env(self):
        """Initialize `self.env`."""
        self.env = np.zeros((self.grid_size, self.grid_size, 3))
        coords = [(i, j) for i in range(self.grid_size)
                  for j in range(self.grid_size)]
        random.shuffle(coords)
        for i, coord in enumerate(coords):
            if i < (self.num_agents // 2):
                self.env[coord] = BLUE
            elif i < self.num_agents:
                self.env[coord] = RED
            else:
                self.env[coord] = EMPTY

    def swap_cells(self, coord1: tuple[int], coord2: tuple[int]) -> None:
        """Swap 2 cells in place in `self.env`.

        Parameters
        ----------
        coord1, coord2 : np.ndarray
            The coordinates of the cells to swap in the form (x, y).
        """
        temp_cell = np.copy(self.env[coord1])
        self.env[coord1] = np.copy(self.env[coord2])
        self.env[coord2] = temp_cell

    def cell_stats(
        self,
        agent_coord: tuple[int],
        agent_type: np.ndarray,
    ) -> tuple[bool, int]:
        """Check the stats of a cell for a given agent type.

        Status includes whether or not the given `agent_type` is happy
        at `agent_coord`, as well as how many neighbors are the same
        type. The agent is happy if at least `num_agents` neighbors are
        of type `agent_type` surrounding `agent_coord`.

        Parameters
        ----------
        agent_coord : tuple[int]
            Coordinates of the agent to check.
        agent_type : np.ndarray
            Type of the agent to check. This is not derived from
            `agent_coord` to facilitate scouting potential moves without
            actually doing the move.

        Returns
        -------
        tuple[bool, int]
            A tuple containing a bool indicating whether the agent of
            type `agent_type` is happy at `agent_coord` and an int
            showing the number of neighbors of type `agent_type`.
        """
        agent_type = self.env[agent_coord]
        matching_neighbors = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                if (self.env[(agent_coord[0] + i) % self.grid_size,
                             (agent_coord[1] + j) %
                             self.grid_size] == agent_type).all():
                    matching_neighbors += 1
        return ((matching_neighbors >= self.min_neighbors), matching_neighbors)

    def get_matching_coords(self, agent_type: np.ndarray) -> np.ndarray:
        """Get a list of coordinates matching agent_type.

        Parameters
        ----------
        agent_type : np.ndarray
            The agent type, either `RED`, `BLUE`, or `EMPTY`. These are
            in the form [red_val, green_val, blue_val].

        Returns
        -------
        np.ndarray
            Array of coordinates of cells containing agent_type in the form
            [[x0, y0], [x1, y1], ...].
        """
        return np.array([(i, j) for i in range(self.grid_size)
                         for j in range(self.grid_size)
                         if (self.env[i, j] == agent_type).all()])

    def log_happiness(self):
        """Log the portion of happy agents."""
        agent_cells = np.concatenate(
            (self.get_matching_coords(RED), self.get_matching_coords(BLUE)))
        happy_agents = 0
        for agent in agent_cells:
            agent = tuple(agent)
            happy, _ = self.cell_stats(agent, self.env[agent])
            if happy:
                happy_agents += 1
        happy_portion = happy_agents / self.num_agents
        self.happiness[self.iteration, self.epoch] = happy_portion

    def make_temp_gif_dir(self) -> None:
        """Generates and defines the temporary gif dir."""
        if not self.make_gif:
            return
        self.temp_gif_dir = IMAGE_DIR.joinpath(
            f"{self.file_prefix}_{self.iteration:02d}")
        shutil.rmtree(self.temp_gif_dir, ignore_errors=True)
        self.temp_gif_dir.mkdir(mode=0o775, exist_ok=True)

    def save_env(self) -> None:
        """Save a single frame image."""
        if not self.make_gif:
            return
        # Create the plot/image
        fig, axis = plt.subplots()
        axis.imshow(self.env)
        axis.axis("off")
        axis.set_title(f"{self.file_prefix} epoch {self.epoch}\n"
                       f"step {self.step:06d}")
        # Save the figure as a PNG
        fig.savefig(self.temp_gif_dir.joinpath(f"step_{self.step:06d}"))
        fig.clf()
        plt.close()

    def save_gif(self) -> None:
        """Save the images into a gif."""
        if not self.make_gif:
            return
        LOG.info("Generating GIF...")
        temp_gif_dir = IMAGE_DIR.joinpath(
            f"{self.file_prefix}_{self.iteration:02d}")
        images = []
        for path in sorted(list(temp_gif_dir.iterdir())):
            image = Image.open(path)
            images.append(image.copy())
            image.close()
        gif_path = IMAGE_DIR.joinpath(
            f"{self.file_prefix}_{self.iteration:02d}.gif")
        images[0].save(
            gif_path,
            save_all=True,
            duration=25,
            append_images=images[1:],
        )
        LOG.info(f"A GIF of the simulation has been saved in:\n{gif_path}")

    def save_happiness(self) -> None:
        """Save a plot of mean happiness and standard deviation."""
        # Calculate metrics to be plotted
        mean = self.happiness.mean(axis=0)
        stdev = self.happiness.std(axis=0)
        epochs = np.arange(self.max_epochs + 1)
        # Create the plot
        fig, axis = plt.subplots()
        # TODO: (#5) If someone wants to make this look nicer, PLEASE DO!
        axis.errorbar(epochs, mean, yerr=stdev)
        axis.set_xlabel("Epoch")
        axis.set_xlim([0, self.max_epochs])
        axis.set_xticks(epochs)
        axis.set_ylabel("Happiness")
        axis.set_ylim([0, 1])
        axis.set_title(f"{self.file_prefix} mean happiness vs epoch number")
        plt_path = IMAGE_DIR.joinpath(f"{self.file_prefix}_happiness.png")
        fig.savefig(plt_path)
        fig.clf()
        plt.close()
        LOG.info("The mean happiness time series plot has been saved in:\n"
                 f"{plt_path}")


class RandomModel(SegregationModel):
    """Segregation model which implements the random policy.

    If the agent is unhappy, it searches random empty cells until it
    finds one which makes it happy, or if it searches `MAX_SEARCHES`
    it will select the one with the most matching neighbors.

    Parameters
    ---------
    grid_size : int
        Size of the environment grid (the length).
    min_neighbors : int
        Minimum neighbors of the same type to be happy.
    num_agents : int
        Number of agents to populate the grid.
    max_epochs : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    iterations : int
        Number of iterations to run the simulation.
    make_gif: bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    """
    def __init__(
        self,
        grid_size: int,
        min_neighbors: int,
        num_agents: int,
        max_epochs: int,
        iterations: int,
        make_gif: bool,
    ):
        super().__init__(
            grid_size,
            min_neighbors,
            num_agents,
            max_epochs,
            iterations,
            make_gif,
        )
        self.file_prefix = "random_policy"

    def move_policy(self, agent_coord: tuple[int]) -> bool:
        """Randomly choose a tile that makes the agent happy.

        If the agent cannot be happy in MAX_SEARCHES, it chooses the one it
        saw which has the most neighbors of the same type.

        Parameters
        ----------
        agent_coord : tuple[int]
            Coordinates of the agent to move.

        Returns
        -------
        bool
            True if the agent moved, else False.
        """
        agent_type = self.env[agent_coord]
        # Get randomized list of empty cells
        empty_cells = self.get_matching_coords(EMPTY)
        np.random.shuffle(empty_cells)
        best_cell = tuple(empty_cells[0])
        _, best_cell_matching_neighbors = self.cell_stats(
            agent_coord, agent_type)
        for i, cell in enumerate(empty_cells):
            cell = tuple(cell)
            happy, matching_neighbors = self.cell_stats(
                agent_coord, agent_type)
            if matching_neighbors > best_cell_matching_neighbors:
                best_cell = cell
                best_cell_matching_neighbors = matching_neighbors
            if happy:
                break
            if i >= MAX_SEARCHES:
                break
        self.swap_cells(agent_coord, best_cell)
        return True


################################################################################
# CLI handler functions
################################################################################


def main(
    grid_size: int,
    min_neighbors: int,
    num_agents: int,
    max_epochs: int,
    iterations: int,
    make_gif: bool,
) -> None:
    """Main function.

    Parameters
    ---------
    grid_size : int
        Size of the environment grid (the length).
    min_neighbors : int
        Minimum neighbors of the same type to be happy.
    num_agents : int
        Number of agents to populate the grid.
    max_epochs : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    iterations : int
        Number of iterations to run the simulation.
    make_gif : bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    """
    LOG.debug("Running simulation with parameters:")
    LOG.debug(f"grid_size (L) = {grid_size}")
    LOG.debug(f"min_neighbors (k) = {min_neighbors}")
    LOG.debug(f"num_agents (N) = {num_agents}")
    LOG.debug(f"max_epochs={max_epochs}")
    random_model = RandomModel(
        grid_size,
        min_neighbors,
        num_agents,
        max_epochs,
        iterations,
        make_gif,
    )
    random_model.run_sim()


def parse_args(arg_list: list[str] = None) -> argparse.Namespace:
    """Parse the arguments.

    Parameters
    ----------
    arg_list : list[str]
        Arguments to be parsed.

    Returns
    -------
    argparse.Namespace
        A namespace of parsed arguments.
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
        "-g",
        "--make_gif",
        help="create a gif of each iteration\n"
        "takes much longer - only recommended with -i 1",
        default=False,
        action="store_true",
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
        help="maximum number of simulation epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "-i",
        "--iterations",
        help="number of times to run the simulation",
        type=int,
        default=30,
    )
    return parser.parse_args(args=arg_list)


if __name__ == "__main__":
    import sys
    args = parse_args()
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
        args.iterations,
        args.make_gif,
    )
    logging.shutdown()
