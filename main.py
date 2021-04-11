#! /usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
# Authors: Wayne Stegner, Zuguang Liu, and Siddharth Barve
# Course: EECE7065
# Assignment: Homework 2 - Schelling Segregation Model
################################################################################
"""This module is the main module for the segregation model.
TODO: Usage, etc.
"""
from __future__ import annotations
import logging
import pathlib
import shutil
import argparse
import random
from abc import ABC, abstractmethod
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
PROJ_DIR = pathlib.Path(__file__).parent.resolve()
IMAGE_DIR = PROJ_DIR.joinpath("img")
IMAGE_DIR.mkdir(mode=0o775, exist_ok=True)
if pathlib.Path("/tmp").exists():
    TMP_DIR = pathlib.Path("/tmp/segregation_model").resolve()
else:
    TMP_DIR = IMAGE_DIR.joinpath("tmp")
TMP_DIR.mkdir(mode=0o775, exist_ok=True)

# Simulation constants
MAX_SEARCHES = 100  # The parameter Q, used in the random policy
RED = np.array([1, 0, 0])
BLUE = np.array([0, 0, 1])
EMPTY = np.array([0, 0, 0])

################################################################################
# Simulation classes
################################################################################


class Agent():
    """Class for holding agent information

    Parameters
    ----------
    color : np.ndarray
        The color of the agent
    pos : np.ndarray
        The initial position of the agent
    """
    def __init__(self, color: np.ndarray, pos: np.ndarray):
        self.color = color
        self.pos = pos
        self.friends = list()

    def __str__(self) -> str:
        """Return a string of the agent"""
        return f"Agent color {self.color} at position {self.pos}"


class SegregationModel(ABC):
    """Abstract class for the segregation model interface.

    Parameters
    ----------
    arg_dict : dict
        Dictionary of arguments for clean passing all arguments. The
        relevant items are enumerated below.
    arg_dict["make_gif"] : bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    arg_dict["grid_size"] : int
        Size of the environment grid (the length).
    arg_dict["min_neighbors"] : int
        Minimum neighbors of the same type to be happy.
    arg_dict["num_agents"] : int
        Number of agents to populate the grid.
    arg_dict["max_epochs"] : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    arg_dict["iterations"] : int
        Number of iterations to run the simulation.
    """
    def __init__(self, arg_dict):
        self.make_gif = arg_dict["make_gif"]
        self.grid_size = arg_dict["grid_size"]
        self.min_neighbors = arg_dict["min_neighbors"]
        self.num_agents = arg_dict["num_agents"]
        self.max_epochs = arg_dict["max_epochs"]
        self.iterations = arg_dict["iterations"]
        self.epoch = 0
        self.iteration = 0
        self.step = 0
        self.happiness = np.zeros((self.iterations, self.max_epochs + 1))
        # These are used in `init_population`
        self.env = None
        self.population = None
        self.blue_agents = None
        self.red_agents = None
        self.empty_cells = None
        self.file_prefix = "segregation_model"
        self.model_name = "Segregation Model"
        # This gets initialized in a function but the linter doesn't like that
        self.temp_gif_dir = TMP_DIR.joinpath(self.file_prefix)

    def run_sim(self):
        """Run the simulation with the parameters of the object."""
        LOG.info(f"Starting simulation for {self.model_name}")
        # Clear old gifs from this policy
        for path in list(IMAGE_DIR.glob(f"{self.file_prefix}*")):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        for self.iteration in range(self.iterations):
            # TODO: (#4) This could be better with progress bars
            LOG.info(f"Begin iteration {self.iteration+1} of "
                     f"{self.iterations}")
            self.init_population()
            self.step = 0
            self.epoch = 0
            self.make_temp_gif_dir()
            self.save_env()
            self.log_happiness()
            for self.epoch in range(1, self.max_epochs + 1):
                # TODO: (#4) Again, could be better with progress bars
                LOG.info(f"Begin epoch {self.epoch} of {self.max_epochs}")
                random.shuffle(self.population)
                for agent in self.population:
                    if self.move_policy(agent):
                        self.step += 1
                        self.save_env()
                self.log_happiness()
            self.save_gif()
            shutil.rmtree(self.temp_gif_dir, ignore_errors=True)
        self.save_happiness()
        LOG.info(f"Completed simulation for {self.model_name}")

    @abstractmethod
    def move_policy(self, agent: Agent) -> bool:
        """Find a place to move for the given agent.

        This must be implemented for each policy.

        Parameters
        ----------
        agent : Agent
            The agent to move.

        Returns
        -------
        bool
            True if the agent moved, else False.
        """
        ...

    def init_population(self):
        """Initialize `self.population`."""
        self.env = np.empty((self.grid_size, self.grid_size), dtype=object)
        self.population = list()
        self.blue_agents = list()
        self.red_agents = list()
        self.empty_cells = list()
        coords = self.get_coords()
        np.random.shuffle(coords)
        for i, coord in enumerate(coords):
            if i < (self.num_agents // 2):
                self.population.append(Agent(BLUE, coord))
                self.blue_agents.append(self.population[-1])
                self.env[tuple(coord)] = self.population[-1]
            elif i < self.num_agents:
                self.population.append(Agent(RED, coord))
                self.red_agents.append(self.population[-1])
                self.env[tuple(coord)] = self.population[-1]
            else:
                self.empty_cells.append(Agent(EMPTY, coord))
                self.env[tuple(coord)] = self.empty_cells[-1]

    def get_coords(self) -> np.ndarray:
        """Get a list of coordinates for the env grid

        Returns
        -------
        np.ndarray
            Array in the form [[0,0], [0,1], ..., [0,self.grid_size], ...,
            [self.grid_size,self.grid_size]]
        """
        num_grids = self.grid_size * self.grid_size
        return np.mgrid[0:self.grid_size,
                        0:self.grid_size].reshape(2, num_grids).transpose()

    def swap_cells(self, agent1: Agent, agent2: Agent) -> None:
        """Swap 2 cells in place in `self.env`.

        Parameters
        ----------
        agent1, agent2 : Agent
            The agents to swap in place
        """
        agent1.pos, agent2.pos = agent2.pos, agent1.pos
        self.env[tuple(agent1.pos)] = agent1
        self.env[tuple(agent2.pos)] = agent2

    def get_neighbors(self, agent: Agent) -> list[Agent]:
        """Return a list of neighboring agents

        Parameters
        ----------
        agent : Agent
            The agent to find neighbors of

        Returns
        -------
        list[Agent]
            List of agents neighboring `agent`
        """
        neighbors = list()
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                cell = self.env[(agent.pos[0] + i) % self.grid_size,
                                (agent.pos[1] + j) % self.grid_size]
                if cell:
                    neighbors.append(cell)
        return neighbors

    def agent_stats(self, agent: Agent) -> tuple[bool, int]:
        """Check the stats of an agent.

        Status includes whether or not `agent` is happy, as well as how
        many neighbors are the same type. The agent is happy if at
        least `self.num_agents` neighbors are of type `agent.color`
        surrounding `agent.pos`.

        Parameters
        ----------
        agent : Agent
            The agent to check.

        Returns
        -------
        tuple[bool, int]
            A tuple containing a bool indicating whether `agent` is
            happy and an int showing the number of neighbors of the
            same color.
        """
        matching_neighbors = 0
        neighbors = self.get_neighbors(agent)
        for neighbor in neighbors:
            if (neighbor.color == agent.color).all():
                matching_neighbors += 1
        return ((matching_neighbors >= self.min_neighbors), matching_neighbors)

    def log_happiness(self):
        """Log the portion of happy agents."""
        LOG.debug("Logging happiness")
        happy_agents = 0
        for agent in self.population:
            happy, _ = self.agent_stats(agent)
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
        LOG.debug("Saving environment frame")
        # Draw the env
        env = np.zeros((self.grid_size, self.grid_size, 3))
        for i, row in enumerate(self.env):
            for j, agent in enumerate(row):
                env[i][j] = agent.color
        # Create the plot/image
        fig, axis = plt.subplots()
        axis.imshow(env)
        axis.axis("off")
        axis.set_title(f"{self.model_name} epoch {self.epoch}\n"
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
        LOG.debug("Creating happiness plot")
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
        axis.set_title(f"{self.model_name} mean happiness vs epoch number")
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
    arg_dict : dict
        Dictionary of arguments for clean passing all arguments. The
        relevant items are enumerated below.
    arg_dict["make_gif"] : bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    arg_dict["grid_size"] : int
        Size of the environment grid (the length).
    arg_dict["min_neighbors"] : int
        Minimum neighbors of the same type to be happy.
    arg_dict["num_agents"] : int
        Number of agents to populate the grid.
    arg_dict["max_epochs"] : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    arg_dict["iterations"] : int
        Number of iterations to run the simulation.
    """
    def __init__(self, arg_dict: dict):
        super().__init__(arg_dict)
        self.file_prefix = str(f"random_policy_{self.grid_size}L_"
                               f"{self.num_agents}N_{self.min_neighbors}k")
        self.model_name = "Random Policy"

    def move_policy(self, agent: Agent) -> bool:
        """Randomly choose a tile that makes the agent happy.

        If the agent cannot be happy in MAX_SEARCHES, it chooses the one it
        saw which has the most neighbors of the same type.

        Parameters
        ----------
        agent : Agent
            The agent to move.

        Returns
        -------
        bool
            True if the agent moved, else False.
        """

        # If already happy, don't move
        happy, matching_neighbors = self.agent_stats(agent)
        if happy:
            return False
        np.random.shuffle(self.empty_cells)
        best_cell = agent
        best_matching_neighbors = matching_neighbors
        for i, cell in enumerate(self.empty_cells):

            # Swap the cells, check happiness, then swap back
            self.swap_cells(agent, cell)
            happy, matching_neighbors = self.agent_stats(agent)
            self.swap_cells(agent, cell)

            # Check for exit conditions
            if matching_neighbors > best_matching_neighbors:
                best_cell = cell
                best_matching_neighbors = matching_neighbors
            if happy:
                break
            if i >= MAX_SEARCHES:
                break

        # Swap with the best cell and report success
        self.swap_cells(agent, best_cell)
        return True


class SocialModel(SegregationModel):
    """Segregation model which implements the social policy.

    At the beginning, each agent randomly picks `num_friends` friends.
    Each move, the agent polls its friends for available locations
    which will make it happy, and it randomly picks one.

    Parameters
    ---------
    arg_dict : dict
        Dictionary of arguments for clean passing all arguments. The
        relevant items are enumerated below.
    arg_dict["make_gif"] : bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    arg_dict["grid_size"] : int
        Size of the environment grid (the length).
    arg_dict["min_neighbors"] : int
        Minimum neighbors of the same type to be happy.
    arg_dict["num_agents"] : int
        Number of agents to populate the grid.
    arg_dict["max_epochs"] : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    arg_dict["iterations"] : int
        Number of iterations to run the simulation.
    arg_dict["num_friends"] : int
        Number of friends for each agent for "social" policy. Different
        policies may use this differently if desired.
    arg_dict["search_radius"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.num_friends = arg_dict["num_friends"]
        self.search_radius = arg_dict["search_radius"] // 2
        self.file_prefix = str(f"social_policy_{self.grid_size}L_"
                               f"{self.num_agents}N_{self.min_neighbors}k_"
                               f"{self.num_friends}p_{self.search_radius}n")
        self.model_name = "Social Policy"

    def init_population(self) -> None:
        """Initialize the population with friends"""
        super().init_population()
        for agent in self.population:
            indeces = list(range(len(self.population)))
            random.shuffle(indeces)
            for i in indeces[:self.num_friends]:
                agent.friends.append(self.population[i])

    def move_policy(self, agent: Agent) -> bool:
        """Randomly choose a tile within a friend's search radius.

        Parameters
        ----------
        agent : Agent
            The agent to move.

        Returns
        -------
        bool
            True if the agent moved, else False.
        """

        # If already happy, don't move
        happy, _ = self.agent_stats(agent)
        if happy:
            return False

        # Get friend recommendations
        recommendations = list()
        for friend in agent.friends:
            recommendations += self.make_recommendation(agent, friend)

        # If no recommendations, don't move
        if len(recommendations) == 0:
            return False

        # Choose a random recommendation
        random.shuffle(recommendations)
        self.swap_cells(agent, recommendations[0])
        return True

    def make_recommendation(self, agent: Agent, friend: Agent) -> list[Agent]:
        """Make recommendations for `agent` within `search_radius` of `friend`

        Parameters
        ----------
        agent : Agent
            Agent which needs recommendations
        friend: Agent
            Agent to give recommendations

        Returns
        -------
        list[Agent]
            A list of empty cells which make `agent` happy
        """
        recommendations = list()
        for i in range(-self.search_radius, self.search_radius + 1):
            for j in range(-self.search_radius, self.search_radius + 1):
                if i == 0 and j == 0:
                    continue
                row = (friend.pos[0] + i) % self.grid_size
                col = (friend.pos[1] + j) % self.grid_size
                cell = self.env[row][col]
                if cell in self.empty_cells:
                    temp_agent = Agent(np.copy(agent.color), np.copy(cell.pos))
                    if self.agent_stats(temp_agent)[0]:
                        recommendations.append(cell)
        return recommendations


################################################################################
# CLI handler functions
################################################################################

# These have to be here because Python does not support forward declaration...
MODELS = {
    "random": RandomModel,
    "social": SocialModel,
}


def main(arg_dict) -> None:
    """Main function.

    The main function handles starting the correct model.. Eventually,
    it will handle multiple simulation targets and facilitate plotting
    happiness time-series data on the same plots for the report.

    Parameters
    ---------
    arg_dict : dict
        Dictionary of arguments for clean passing all arguments. The
        relevant items are enumerated below.
    arg_dict["model"] : str
        The model to run.
    arg_dict["make_gif"] : bool
        Whether or not to save a gif. Saving a gif takes significantly
        longer.
    arg_dict["grid_size"] : int
        Size of the environment grid (the length).
    arg_dict["min_neighbors"] : int
        Minimum neighbors of the same type to be happy.
    arg_dict["num_agents"] : int
        Number of agents to populate the grid.
    arg_dict["max_epochs"] : int
        Maximum epochs for each iteration. One epoch is one time
        through the population of agents.
    arg_dict["iterations"] : int
        Number of iterations to run the simulation.
    arg_dict["num_friends"] : int
        Number of friends for each agent for "social" policy. Different
        policies may use this differently if desired.
    arg_dict["search_radius"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    model_obj = MODELS[arg_dict["model"]](arg_dict)
    model_obj.run_sim()
    shutil.rmtree(TMP_DIR)


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
        description="A segregation policy simulator.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "model",
        help="model to run",
        type=str,
        choices=list(MODELS.keys()),
    )
    parser.add_argument(
        "-ll",
        "--log_level",
        help="set the logging level:\n"
        "1 = DEBUG\n"
        "2 = INFO\n"
        "3 = WARNING\n"
        "4 = ERROR\n"
        "5 = CRITICAL\n",
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
    parser.add_argument(
        "-n",
        "--num_friends",
        help="\"social\" policy - number of friends",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-p",
        "--search_radius",
        help="\"social\" policy - search radius of friend (must be odd)",
        type=int,
        default=3,
    )
    return parser.parse_args(args=arg_list)


if __name__ == "__main__":
    import sys
    args = parse_args()
    coloredlogs.install(level=args.log_level * 10,
                        logger=LOG,
                        milliseconds=True)
    # Check boundaries on arguments
    if args.grid_size < 2:
        LOG.error("GRID_SIZE must be at least 2")
        sys.exit(1)
    if args.num_agents < 2 or args.num_agents >= args.grid_size**2:
        LOG.error("NUM_AGENTS must be in the interval [2, L*L)")
        sys.exit(1)
    if args.num_friends < 0:
        LOG.error("NUM_FRIENDS must be at least 0")
        sys.exit(1)
    if args.search_radius < 3:
        LOG.error("SEARCH_RADIUS must be at least 3")
        sys.exit(1)
    if (args.search_radius % 2) == 0:
        LOG.error("SEARCH_RADIUS must be odd")
        sys.exit(1)
    # Run main
    main(vars(args))
    logging.shutdown()
