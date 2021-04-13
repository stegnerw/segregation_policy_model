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
# Standard library
from __future__ import annotations
import argparse
import configparser
import logging
import pathlib
import random
import shutil
from abc import ABC, abstractmethod
# Packages
import coloredlogs
import matplotlib.pyplot as plt
import numpy as np
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
if pathlib.Path("/tmp").resolve().exists():
    TMP_DIR = pathlib.Path("/tmp/segregation_model").resolve()
else:
    TMP_DIR = IMAGE_DIR.joinpath("tmp")
TMP_DIR.mkdir(mode=0o775, exist_ok=True)

# Simulation constants
MAX_SEARCHES = 100  # The parameter Q, used in the random policy
RED = np.array([1, 0, 0], dtype=np.float64)
BLUE = np.array([0, 0, 1], dtype=np.float64)
EMPTY = np.array([0, 0, 0], dtype=np.float64)

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
        self.legend_name = "Base model"
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
                self.population.append(Agent(np.copy(BLUE), coord))
                self.blue_agents.append(self.population[-1])
                self.env[tuple(coord)] = self.population[-1]
            elif i < self.num_agents:
                self.population.append(Agent(np.copy(RED), coord))
                self.red_agents.append(self.population[-1])
                self.env[tuple(coord)] = self.population[-1]
            else:
                self.empty_cells.append(Agent(np.copy(EMPTY), coord))
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
        self.temp_gif_dir = TMP_DIR.joinpath(
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
        images = []
        for path in sorted(list(self.temp_gif_dir.iterdir())):
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
        fig.savefig(plt_path, dpi=500)
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
        self.legend_name = "Random"

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
    arg_dict["search_diameter"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.num_friends = arg_dict["num_friends"]
        self.search_diameter = arg_dict["search_diameter"] // 2
        self.file_prefix = str(
            f"social_policy_{self.grid_size}L_"
            f"{self.num_agents}N_{self.min_neighbors}k_"
            f"{self.num_friends}n_{self.search_diameter*2+1}p")
        self.model_name = "Social Policy"
        self.legend_name = str(f"Social p={self.search_diameter*2+1} "
                               f"n={self.num_friends}")

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
        """Make recommendations for `agent` within `search_diameter` of `friend`

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
        for i in range(-self.search_diameter, self.search_diameter + 1):
            for j in range(-self.search_diameter, self.search_diameter + 1):
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


class ExclusiveSocialModel(SocialModel):
    """Wayne Stegner
    Segregation model which implements the exclusive social policy.

    At the beginning, each agent randomly picks `num_friends` friends.
    This time, the friends must be the same color as the agent.
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
    arg_dict["search_diameter"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.num_friends = arg_dict["num_friends"]
        self.search_diameter = arg_dict["search_diameter"] // 2
        self.file_prefix = str(
            f"exclusive_social_policy_{self.grid_size}L_"
            f"{self.num_agents}N_{self.min_neighbors}k_"
            f"{self.num_friends}n_{self.search_diameter*2+1}p")
        self.model_name = "Exclusive Social Policy"
        self.legend_name = str(f"Exclusive Social p={self.search_diameter*2+1}"
                               f" n={self.num_friends}")

    def init_population(self) -> None:
        """Initialize the population with friends"""
        super().init_population()
        for agent in self.blue_agents:
            indeces = list(range(len(self.blue_agents)))
            random.shuffle(indeces)
            for i in indeces[:self.num_friends]:
                agent.friends.append(self.blue_agents[i])
        for agent in self.red_agents:
            indeces = list(range(len(self.red_agents)))
            random.shuffle(indeces)
            for i in indeces[:self.num_friends]:
                agent.friends.append(self.red_agents[i])


class GreedySocialModel(SocialModel):
    """Siddharth Barve
    Segregation model which implements the greedy social policy.

    At the beginning, each agent randomly picks `num_friends` friends.
    Each move, the agent polls its friends for available locations
    which will make it happy, and it randomly picks one. If the friends,
    do not recommend a location, the agent selects new friends.

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
    arg_dict["search_diameter"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    def __init__(self, arg_dict):
        super().__init__(arg_dict)
        self.num_friends = arg_dict["num_friends"]
        self.search_diameter = arg_dict["search_diameter"] // 2
        self.file_prefix = str(
            f"greedy_social_policy_{self.grid_size}L_"
            f"{self.num_agents}N_{self.min_neighbors}k_"
            f"{self.num_friends}n_{self.search_diameter*2+1}p")
        self.model_name = "Greedy Social Policy"
        self.legend_name = str(f"Greedy Social p={self.search_diameter*2+1} "
                               f"n={self.num_friends}")

    def init_friends(self, agent):
        agent.friends = []
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
        while (len(recommendations) == 0):
            for friend in agent.friends:
                recommendations += self.make_recommendation(agent, friend)
            #If friends don't make you happy, get new friends.
            if len(recommendations) == 0:
                self.init_friends(agent)

        # Choose a random recommendation
        random.shuffle(recommendations)
        self.swap_cells(agent, recommendations[0])
        return True


################################################################################
# CLI handler functions
################################################################################

# These have to be here because Python does not support forward declaration...
MODELS = {
    "random": RandomModel,
    "social": SocialModel,
    "exclusive_social": ExclusiveSocialModel,
    "greedy_social": GreedySocialModel,
}


def main(arg_dicts: list[dict], plot_name: str,
         plot_path: pathlib.Path) -> None:
    """Main function.

    The main function handles starting the correct model.. Eventually,
    it will handle multiple simulation targets and facilitate plotting
    happiness time-series data on the same plots for the report.

    Parameters
    ---------
    arg_dicts : list[dict]
        Dictionary of arguments for clean passing all arguments. The
        relevant items in each arg_dict are enumerated below.
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
    arg_dict["search_diameter"] : int
        Search radius of each friend for "social" policy. Different
        policies may use this differently if desired.
    """
    # Run all the models
    models = list()
    for arg_dict in arg_dicts:
        models.append(MODELS[arg_dict["model"]](arg_dict))
        models[-1].run_sim()

    # Plot the averages against each other
    fig, axis = plt.subplots()
    axis.set_xlabel("Epoch")
    axis.set_xlim([0, models[0].max_epochs])
    epochs = np.arange(models[0].max_epochs + 1)
    axis.set_xticks(epochs)
    axis.set_ylabel("Happiness")
    axis.set_title(f"{plot_name} mean happiness vs epoch number")
    for model in models:
        mean = model.happiness.mean(axis=0)
        stdev = model.happiness.std(axis=0)
        # TODO: (#5) If someone wants to make this look nicer, PLEASE DO!
        axis.errorbar(epochs,
                      mean,
                      yerr=stdev,
                      label=model.legend_name,
                      linewidth=0.5)
    plt_path = IMAGE_DIR.joinpath(plot_path)
    fig.legend(loc="lower right")
    fig.savefig(plt_path, dpi=500)
    fig.clf()
    plt.close()
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
        "config_file",
        help="config file location relative to current directory",
        type=str,
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
    return parser.parse_args(args=arg_list)


def parse_config(config_file: str) -> list[dict]:
    """Parse the config file into a list of dictionaries.

    Parameters
    ----------
    config_file : str
        Path to the configuration file.

    Returns
    -------
    list[dict]
        List of dictionaries holding the configs for each model run.
        Each dictionary will have each parameter as a key and the value
        as the value. Returns None on read error.
    """
    # Make sure config file exists
    config_file = pathlib.Path(config_file).resolve()
    if not config_file.exists():
        LOG.error(f"Config file not found {str(config_file)}")
        return None

    # Set up parser and read each section
    parser = configparser.ConfigParser()
    parser.read(config_file)
    configs = list()
    for section in parser.sections():
        config = dict()
        config["model"] = str(parser[section]["model"])
        # If not "True" it will be assumed false
        config["make_gif"] = parser[section]["make_gif"] == "True"
        config["grid_size"] = int(parser[section]["grid_size"])
        if config["grid_size"] < 2:
            LOG.error("grid_size must be at least 2")
            return None
        config["min_neighbors"] = int(parser[section]["min_neighbors"])
        if config["min_neighbors"] < 0 or config["min_neighbors"] > 8:
            LOG.error("min_neighbors must be in the interval [0, 8]")
            return None
        config["num_agents"] = int(parser[section]["num_agents"])
        if (config["num_agents"] < 2) or (config["num_agents"] >=
                                          config["grid_size"]**2):
            LOG.error("num_agents must be in the interval [2,"
                      " grid_size*grid_size)")
            return None
        config["max_epochs"] = int(parser[section]["max_epochs"])
        if config["max_epochs"] < 1:
            LOG.error("max_epochs must be at least 1")
            return None
        config["iterations"] = int(parser[section]["iterations"])
        if config["iterations"] < 1:
            LOG.error("iterations must be at least 1")
            return None
        config["num_friends"] = int(parser[section]["num_friends"])
        if config["num_friends"] < 0:
            LOG.error("num_friends must be at least 0")
            return None
        config["search_diameter"] = int(parser[section]["search_diameter"])
        if (config["search_diameter"] < 1) or ((config["search_diameter"] % 2)
                                               == 0):
            LOG.error("search_diameter must be at least 1 and odd")
            return None
        configs.append(config)
    return configs


if __name__ == "__main__":
    import sys
    args = parse_args()
    coloredlogs.install(level=args.log_level * 10,
                        logger=LOG,
                        milliseconds=True)
    configs = parse_config(args.config_file)
    if configs is None:
        LOG.error(f"Error reading config file {args.config_file}")
        sys.exit(1)
    # Run main
    config_file = pathlib.Path(args.config_file).resolve()
    plot_name = config_file.stem
    plot_path = IMAGE_DIR.joinpath(f"{plot_name}_happiness.png")
    plot_name = plot_name.replace("_", " ").capitalize()
    main(configs, plot_name, plot_path)
    logging.shutdown()
