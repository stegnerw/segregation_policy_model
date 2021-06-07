# Segregation Policy Model

Homework 2 for EECE 7065 Complex Systems.

## Objective

The objective of this assignment is to investigate emergent behavior of several
segregation policies.
We take a grid of cells and add members from two classes (red and blue).
Each cell is happy if it has *k* neighbors of the same class (in this case,
*k=3*).
If the cell is unhappy, it moves based on a policy, which we define below.

### Random Policy

The first policy is the random policy, where unhappy agents search for a random
cell until they find one
which makes them happy.
If no such cell is found after 100 searches, the agent moves to the cell which
makes it the happiest.

### Social Recommendation Policy

Each agent is assigned a certain number of friends.
When a agent is unhappy, it asks its friends if there are any cells nearby which
would make them happy.
If such cells are found, the agent chooses a random cell which makes it happy.
Otherwise, the agent does not move.

### Custom Policies

After testing the two above policies, we each came up with a policy.
Those policies are described in the report.

## Running

Use the provided conda environment in `environment.yml` and then run the main
file.
The homework requirements specified that we should only have one file for code,
so it is fuller than ideal.
Run `./main.py -h` for a list of parameters.

## Authors

This project was done by:

- Wayne Stegner <[stegnerw](https://github.com/stegnerw)>
- Zuguang Liu <[liu2z2](https://github.com/liu2z2)>
- Siddharth Barve <[Siddharth-Barve](https://github.com/Siddharth-Barve)>
