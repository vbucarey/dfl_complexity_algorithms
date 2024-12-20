########## DESCRIPTION #########

This repository contains the datasets for Bipartite Matching and Shortest Path problems considered in the following paper

Bucarey, V., Calderón, S., Munoz, G., & Semet, F. Decision-focused predictions via pessimistic bilevel optimization: complexity and algorithms (2024).

########## DATASETS #########

Each file is named as follows:
  db_<type>_N<samples>_<attributes>_<grid>_deg_<degree>_noise_<noise>_<split>.txt

Where:
-- <type>: Specifies the problem type:
   - "bin": Bipartite Matching
   - "sp": Shortest Path
-- <samples>: Indicates the number of samples in the dataset.
-- <attributes>: Number of attributes describing each data point.
-- <grid>: Structure of the grid for the problem.
-- <degree>: Parameter specifies the extent of model misspecification.
-- <noise>: Level of noise added to the data.
-- <split>: Specifies whether the dataset is for training ("train") or testing ("test").

########## FILE FORMAT #########

Each dataset is stored in a CSV-like format with the following structure:
- Header row: Describes the column names.
- Data rows: Each row corresponds to an edge in the graph with the following columns:
  1. `data`: Identifier of the observation.
  2. `node_init`: Initial node of the edge.
  3. `node_term`: Terminal node of the edge.
  4. `c`: Cost of the edge.
  5. `at1`, `at2`, ..., `at5`: Attributes describing the edge.

########## NOTES #########

- Each problem type contains datasets generated with varying parameters such as node degree and noise levels.
- Training datasets (`train`) and testing datasets (`test`) are provided for each parameter combination.

########## CITATION #########

If you use these datasets, please reference them appropriately in your work.
