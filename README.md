# UE COMPVISIO Project
## TOPIC B : PatchMatch

**Student names: Franck UX, Nampoina RAVELOMANANA and Selman SEZGIN**

### Structure of the project
- `PatchMatch.py` contains an implementation of the Patchmatch algorithm. This code has not been developed by us but comes from another repository ([See here](https://github.com/MingtaoGuo/PatchMatch)). Note also that we modified a bit the code to parallelize some parts of the algorithm.
- `functions.py` contains some useful functions to propagate a mask from a NNF, and to compute some metrics to evaluate the estimated masks.
- `integration.py` contains the three integration methods: direct, sequential and hybrid.
- `tracking_bags.ipynb`, `tracking_bear.ipynb` and `tracking_rhino.ipynb` are three notebooks which perform object tracking by using PatchMatch algorithm, with the three integration methods. Each of them is applied to a specific image sequence.

### How to run the code ?
- Install the required libraries with the command `pip install -r requierements.txt`
- Put the folder `sequences-train` (the image sequences) next to the notebooks.
- Run the notebooks to get the results.