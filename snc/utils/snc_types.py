import numpy as np
from typing import Dict, Tuple, List, Any

# ActionSpace is a vector space of dimension equal to number of activities.
ActionSpace = np.ndarray

# ActionProcess is a matrix with number of rows equal to number of activities and number of columns
# equal to the number of points.
ActionProcess = np.ndarray

# 1-dim np.ndarray
Array1D = np.ndarray

# BufferRoutesType represents routes (destination, origin), obtained from B matrix.
BufferRoutesType = Dict[Tuple[int, int], Tuple[int, int]]

# BufferMatrix is a matrix with number rows equal to number of buffers vectors and number of columns
# equal to the number of activities.
BufferMatrix = np.ndarray

# Column vector with one column and any number of rows, i.e. shape = (X, 1)
ColVector = np.ndarray

# A coordinate on the x,y plane in 2D
Coord2D = Tuple[float, float]

# ConstituencyMatrix is a matrix with number rows equal to number of resources, (including physical
# resources, and extra coupled constraints between physical resources), and number of columns equal
# to the number of activities.
ConstituencyMatrix = np.ndarray

# Data Dict, is what is returned as the output of a simulation
DataDict = Dict[str, List[np.ndarray]]

# ExitNodeType represents nodes that take jobs out of the network as (row, column) of B matrix.
ExitNodeType = List[Tuple[int, int]]

# IdlingSet is an array of integers
IdlingSet = np.ndarray

# ResourceCapacity is a 1-D array of integers
ResourceCapacity = np.ndarray

# Generic matrix, with shape ((X, Y)).
Matrix = np.ndarray

# List of matrices
MatricesList = List[np.ndarray]

# NuMatrix is a matrix with number rows equal to number of workload vectors and number of columns
# equal to the number of physical resources (i.e. rows of the constituency matrix).
NuMatrix = np.ndarray

# PolicyMatrix is a matrix with number of rows equal to the number of resources, and number of
# columns equal to the number of activities plus 1.
PolicyMatrix = np.ndarray

# Reporter Cache, is a generic data store, used by the Reporter object for logging
ReporterCache = Dict[str, Any]

# ResourceSpace is a vector space of dimension equal to number of physical resources
ResourceSpace = np.ndarray

# Row vector with one column and any number of rows, i.e. shape = (1, X)
RowVector = np.ndarray

# StateSpace is a vector space of dimension equal to number of buffers
StateSpace = np.ndarray

# StateProcess is a matrix with number rows equal to number of buffers and columns equal to the
# number of points.
StateProcess = np.ndarray

# SupplyNodeType represents supply nodes as (row, column) of B matrix.
SupplyNodeType = List[Tuple[int, int]]

# WorkloadCov is a square matrix with number rows and columns equal to number of workload vectors
WorkloadCov = np.ndarray

# WorkloadMatrix is a matrix with number rows equal to number of workload vectors and number of
# columns equal to the number of buffers
WorkloadMatrix = np.ndarray

# Workload space is a vector space of dimension equal to number of workload vectors
WorkloadSpace = np.ndarray

# WorkloadProcess is a matrix with number rows equal to number of workload vectors and number of
# columns equal to the number of points
WorkloadProcess = np.ndarray
