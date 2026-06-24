## Summary

> note: torch version==1.12.0, if you use other versions, some incompatible errors may occur!

# Requirements

- Python 3.8
- pytorch 1.12
- PyTorch-lighting 1.8.3
- numpy
- scipy
- open3d
- PyTorch Geometric
  - torch-cluster 1.6.0
  - torch-geometric 2.1.0
  - torch-scatter 2.0.9
  - torch-sparse 0.6.15
  - torch-spline-conv 1.2.1

# Data

The data used in this paper are from https://github.com/zhangxiann/TPAMI_Gait_Identification

# Preprocess

For the dataset **DVS128-Gait-Day**, day_downsample.py and to_graph.py in the datascripts folder.
For the dataset **DVS128-Gait-Night**, night_downsample.py and to_graph.py in the datascripts folder.
For dataset **EV-CASIA-B**, casia_downsample.py, casia_graph.py in datascripts folder respectively.

# Train and evaluation

The train and evaluation codes of the above 3 datasets are in the Day, Night, Casia folders, respectively.