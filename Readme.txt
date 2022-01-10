Download data from this link.
https://drive.google.com/drive/folders/19rOmpmwZM_k2fsIhr9WIb-lz5-MxskZa?usp=sharing

It contains two folders
1. finalized (Contains files and code)
2. generatedata (Contains dataset)

Following is the execution sequence.

# Divide data into training and test sets.

python dividedata.py 1

# Generate Sub-datasets based on P-values
python pvalue.py $SLURM_ARRAY_TASK_ID 5.05915e-50
python pvalue.py $SLURM_ARRAY_TASK_ID 5.05915e-30
python pvalue.py $SLURM_ARRAY_TASK_ID 5.05915e-10
python pvalue.py $SLURM_ARRAY_TASK_ID 1


# This file contains code to train 2D CNN. 
python pmodel.py  1


# This file contains code to train 1D CNN. 
python pmodel2.py 1
