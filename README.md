# GNN Conversation Graph

#### Owner: Ke Xu
#### Supervisors: Milan Gritta, Pasquale Minervini

#### MSc Project with Huawei Noah's Ark Lab (London)
#### Research field: graph neural network, conversation graph, LSTM, NLP

#### MSc Data Science and Machine Learning, University College London


-----------------------------------------------------

### Disclaimer
The full codes for this project are not provided due to copyright regulations. Codes not available include the construction of conversation graph in data preprocessing, and a few other modelling methods with copyrights owned by Huawei. The core codes for this part refer to the following paper (Gritta et al., 2021) https://arxiv.org/abs/2010.15411 . 

Hence, the current codes in this repository are not capable to carry out full training with error raised due to the above reason. A Jupyter notebook is provided alongside to show the results of some simple test run. Again, this notebook would need to be run with the pieces censored at the current stage.

The complete version will be available and updated when the codes for the above paper become open sourced.


-----------------------------------------------------

### How to use
**Step 1**: Download datasets from https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz \
(files: test.json.zip, val.json.zip, train.json.zip)

**Step 2**: Unzip and add `test.json`, `val.json`, `train.json` to the master folder.

**Step 3**: Create empty folders in master with names: Data1, Data2, Data3

**Step 4**: Run `multiwoz/multiwoz_save_data.py` to save the subgraph data into Data1, Data2, Data3 for later training. (This takes some time to run for depth_limit > 2)

**Step 5**: Run `GNN_evaluation.py` for training. Change model, depth of the subgraph, hyperparameters, etc. in `GNN_evaluation.py`

