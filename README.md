# UCL-Huawei-MSc-Project
### MSc Project with Huawei
### Research field: graph neural network, conversation graph, LSTM, NLP

### MSc Data Science and Machine Learning, University College London
### Huawei Noah's Ark Lab (London)

-----------------------------------------------------

### How to use
**Step 1**: Download datasets from https://github.com/ConvLab/ConvLab/tree/master/data/multiwoz \
(files: test.json.zip, val.json.zip, train.json.zip)

**Step 2**: Unzip and add `test.json`, `val.json`, `train.json` to the master folder.

**Step 3**: Create empty folders in master with names: Data1, Data2, Data3

**Step 4**: Run `multiwoz/multiwoz_save_data.py` to save the subgraph data into Data1, Data2, Data3 for later training. (This takes some time to run for depth_limit > 2)

**Step 5**: Run `GNN_evaluation.py` for training. Change model, depth of the subgraph, hyperparameters, etc. in `GNN_evaluation.py`
