# DSPMAE: Directed Semantic Prompt Masked Autoencoder for Graph Representation Learning

This is the official implementation of the paper: **DSPMAE: Directed Semantic Prompt Masked Autoencoder for Graph Representation Learning**.

## Quick Start

You can quickly start training by running the following command:
```bash
python main_transductive.py --dataset cora --encoder gat --decoder gat --seed 0 --device 0
```

To run the model using the optimal hyperparameters and configurations reported in the paper, you can simply use the --use_cfg flag:
```bash
python main_transductive.py --dataset cora --device 0 --use_cfg
```

Note on Datasets: Manual dataset preparation is not required. The specified datasets will be automatically downloaded and processed via DGL during the first runtime.