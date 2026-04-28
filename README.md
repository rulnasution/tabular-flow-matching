# (Published at TMLR) Tabular Flow Matching for Data Synthesis

<p align="center">
  <a href="https://github.com/rulnasution/tabular-flow-matching/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/badge/license-Apache 2.0-green">
  </a>
  <a href="https://openreview.net/forum?id=RdOjoAa66L">
    <img alt="Openreview" src="https://img.shields.io/badge/review-OpenReview-red">
  </a>
  <a href="https://arxiv.org/abs/2512.00698">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2512.00698-blue">
  </a>
</p>


This is the implementation repository of **Flow Matching for Tabular Data Synthesis** paper by Bahrul Ilmi Nasution, Floor Eijkelboom, Mark Elliot, Richard Allmendinger, Christian A. Naesseth

This github consists of the algorithm for tabular flow matching, along with its benchmarks. The folders are
- /tabsyn : TabSyn (Hybrid VAE with diffusion for training the latent variables) from Zhang et al. (2024)
- /baselines/tabddpm: TabDDPM (Tabular diffusion model) from Kotelnikov et al. (2023)

Our algorithms are in 

- /tabsynflow: TabSynFlow (Hybrid VAE with flow matching for training the latent variables) 
- /baselines/tabvvfm: TabVFM/TabbyFlow (Tabular Variational Flow Matching) using **MLP**
- /baselines/tabtvfm: TabVFM/TabbyFlow (Tabular Variational Flow Matching) using **Transformers**

Special credit to https://github.com/amazon-science/tabsyn for providing the initial code framework.

## Installing Dependencies

Python version: 3.10

Create environment

```
conda create -n tabsyn python=3.10
conda activate tabsyn
```

Install pytorch
```
pip install torch torchvision torchaudio
```

or via conda
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Install other dependencies

```
pip install -r requirements.txt
```

Install dependencies for GOGGLE

```
pip install  dgl -f https://data.dgl.ai/wheels/cu117/repo.html

pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

Create another environment for the quality metric (package "synthcity")

```
conda create -n synthcity python=3.10
conda activate synthcity

pip install synthcity
pip install category_encoders
```

## Preparing Datasets

### Using the datasets adopted in the paper

Download raw dataset:

```
python download_dataset.py
```

Process dataset:

```
python process_dataset.py
```

### Using your own dataset

First, create a directory for you dataset [NAME_OF_DATASET] in ./data:
```
cd data
mkdir [NAME_OF_DATASET]
```

Put the tabular data in .csv format in this directory ([NAME_OF_DATASET].csv). **The first row should be the header** indicating the name of each column, and the remaining rows are records.

Then, write a .json file ([NAME_OF_DATASET].json) recording the metadata of the tabular, covering the following information:
```
{
    "name": "[NAME_OF_DATASET]",
    "task_type": "[NAME_OF_TASK]", # binclass or regression
    "header": "infer",
    "column_names": null,
    "num_col_idx": [LIST],  # list of indices of numerical columns
    "cat_col_idx": [LIST],  # list of indices of categorical columns
    "target_col_idx": [list], # list of indices of the target columns (for MLE)
    "file_type": "csv",
    "data_path": "data/[NAME_OF_DATASET]/[NAME_OF_DATASET].csv"
    "test_path": null,
}
```
Put this .json file in the .Info directory.

Finally, run the following command to process the UDF dataset:
```

```

## Training Models

Direct training can be seen in ```example_training_job.slurm```. This part is if you want to do step by step and want to do it in slurm.

For baseline methods, use the following command for training:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode train
```

Options of [NAME_OF_DATASET]: adult, default, shoppers, magic, beijing, news
Options of [NAME_OF_BASELINE_METHODS]: tabddpm, tabsyn, tabvvfm, tabsynflow

For Tabsyn or TabSynFlow, use the following command for training:

```
# train VAE first
python main.py --dataname [NAME_OF_DATASET] --method vae --mode train

# after the VAE is trained, train the diffusion model
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode train
```

## Tabular Data Synthesis

For baseline methods, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method [NAME_OF_BASELINE_METHODS] --mode sample --save_path [PATH_TO_SAVE]
```

For Tabsyn, use the following command for synthesis:

```
python main.py --dataname [NAME_OF_DATASET] --method tabsyn --mode sample --save_path [PATH_TO_SAVE]

```

The default save path is "synthetic/[NAME_OF_DATASET]/[METHOD_NAME].csv"

## Evaluation
We evaluate the quality of synthetic data using metrics from various aspects.


#### Utility and Risk evaluation (for census dataset)

To evaluate the utility and risk of the synthetic data, use the following command:

 ```
python bash_urisk.py --dataname adult
```

The evaluation consists of ROC univariate and bivariate, CIO, TCAP, Utility, and Risk 


## Citation

If you use this repository, please don't forget to cite:

```
@article{nasution2026flowmatchingtabulardata,
      title={Flow Matching for Tabular Data Synthesis}, 
      author={Bahrul Ilmi Nasution and Floor Eijkelboom and Mark Elliot and Richard Allmendinger and Christian A. Naesseth},
      year={2026},
      journal={Accepted in Transactions on Machine Learning Research}
      url={https://arxiv.org/abs/2512.00698}, 
}
```

## License

This project is licensed under the Apache-2.0 License.
