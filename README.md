# spEMO 😒
This is the official code repo for the paper: spEMO: Exploring the Capacity of Foundation Models for Analyzing Spatial Multi-Omic Data.

## Installation

To install spEMO, considering to install python, [torch](https://pytorch.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html) with latest version in ahead, and then prepare the environment for pathology foundation models:

```
pip install timm
pip install 'transformers[torch]'
```

And then prepare the environment for large language model:

```
pip install openai
```

And then prepare the environment for analyzing spatial transcriptomics:

```
pip install scanpy
pip install squidpy
```

If you meet error while loading the pickle files, please consider to downgrade the version of package (anndata)[https://anndata.readthedocs.io/en/latest/].


If you intend to test different domain-specific expert models, considering to follow the instructions of their own tutorial for installation: [SpaGCN](https://github.com/jianhuupenn/SpaGCN/tree/master), [MultiMIL](https://github.com/theislab/multimil), and [SC3](https://github.com/hemberg-lab/sc3s).

Finally, prepare the environment for evaluation:

```
pip install scib-metrics
```

We also upload a conda yml file to reproduce our environment used for this project. We recommend users deploying this project in High Performance Computing centers. In personal computer, its installation time should be less than 10 minutes.

## Accessing foundation models

For pathology foundation models, some of them need application. Please consider applying for the access before starting, including (UNI)[https://huggingface.co/MahmoodLab/UNI] and (GigaPath)[https://huggingface.co/prov-gigapath/prov-gigapath]. For more methods, please check the descriptions of (HEST)[https://github.com/mahmoodlab/HEST].

## Tutorials

We provide a demo notebook under the demo folder. We provide the embeddings used in the demo notebook in this location.

For other tasks, we include our codes under different folders. Please check these folders for information.

To protect personal information, we will not release expert feedbacks for medical report generation.

# Incoming functions

- [x] Support spatial foundation models (Novae, Nichformer, etc.)
- [x] Support more pathology foundation models (Virchow2, etc.)

## Acknowledgement

We thank the developers of different pathology foundation models and large langugage models, as well as the developers of HEST, for their great work.

## Citation
```
@article {Liu2025.01.13.632818,
	author = {Liu, Tianyu and Huang, Tinglin and Tong, Ding and Wu, Hao and Humphrey, Peter and Perincheri, Sudhir and Schalper, Kurt and Ying, Rex and Xu, Hua and Zou, James and Mahmood, Faisal and Zhao, Hongyu},
	title = {spEMO: Exploring the Capacity of Foundation Models for Analyzing Spatial Multi-Omic Data},
	elocation-id = {2025.01.13.632818},
	year = {2025},
	doi = {10.1101/2025.01.13.632818},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/08/23/2025.01.13.632818},
	eprint = {https://www.biorxiv.org/content/early/2025/08/23/2025.01.13.632818.full.pdf},
	journal = {bioRxiv}
}
```

