# residual-stream-interp

Codebase for paper:  [Naturally Computed Scale Invariance in the Residual Stream of ResNet18](https://arxiv.org/abs/2504.16290)

To get started, clone the repository and then create a conda environment from the environment.yml file:

```
conda env create -f environment.yml
```


**batch_process.sh** contains calls to generate data required for analyses.  Python calls must be supplied with directory arguments of where to save data to and here to load from.

**center_lucent_optimize.py** produces a feature visualization of the center unit in a specified layer's channel.

**tuning_curve.py** finds and saves the top-9 activating ImageNet validation images for every center unit in a specified layer.

**stream_inspect.py** performs activation analyses to extract scale invariant channels.

**scale_robust.py** implements the ablation experiments.

GCC-related files and dict_tools contains experimental code pertaining to group crosscoders.  Work in progress.
