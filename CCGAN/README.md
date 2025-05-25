# Conditioned CryinGAN (CCGAN)

This project modifies the original CryinGAN (Yong, 2024), by adding conditioning. Finally the resulting architecture is applied to the hard spheres dataset.

## Citation
```
@article{yong2024dismaibench,
         author = {Yong, Adrian Xiao Bin and Su, Tianyu and Ertekin, Elif},
         title = {Dismai-Bench: benchmarking and designing generative models using disordered materials and interfaces},
         journal = {Digital Discovery},
         year = {2024},
         volume = {3},
         issue = {9},
         pages = {1889-1909},
         publisher = {RSC},
         doi = {10.1039/D4DD00100A},
         url = {http://dx.doi.org/10.1039/D4DD00100A}
}
```

## Instructions

1. clone repository
2. Experiments are created as bash scripts in folder 'experiments'
3. Experiments can be run from root folder (CryinGAN), with ``bash experiments/1-with-dist.sh``
   1. NOTE: ``experiments/`` containts a script ``create-venv.sh`` which can be used to creating the virtual environment that the experiments use.
4. Experiment results created to folder ``results/``.



## Operating System

Developed on MacOS, deployed on MacOS and Ubuntu.