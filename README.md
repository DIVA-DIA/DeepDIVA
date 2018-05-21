# DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments

DeepDIVA is an infrastructure designed to enable quick and intuitive
setup of reproducible experiments with a large range of useful analysis
functionality.
Reproducing scientific results can be a frustrating experience, not only
in document image analysis but in machine learning in general.
Using DeepDIVA a researcher can either reproduce a given experiment with
a very limited amount of information or share their own experiments with
others.
Moreover, the framework offers a large range of functions, such as
boilerplate code, keeping track of experiments, hyper-parameter
optimization, and visualization of data and results.
DeepDIVA is implemented in Python and uses the deep learning framework
[PyTorch](http://pytorch.org/).
It is completely open source and accessible as Web Service through
[DIVAServices](http://divaservices.unifr.ch).

## Additional resources

- [DeepDIVA Homepage](https://diva-dia.github.io/DeepDIVAweb/index.html)
- [Tutorials](https://diva-dia.github.io/DeepDIVAweb/articles.html)
- [Paper on arXiv](https://arxiv.org/abs/1805.00329) 

## Getting started

In order to get the framework up and running it is only necessary to clone the latest version of the repository:

``` shell
git clone https://github.com/DIVA-DIA/DeepDIVA.git
```

Run the script:

``` shell
bash setup_environment.sh
```

Reload your environment variables from `.bashrc` with: `source ~/.bashrc`

## Verifying Everything Works

To verify the correctness of the procecdure you can run a small experiment. Activate the DeepDIVA python environment:

``` shell
source activate deepdiva
```

Download the MNIST dataset:

``` shell
python util/data/get_a_dataset.py mnist --output-folder toy_dataset
```

Train a simple Convolutional Neural Network on the MNIST dataset using the command:

``` shell
python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --no-cuda
```

## Citing us

If you use our software, please cite our paper as (will be updated soon):

``` latex
@inproceedings{albertipondenkandath2018deepdiva,
    archivePrefix = {arXiv},
    arxivId = {1805.00329},
    eprint = {1805.00329},
    author = {Alberti, Michele Pondenkandath, Vinaychandran Würsch, Marcel Ingold, Rolf Liwicki, Marcus},
    title = {{DeepDIVA: A Highly-Functional Python Framework for Reproducible Experiments}},
    year = {2018},
    month = {apr},
}
```

## License

Our work is on GNU Lesser General Public License v3.0

