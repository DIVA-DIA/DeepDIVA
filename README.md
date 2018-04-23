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

- [Fancy page](https://diva-dia.github.io/DeepDIVAweb/index.html)
- [Tutorials](https://diva-dia.github.io/DeepDIVAweb/tutorial.html)
- [Paper on arXiv](https://github.com/DIVA-DIA/DeepDIVA)

## Getting started

In order to get the framework up and running it is only necessary to:

- Clone the latest version of the repository: `git clone https://github.com/DIVA-DIA/DeepDIVA.git`
- Run the script `bash setup_environment.sh`.
- Reload your environment variables from `.bashrc` with: `source ~/.bashrc`

To verify the correctness of the procecdure you can run a small experiment:

- Activate the DeepDIVA python environment: `source activate deepdiva`
- Download the MNIST dataset `python util/data/get_a_dataset.py mnist --output-folder toy_dataset`
- Train a simple Convolutional Neural Network on the MNIST dataset using the command: `python template/RunMe.py --output-folder log --dataset-folder toy_dataset/MNIST --lr 0.1 --ignoregit --no-cuda`

