# Verify Conda installation: (https://conda.io/docs/user-guide/install/index.html)
PSEUDO{
if conda NOT installed:
    install conda
PSEUDO}

# Create conda environment (https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-packages)
conda create --file requirements.txt --name deepdiva

# Activate the environment
source activate deepdiva

# Install missing packages from pip
pip install tensorboardX
pip install tqdm
pip install sigopt

# Congratulate user on success
echo "You're the best! Everything worked!"

