# Verify Conda installation: (https://conda.io/docs/user-guide/install/index.html)

# return 1 if global command line program installed, else 0
# example
# echo "node: $(program_is_installed node)"
function program_is_installed {
  # set to 1 initially
  local return_=1
  # set to 0 if not found
  type $1 >/dev/null 2>&1 || { local return_=0; }
  # return value
  echo "$return_"
}


if [ $(program_is_installed conda) == 1 ]; then
  echo "conda installed"
else
  echo "installing conda"
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  source .bashrc

fi
# Create conda environment (https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-packages)
conda create --file requirements.txt --name deepdiva

# Activate the environment
source activate deepdiva
# Install missing packages from pip
pip install tensorboardX
pip install tqdm
pip install sigopt
pip install colorlog

#pytorch
#conda install pytorch torchvision cuda91 -c pytorch


# Congratulate user on success
echo "You're the best! Everything worked!"

