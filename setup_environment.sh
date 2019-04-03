#!/bin/bash
source ~/.bashrc
mac=0
if [[ "$OSTYPE" =~ ^darwin ]]; then
    echo "You use MacOS."
    mac=1
fi

function program_is_installed {
  # set to 1 initially
  local return_=1
  # set to 0 if not found
  type $1 >/dev/null 2>&1 || { local return_=0; }
  # return value
  echo "$return_"
}

if [ $(program_is_installed conda) == 1 ]; then
  echo "Conda installed."
else
  echo "Installing Conda."
  if [ $mac -eq 1 ]
  then
    curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -o miniconda.sh
  else
      wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh -q
  fi

  clear
  chmod +x miniconda.sh
  ./miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  echo "## DeepDIVA ##" >> $HOME/.bashrc
  echo export PATH="$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
fi

clear

# Create an environment
echo "Installing packages. This will take some time."
if [ $mac -eq 1 ]
then
    conda env create -q -f environment_mac.yml
else
    conda env create -q -f environment.yml
fi

clear

# Set up PYTHONPATH
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> $HOME/.bashrc
echo "## DeepDIVA ##" >> $HOME/.bashrc
echo "Setup completed!"
echo "Please run 'source ~/.bashrc' to refresh your environment"
echo "You can activate the deepdiva environment with 'source activate deepdiva'"