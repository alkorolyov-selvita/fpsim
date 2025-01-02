ENV_NAME="rdkit"

# Function to check the source of `conda`
check_conda_source() {
    local conda_path
    conda_path=$(which conda 2>/dev/null)

    if [[ -z "$conda_path" ]]; then
        echo "Conda is not installed."
        return 1
    fi

    if [[ "$conda_path" == "${HOME}/miniforge3/condabin/conda" ]]; then
        echo "Conda is already installed and originates from ${HOME}/miniforge3/condabin/conda."
        return 0
    else
        echo "Conda is installed, but not from ${HOME}/miniforge3/condabin/conda."
        return 1
    fi
}

# install dependencies
sudo apt-get install jq pv curl tar -y

# Check if Miniforge is already installed
if check_conda_source; then
    echo "Skipping Miniforge installation."
else
    # Download and install Miniforge
    echo "Downloading and installing Miniforge..."
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh -b
    # Source Conda and Mamba initialization scripts
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    source "${HOME}/miniforge3/etc/profile.d/mamba.sh"
    conda activate
fi

mamba env create
conda activate $ENV_NAME
#python -m ipykernel install --name $ENV_NAME --user