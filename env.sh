# Create a Conda environment named "reconstruct" and install necessary packages
conda create --name reconstruct python=3.11 numpy Cython scipy h5py matplotlib tqdm ipython scikit-learn tensorboard ipython imageio scikit-image future colorama

# Activate the newly created environment
conda activate reconstruct

# Install specific versions of PyTorch and torchvision
conda install pytorch==2.1.0 torchvision==0.15.2 -c pytorch

# Install additional packages using pip
pip install kornia==0.7.0 moviepy pypng opencv-python

