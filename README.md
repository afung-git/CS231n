# CS231n

### How to use Google Cloud ###
1) Create disk from image (default Gooogle deep learning tensorflow disk)
2) Specify settings of the disk (#CPU, RAM, #GPU, traffic settings etc.)
3) Set up static IP and firewall rule
4) SSH into instance
5) Download zip file from GitHub
6) Source dataset retrieval script
7) Generate Jupyter config file
8) Edit settings (port # etc.)
9) jupyter notebook
10) Replace instance address with static IP address

### How to use Cython
1) Create virtualenv
2) Activate virtualenv
3) Install dependencies via requirements.txt
4) jupyter notebook
5) Replace address with static IP address

### RECOMMENDED!
### Setup virtual environment ###
cd assignment1

sudo pip install virtualenv      # This may already be installed

virtualenv .env                  # Create a virtual environment

source .env/bin/activate         # Activate the virtual environment

pip install -r requirements.txt  # Install dependencies

Work on the assignment for a while ...

deactivate                       # Exit the virtual environment
