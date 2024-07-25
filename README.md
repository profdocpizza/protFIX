## Setup (not tested)

- Clone and enter protFIX repository:
    ```
    git clone https://github.com/profdocpizza/protFIX.git
    cd protFIX
    ```

- Recreate and activate protfix conda environment (Cython must be installed before ampal):
    ```
    conda create -n protfix  python=3.8.18
    conda activate protfix
    pip install Cython
    conda env update -f config/environment.yml
    conda activate protfix
    ```

- Install [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) on your system and update install path to [config/config.yaml](config/config.yaml).

- Populate [tasks.yaml](/tasks.yaml) with your protein information that needs to be fixed.

- Run [protFIX.py](/protFIX.py) to diffuse the missing bits with rfdiffusion inpaint_str feature and pack the residue atoms.

- Download [Scwrl4](http://dunbrack.fccc.edu/lab/SCWRLdownload) and install it. Then add your Scwrl4 installation folder to PATH.
    ```
    echo 'export PATH="$PATH:/path/to/your/Scwrl4"'>> ~/.bashrc
    ```