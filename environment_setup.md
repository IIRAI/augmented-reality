# Environment Setup

Herein it is explained the python setup I used for the project, I tried to say everything and to be clear, obviously there can be, and there are, some errors so look for other guides to setup your project.  
I used a *virtual environment*, which is always preferable, and herein I explain also how to setup it from scratch for *python 3*.

## 1) Virtual Environement setup

> Note: This [site](https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-18-04) explains how to set up a virtual environment where to install *jupyter notebook* and how to use *ssh tunneling*.

### Basic commands for python, just a reminder

The meaning should be obvious

    python --version
    python3 --version
    pip --version
    pip3 --version
    pip show pip
    pip3 show pip

### Basic install

Below I show the basic install process I followed to setup a virtual environment.

    sudo apt update

Install `pip` (sometimes already installed by default), which is foundamental to work with *python*, and `dev`, which is needed to work with *jupyter*. Install for python3:

    sudo apt install python3-pip python3-dev

Then upgrade them to the latest version

    sudo -H pip3 install --upgrade pip

**Remark:** Here `pip3` crashed and the command `pip3 --version` returned error. I *rebooted* the computer and everything was fine.

Install the package to create virtual environment:

    sudo pip install virtualenv
    sudo pip install virtualenvwrapper

`virtualenv` is the base tool, `virtualenvwrapper` adds usefull tools on top of virtualenv and makes things a lot easier. Please take a moment to read the documentation on their sites: [virtualenv](https://virtualenv.pypa.io/en/stable/installation/) and [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/).

### Setup virtualenv

Create the directory were our *virtual environment* will live:

    mkdir -p .virtualenvs

Add to `.bashrc` the following lines:

    export WORKON_HOME=$HOME/.virtualenvs
    export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
    source /usr/local/bin/virtualenvwrapper.sh

**Remark:** The path to `virtualenvwrapper.sh` may change (e.g. `/usr/bin/virtualenvwrapper.sh`), so if this does not work don't panic, just check where is it.

The first one create the environment variable that specify where are our virtual environment are, the second tells explicitly what version of python to use (this line cannot be specified and should work by default, but in my case it was not) the third allows to use virtaulenvwrapper commands.  
Then source it:

    source .bashrc

### Foundamental viertualenvwrapper commands

Create a virtual environment in `$WORKON_HOME`:

    mkvirtualenv <name_env>

Access the environment

    workon <name_env>

Exit the environment

    deactivate

You can list all the environment created just with

    workon

To delete a virtual environment

    rmvirtualenv <name_env>
