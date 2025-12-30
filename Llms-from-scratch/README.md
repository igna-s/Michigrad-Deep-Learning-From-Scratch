#Teacher notes

# LLM Workshop
Repository with datasets and examples for the Large Language Models Workshop  
- First edition – Jun/2024 (For LIFIA)

## How to use this repository
Fork this repository to your GitHub account so you can make Push Requests comfortably. This also makes it easier to keep track of authorship for contributions. Then, `git clone` a local copy from your own repo.

git clone git@github.com:<your_github_username>/Michigrad-Autograd-Engine.git



If you want to contribute anything, don’t hesitate to send me a Pull Request.

## Required during the workshop (Growing...)
Here we will detail the tools needed for the workshop, in terms of software and cool tools that might be useful.

### Conda Environments
We will make extensive use of conda to avoid contaminating the system Python with all the libraries we are going to install. Additionally, conda will allow us to work with specific versions of certain libraries, and even specific Python versions.

My suggestion is to install [Miniconda 3](https://docs.anaconda.com/miniconda/). There is a [very good tutorial in English](https://www.whiteboxml.com/blog/the-definitive-guide-to-python-virtual-environments-with-conda) and a [somewhat improvable one in Spanish](https://github.com/jwackito/conda-environments-tutorial/blob/main/Conda%20Environments.md) (pull requests are welcome) covering everything you need to know about installing, creating, and managing conda virtual environments.

### Jupyter Hub/Lab (option 1)
[JupyterHub](https://jupyter.org/hub) is a multi-user version of Jupyter Notebooks. In case you are not familiar with Jupyter Notebooks, it is basically an interactive Python console that runs in the browser. JupyterHub also allows running notebooks using different kernels (a Python instance installed in a conda environment, with a set of libraries installed). It works at the cell level and allows editing and executing small sections of code comfortably. JupyterLab also allows creating projects and managing kernels conveniently.

### IPython (option 2)
An interactive Python console on steroids. Honestly, I prefer it over JupyterHub except in very specific cases. It is super lightweight and powerful. It comes with built-in magic commands (%magics). It can be customized and scripted as desired. Without a doubt, one of the best options to run Python cells interactively. If I tell you that I did my PhD almost exclusively using this tool, I wouldn’t be exaggerating. However, it is a terminal-based tool (it has no graphical environment). If you’re not comfortable with the terminal, it’s better to use JupyterHub.

### NumPy
For the first session (the bigrams part), only NumPy is needed. It is a very powerful linear algebra library that allows vectorized operations, and if one is a bit clever, it is possible to model an entire LLM using only this library. For convenience, we will later use Torch, but for now, NumPy is enough.

---
Don’t forget to give me a star on GitHub if this repo was useful to you!
