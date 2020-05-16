# optimization
Repository of the code (Python 3) for the homework of Optimization for Data Science, academic year 2019/2020, UNIPD.

# Packages required:
numpy,
matplotlib,
sklearn,
argparse,
json,
time,
tabulate.
tqdm

# Main scripts (src)
In *src* it is possible to find the main scripts used for this work: <br>
1. **loss.py**: containing the LogisticLoss class
2. **optimizer.py**: containing the abstract Optimizer class and the main three optimization algorithms (Gradient Descent, SGD and SVRG) <br>
3. **model.py**: containing the Model class, which is a simple Linear Classifier with the fit and predict functions. It also contains a *main* which is a demo of how the model works with some 1D and 2D data and the different optimizations algorithms.

# Test script (src)
In *src* there is the script **test.py** that gives the possibilities to test the model with the choosen optimizer on three different datasets (IRIS, MNIST, a9a). <br>
The script has the following arguments:
- data: String. The dataset to use (mnist, a9a, iris). Default is iris.
- optim: String. the optimization algorithm chosen (gd, sgd, svrg). Default is gd <br>
- tollerance: Float. The minimum norm that the gradient must have in order to stop the optimization algorithm. Default is 0.001 <br>
- iter_epoch: Int. The number of example to take in every step of SVRG. Default 5000. Ignored if not SVRG. <br> 
- epochs: Int. The number of epochs. Default is 10.
- lr: Float. the step-size. Default is 0.05
- reg_coeff: Float. The regularization lambda coefficient. Default is 0.001 <br>
- init: String. The weights initialization. Either "zeros" or "random" (normal distribution). Default is zeros <br>
- verbose: 0/1. Print some information on every step (1) or don't (0). Default is 0.
- seed: Int. Define a random seed. Default is None 

Example:

    cd C:\Users\fgrim\Desktop\Optimization\optimization
    python src\model.py --data "mnist" --optim "gd" --epochs 300 --lr 0.2 --reg_coeff 0.01 --init "random" --seed 42

# Result notebook:
result.ipynb is a notebook (set up in a colab enviroment, but easy to modfiy to run it in local) that fit different models (different dataset and optimizer) and return some results.

# Results (results)
In the *results* folder there are some json files containing some results and two scripts **display.py** and **reduce.py** that were done in order to process and visualize the results

# Datasets (data)
In the *data* folder there are the **MNIST** dataset and the **a9a** dataset. The **IRIS** dataset is called by *sklearn.dataset**.

# Images (images)
In the *images* folder there are some results images
