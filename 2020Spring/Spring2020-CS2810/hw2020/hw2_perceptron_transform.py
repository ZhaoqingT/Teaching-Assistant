#!/usr/bin/env python
# coding: utf-8

# # CS 2810 HW 2: Programming
# 
# Due Jan 23 @ 9 PM
# 
# Read and study `hw2_linear_perceptron_fish_tutorial.ipynb` before continuing.
# 
# #### Submission Instructions
# 
# You can download any Jupyter Notebook as a python file via `file` -> `download as...` -> `python .py`.  We reccomend working in this jupyter notebook for the HW but you may also download this notebook as a `.py` file before completing the assignment and working with the terminal or your favorite IDE.
# 
# Submit this problem as a single `.py` file which is named `hw02_<firstname>_<lastname>.py` (ensure only those two underscores are used).
# 
# The names of all variables and function definitions must be consistent with what is used here.  For example, a problem may ask you to define function `matmult_n(X, y, n)`.  Name the function and its input arguments exactly as shown, noting capitalization.  This is critical to streamline grading.  If your solutions are not named appropriately, the grader may elect to remove all credit.  Thanks for your help in making our grading more efficient!
# 
# #### Academic Integrity
# 
# Under no circumstances should you observe another student's code which was written for this programming assignment, from this year or past years.

# ## Setup
# 
# We begin by borrowing some code from the Perceptron tutorial.  Note that our `plot_w` now requires we pass `x` as well.  This allows us to move away from a hard-coded limits on the x axis, as we used in our last example.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math


np.set_printoptions(precision=2)

def scatter_data(x, label):
    """ a scatter plot of fish
    
    Args:
        x (np.array): (n_fish, 2) float, fish features for all fish, from
            both fish types.  Each row represents a fish's features.
        label (np.array): (n_fish) boolean.  describes which rows of x 
            correspond to which fish type.  (if label[idx_fish] = 0
            then the idx_fish row of x is of type 0)
    """
    plt.scatter(x[label==0, 1], x[label==0, 2], label='group 0', color='b')
    plt.scatter(x[label==1, 1], x[label==1, 2], label='group 1', color='r')
    plt.legend()
    plt.xlabel('size s (lbs)')
    plt.ylabel('length l (cm)')

def plot_w(w, x):
    m = -w[1] / w[2]
    b = -w[0] / w[2]
    
    # a column vector of feature 1 for every sample
    feat_1 = x[:, 1]
    feat_2 = x[:, 2]
    
    x_boundary = np.linspace(feat_1.min(), feat_1.max(), 5)
    y_boundary =  m * x_boundary + b
    
    plt.plot(x_boundary, y_boundary, color='k', linewidth=2, label='perceptron (w)')
    plt.ylim(feat_2.min(), feat_2.max())
    plt.legend()
    
def update_perceptron(x, label, w):
    """ updates perceptron by updating once for each sample

    Args:
         x (np.array): (n_sample, n_feature) features
         label (np.array): (n_sample), boolean class label
         w (np.array): (n_feature) initial perceptron weights
         
    Returns:
        w (np.array): n_features weight vector, defines Linear
            Perceptron
    """
    n_sample, n_feature = x.shape

    for idx_sample in range(n_sample):
        # _x are the features of a single fish
        _x = x[idx_sample, :]
        
        # _label is the label, 0 or 1, of _x
        _label = label[idx_sample]

        if np.dot(_x, w) < 0 and _label:
            # perceptron estimate: type 0, actual fish type: 1
            w = w + _x
        elif np.dot(_x, w) >= 0 and not _label:
            # perceptron estimate: type 1, actual fish type: 0
            w = w - _x

    return w


# Let's build a new dataset and take a look:

# In[2]:


def get_data(n_per_grp):
    """ builds synthetic data
    
    NOTE: this data already has a constant 1 as its first feature
    
    Args:
        n_per_grp (int): number of observations, per group, to sample
        
    Returns:
        x (np.array): (n_sample, 3) float, features of each sample, from
            both groups.  Each row represents one sample (e.g. a fish).
        label (np.array): (n_fish) boolean.  describes which rows of x 
            correspond to which group
    """
    # initalize random seed (we get the 'same' random values each time)
    np.random.seed(0)
    
    # mean of each group
    mean_grp_0 = np.array([0, 0])
    mean_grp_1 = np.array([1, 1])
    
    # covariance of each group (same cov)
    cov = np.array([[1, -1],
                    [-1, 2]])
    
    # sample observations from each group
    x_0 = np.random.multivariate_normal(size=n_per_grp, 
                                        mean=mean_grp_0, 
                                        cov=cov)
    x_1 = np.random.multivariate_normal(size=n_per_grp,
                                        mean=mean_grp_1, 
                                        cov=cov)
    # combine samples
    x = np.concatenate((x_0, x_1), axis=0)

    # build label vector
    n_sample = x.shape[0]
    label = np.ones(n_sample, dtype=bool)
    label[:n_per_grp] = False
    
    # append column of ones (see 'Adding a y-intercept to the Linear Perceptron')
    n_sample = x.shape[0]
    constant_one_col = np.ones((n_sample, 1))
    x = np.concatenate((constant_one_col, x), axis=1)
    
    return x, label

x, label = get_data(n_per_grp=100)

scatter_data(x, label)


# ## Training Stop Criteria
# 
# One topic which the tutorial did not cover was how many epochs of training to perform before stopping.  Stop too early and there may be a better `w` out there which we would have found.  But if we're too aggressive about continuing to train the Perceptron it may waste computation or, worst of all, never stop training!
# 
# ### P 2.1: Write a Training Stop Criteria Function (5 pts each)
# Your task is to write two functions which tell when we should stop training.  The intuition is that if `update_perceptron()` doesn't change `w` all that much, we ought to stop.  In particular, they stop when ...
# 
# 1. ... the norm does not change more than 1% from w_old to w
# 1. ... the angle between w_old to w is less than .01 radians
# 
# Templates for each function are given below:

# In[3]:


def stop_norm(w, w_old):
    """ return True when norm does not change more than 1% from w_old to w
    
    Args:
        w (np.array): (n_feature) Perceptron Weights after update
        w_old (np.array): (n_feature) Perceptron Weights before update
        
    Returns:
        stop (bool): boolean, True indicates that training should stop
    """
    # your code here
    Norm_w = np.linalg.norm(w)
    Norm_w_old = np.linalg.norm(w_old)
    if (Norm_w - Norm_w_old)/Norm_w_old < 0.01:
        return True
    # this is a placeholder, will always stop
    return False
    
def stop_angle(w, w_old):
    """ return True when norm does not change more than 1% from w_old to w
    
    Args:
        w_new (np.array): (n_feature) Perceptron Weights after update
        w_old (np.array): (n_feature) Perceptron Weights before update
        
    Returns:
        stop (bool): boolean, True indicates that training should stop
    """
    # your code here
    Norm_w = np.linalg.norm(w)
    Norm_w_old = np.linalg.norm(w_old)
    Unit_w = w / Norm_w
    Unit_w_old = w_old / Norm_w_old
    angle = np.arccos(np.clip(np.dot(Unit_w, Unit_w_old), -1.0, 1.0))
    if angle > 0.01:
        return True
    # this is a placeholder, will always stop
    return False
    


# Below is example code to help you evaluate your functions.  Initially, it always runs until `max_epoch` epochs are complete because the functions only `return False`, so that training never stops.  
# 
# Remember: clicking to the left of the output will expand or hide it, which is helpful if you've got 100 epochs to look at.

# In[4]:


w = np.array([0, 0, 0])

idx_epoch = 0

# we include a maximum number of iterations
max_epoch = 10

# runs forever, unless we stop it
while True:
    w_old = w
    w = update_perceptron(x, label, w)
    
    print(f'\n ---epoch {idx_epoch}---')
    print(f'w_old: {w_old}')
    print(f'w: {w}')
    
    # HINT: you may want to replace stop_norm() with stop_angle()
    if stop_angle(w_old, w):
        print('Stop Training')
        break
        
    idx_epoch = idx_epoch + 1
    
    if idx_epoch >= max_epoch:
        print(f'max_epoch ({max_epoch}) reached: Hard stop')
        break
        
    print('Keep Training')
    
    


# ### P 2.2: Analyze Training Stop Function (4 pts)
# Explain which convergence method from P 2.1 you think is best and why.  Provide an example of `w` and `w_old` which illustrates why the other method is so undesirable.  Write your answer as a string below

# In[5]:


p22 = """ insert answer here """


# ### P 2.3: Transformations (4 pts)
# 
# Write expressions for each transformation matrix described below:
# 
# 1. (3 x 3) matrix A which scales the first dimension by 1, the second by 2 and the last by 100
# 1. (2 x 2) matrix B which rotates vectors pi / 4 radians counter-clockwise
# 
# Trig functions are taken from the [math library](https://docs.python.org/3.6/library/math.html).

# In[6]:


from math import sin, cos, pi

A = np.array([[1,0,0],[0,2,0],[0,0,100]])
B = np.array([[cos(pi/4), -sin(pi/4)],[sin(pi/4), cos(pi/4)]])


# Here are some test cases with expected behavior.  If the matrices for A, B are correct, these won't throw an error.

# In[7]:


# these test cases are provided to ensure your work is correct.  
# they will throw errors if your solution is wrong. 
# uncomment to use them

x = np.array([1, 1, 1])
Ax_expected = np.array([1, 2, 100])
np.testing.assert_array_equal(A @ x, Ax_expected)

x = np.array([1, 1])
Bx_expected = np.array([0, np.sqrt(2)])
np.testing.assert_array_equal(B @ x, Bx_expected)

# ## Transformations and the Linear Perceptron
# 
# Lets see how scaling impacts the Linear Perceptron.  We could scale our data set by a matrix multiplication, but that would give away the solution above.  Instead, we'll just re-define each of the columns:

# In[8]:


x, label = get_data(n_per_grp=100)
# modify the constants .1 and 1000 below to play with this yourselves
x[: ,1] = x[:, 1] * .1
x[: ,2] = x[:, 2] * 1000

# choose a random w
w = np.random.standard_normal(3)

scatter_data(x, label)
for _ in range(100):
    w = update_perceptron(x, label, w)
plot_w(w, x)
print(f'first 5 rows of x: ')
print(x[:5, :])


# In[9]:


x, label = get_data(n_per_grp=100)
x[: ,1] = x[:, 1] * 1000
x[: ,2] = x[:, 2] * .1

# choose a random w
w = np.random.standard_normal(3)

scatter_data(x, label)
for _ in range(100):
    w = update_perceptron(x, label, w)
plot_w(w, x)

print(f'first 5 rows of x: ')
print(x[:5, :])


# ### P 2.4: Transformations and the Linear Perceptron (4 pts)
# 
# Whats happening here?  Why is the perceptron focusing on only one feature at a time?

# In[10]:


p24 = """ insert answer here """


# ### P 2.5: Transformations and the Linear Perceptron (up to 3 pts extra credit)
# 
# What pre-processing step might you take to ensure that you don't have this problem in a real dataset?  
# 
# HINT: 
# Consider the following problem: build a Linear Perceptron which distinguishes romantic comedies from horror movies.  Like above, your dataset is 2-dimensional (3 if you count the constant 1).  Your features are the gross ticket sales, in dollars (a value typically in the millions) and running time of the movie, in hours.  Your pre-processing step is sorely needed for this problem!

# In[11]:


p25 = """ insert answer here (or leave as is)"""

