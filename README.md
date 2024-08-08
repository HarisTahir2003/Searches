# Hi, I'm Haris! 👋


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) 


# Searches Assignment (Artificial Intelligence)

This Jupyter Notebook explores the concepts of Searches in Artificial Intelligence, with a particular focus on three search algorithms (and slight variations of each):
<ol>
<li> A* </li>
<li> Best-first search </li>
<li> Breadth-first search </li>
</ol> <br>

The notebook is structured to provide a comprehensive understanding of these search algorithms, and includes practical implementations, visualizations, and evaluations of the algorithms. <br> 

The Searches folder contains the following files:
- A .ipynb file (Jupyter Notebook) that contains all the code regarding the assignment including text blocks explaining portions of the code
- A corresponding .py file
- a png file of a screenshot of how visualizations are used in the Jupyter Notebook

## Table of Contents

1. [Introduction](#introduction)
2. [Installation Requirements](#installation-requirements)
3. [Project Structure](#project-structure)
4. [Training and Evaluation](#training-and-visualization)
5. [Lessons](#lessons)
6. [Screenshots](#screenshots)
   
## Introduction

Logistic regression is a Machine Learning technique used for binary or multi-class classification that models the relationship between a dependent binary/multi-class variable and one or more independent variables. Unlike linear regression, which predicts continuous outcomes, logistic regression predicts the probability of an outcome.

 This assignment provides a clear and concise example of how to implement multi-class logistic regression from scratch using Python.
 
## Installation Requirements

To run this notebook, you will need the following packages:
- osmnx
- Iframe
- networkx
- folium
- display
- matplotlib
- numpy
- math
- queue

You can install these packages using pip:

```bash
 import osmnx as ox
```
```bash
 from IPython.display import IFrame
```
```bash
 import networkx as nx
```
```bash
import folium
```
```bash
from IPython.display import display
```
```bash
import matplotlib.pyplot as plt
```
```bash
 import numpy as np
```
```bash
 from math import radians, sin, cos, sqrt, atan2
```
```bash
 import queue
```
<strong> Make sure to run the first command in the code block, i.e run all other commands in your terminal whereas the first command should be run in the Jupyter Notebook code block. </strong> <br>
Useful Links for installing Jupyter Notebook:
- https://youtube.com/watch?v=K0B2P1Zpdqs  (MacOS)
- https://www.youtube.com/watch?v=9V7AoX0TvSM (Windows)

It's recommended to run this notebook in a conda environment to avoid dependency conflicts and to ensure smooth execution.
<h4> Conda Environment Setup </h4>
<ul> 
   <li> Install conda </li>
   <li> Open a terminal/command prompt window in the assignment folder. </li>
   <li> Run the following command to create an isolated conda environment titled AI_env with the required packages installed: conda env create -f environment.yml </li>
   <li> Open or restart your Jupyter Notebook server or VSCode to select this environment as the kernel for your notebook. </li>
   <li> Verify the installation by running: conda list -n AI_env </li>
   <li> Install conda </li>
</ul>


## Project Structure

The notebook is organized into the following sections:
<ul>
<li> Problem Description: describes the purpose of the assignment </li> <br> 
<li> Details about the osmnx and folium python libraries and how they are to be used in the assignment.  </li> <br> 
<li> Three functions that were already provided in the Jupyter Notebook: 
   <ol>
      <li> Road Network Data Retrieval Function (get_road_network) </li>
      <li> Road Network Visualization Function (visualize_road_network) </li>
      <li> Shortest Path Visualization Function (visualize_path_folium) </li>
   </ol> 
</li> <br> 
<li> Mathematical details of the Eulidean, Manhattan and Haversine distances, that are used in the implementation of heuristic functions </li> <br> 
<li> Task 0: Implementing Heuristics --- implements functions to calculate the three heuristics </li> <br> 
<li> Task 1: A* Algorithm Implementation --- implements the A* algorithm to find the shortest path between a source node and a target node. </li> <br> 
<li> Task 2: Best-First Search Algorithm Implementation --- implements the Best-First Search algorithm to find a path between a source node and a target node.</li> <br> 
<li> Task 3: Informed Breadth-First Search Algorithm Implementation --- implements the Breadth-First Search algorithm using heuristics to find a path between a source node and a target node.</li> <br> 
<li> Visual representation of the maps showing the various paths identified by the algorithms </li> <br> 
<li> Implementation of the A* Algorithm using the built-in function of the NetworkX library, along with its visual map representation </li> <br> 
<li> Task 4: Single Source And Multiple Destinations using A star --- implements the A* Algorithm such that it can identify the optimal paths for MULTIPLE destinations, along with its visual map representation. </li> <br> 
<li> Brief comparative analysis of the different algorithms used in the assignment to implement the searches. </li> <br> 
</ul>


## Training and Visualization

The entire training process alongside the maths involved is explained in detail in the jupyter notebook. 
- Note: You need to be proficient in Python Programming to fully implement the complex algorithms shown in this assignment.

## Lessons

A logistic regression project can teach a variety of valuable skills and concepts, including:

- Data Preprocessing: How to clean and prepare data for analysis, including handling missing values, scaling features, and encoding categorical variables.

- Feature Selection: Identifying which features (variables) are most important for making predictions and how to choose them effectively.

- Model Building: Understanding how to build a logistic regression model, including splitting data into training and testing sets, fitting the model, and predicting outcomes.

- Performance Evaluation: Using metrics like Root Mean Squared Error (RMSE) to evaluate the performance of your model and understand its accuracy.

- Interpreting Results: Understanding the results of the logistic regression model and what they signify.

- Algorithm Implementation: Learning about the underlying algorithm used in linear regression and how it optimizes the line of best fit.


## Screenshots
<h3> Ridge Regression </h3>
<h4> 1. This image shows how the value of the Root-Mean-Square-Error changes for various training and testing datasets as the value of the regularization parameter (lambda) is gradually increased from 0 to 10. The four datasets include the training and testing datasets of each of the analytical and gradient-descent solutions. </h4>
<img src="pic1.png" width="450px"> <br> 







## License

[MIT](https://choosealicense.com/licenses/mit/)
