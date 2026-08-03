# AutoML Pipeline

## Problem

Building a machine learning model involves many repetitive steps, including loading data, validating it, preprocessing features, selecting models, tuning hyperparameters, and evaluating performance. These tasks are often implemented differently for each project, making experimentation slower and reducing code reusability.

## Solution

This project aims to build a modular AutoML pipeline that automates the end-to-end machine learning workflow for tabular datasets. Each stage of the pipeline is designed as an independent, reusable component, making the system easy to extend, test, and maintain while following software engineering best practices.