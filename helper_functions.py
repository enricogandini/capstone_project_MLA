#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Created on Wed Apr  1 19:54:56 2020
# Copyright © Enrico Gandini <enricogandini93@gmail.com>
#
# Distributed under terms of the MIT License.

"""
"""

from typing import Dict
from typing import List
from itertools import product

import pandas as pd
from sklearn.base import clone



def various_metrics_binary_classification(model,
                                          metrics: List,
                                          X: pd.DataFrame,
                                          y_true: pd.Series,
                                          identifier: str = None) -> Dict:
    
    if not (isinstance(metrics, List) and any(metrics)):
        raise ValueError("First argument should be a non-empty list!")
        
    pred_proba = model.predict_proba(X)[:, 1]
    
    results = {}
    for metric in metrics:
        result = metric(y_true, pred_proba)
        label_metric = metric.__name__ if not identifier else f"{metric.__name__}_{identifier}"
        results[label_metric] = result
        
    return results


def add_identifier_to_names_pipelines_dict(pipelines: Dict, identifier: str=None):

    if not (isinstance(pipelines, Dict) and any(pipelines)):
        raise ValueError("First argument should be a non-empty dictionary!")


    old_names = list(pipelines.keys()) #(!)otherwise, each key will be changed twice!
    for old_name in old_names:
        new_name = f"{old_name}_{identifier}"
        pipelines[new_name] = pipelines.pop(old_name) #inplace!


    return None


def insert_steps_to_pipelines_dict(steps: List,
                                   pipelines: Dict,
                                   identifier: str = None,
                                   ):

    if not (isinstance(steps, List) and any(steps)):
        raise ValueError("First argument should be a non-empty list!")
    elif not (isinstance(pipelines, Dict) and any(pipelines)):
        raise ValueError("Second argument should be a non-empty dictionary!")


    new_pipelines = {name: clone(pipeline)
                     for name, pipeline in pipelines.items()}

    reverse_steps = steps.copy()
    reverse_steps.reverse() #remember, list.reverse() is inplace!
    for step, new_pipeline in product(reverse_steps, new_pipelines.values()):
        new_pipeline.steps.insert(0, step)


    if identifier:
        add_identifier_to_names_pipelines_dict(new_pipelines, identifier) #this step is inplace


    return new_pipelines
