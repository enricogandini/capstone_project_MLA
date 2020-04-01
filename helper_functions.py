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

import pandas as pd



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