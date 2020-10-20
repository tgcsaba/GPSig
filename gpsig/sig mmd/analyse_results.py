#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:52:18 2020

@author: oberhauser
"""

import pandas as pd
df=pd.read_pickle("df_results.pkl")
df=df['Success']

for dataset,df_dataset in df.groupby(level=['Dataset']):
    print(dataset)
    for experiment,df_stat_hyp in df_dataset.groupby(level=['Dataset','Statistic', 'Hypothesis']):

        results = df_stat_hyp.to_numpy()
        percentage = results.sum()/len(results)
   
        print(experiment[1], experiment[2],':', percentage)
        
    print('\n')
    #    print(df_stat_hyp)
    
    