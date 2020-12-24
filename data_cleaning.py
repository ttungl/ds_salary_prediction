#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:03:26 2020

@author: Tung
"""

import pandas as pd 
import numpy as np 

df = pd.read_csv("glassdoor_jobs.csv")

# lowercase columns name
df.columns = map(str.lower, df.columns)
df.columns = [x.replace(" ","_") for x in df.columns]

# drop unnamed column
df = df[[i for i in df.columns if i not in "unnamed:_0"]]

# #output:
# 'job_title', 'salary_estimate', 'job_description',
# 'rating', 'company_name', 'location', 'headquarters', 'size', 'founded',
# 'type_of_ownership', 'industry', 'sector', 'revenue', 'competitors'

# drop invalid values in columns.
df = df[df['salary_estimate']!='-1']

# salary parsing
df['hourly'] = df['salary_estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
df['employer_provided'] = df['salary_estimate'].apply(lambda x: 1 if 'employer provided salary' in x.lower() else 0)

salary = df['salary_estimate'].apply(lambda x: x.split('(')[0])
minus_kd = salary.apply(lambda x: x.replace('K','').replace('$',''))
min_hr = minus_kd.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:',''))

df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary']+df['max_salary'])/2

# company text
df['company_text'] = df.apply(lambda x: x['company_name'] if x['rating']<0 else x['company_name'][:-3], axis=1)

# state field
df['job_state'] = df['location'].apply(lambda x: x.split(',')[1])
df['same_state'] = df.apply(lambda x: 1 if x.location==x.headquarters else 0, axis=1)

# age of company
df['age'] = df['founded'].apply(lambda x: x if x < 1 else 2020-x)

# job desc parsing (languages, ie. python, etc.)
cols = ['python','r-studio','spark','aws','excel','scala']

for col in cols:
	if col == 'r-studio':
		df[col] = df['job_description'].apply(lambda x: 1 if 'r studio' in x.lower() or col in x.lower() else 0)
	else:
		df[col] = df['job_description'].apply(lambda x: 1 if col in x.lower() else 0)

# save the result
df.to_csv("salary_data_cleaned.csv", index=False)




























