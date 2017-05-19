#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 16:20:48 2017

@author: danny
"""

file= open('/scratch/danny/processed/features')
y = 0
for x in file:
    y=y+1
file.close()

file= open('/scratch/danny/processed/labels')

z = 0
for x in file:
    z=z+1
file.close()

print (z)
print (y)