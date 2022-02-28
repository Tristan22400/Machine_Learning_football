# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 11:05:58 2021

@author: trist
"""

import Functions as f
import matplotlib.pyplot as plt
proba_ranking=[k for k in range (20)]
plt.hist( proba_ranking,bins=20, color = 'blue')
plt.xlabel('Classement')
plt.ylabel('Probabilités')
plt.title('Probabilité de classement à la fin du championnat')