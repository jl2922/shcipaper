#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import re

rawData = open('data.txt').read()
print(rawData)

rawData = rawData.strip().split('\n')
levelArray = []
nDetsArray = []
sumWeightsArray = []
for row in rawData:
    row = re.split(r'\s+', row.strip())
    level = int(row[0])
    nDets = float(row[1])
    sumWeights = float(row[2])
    print(level, nDets, sumWeights)
    levelArray.append(level)
    nDetsArray.append(nDets)
    sumWeightsArray.append(sumWeights)

plt.figure(figsize=(5.5, 4.0))
plt.tight_layout()

ax1 = plt.subplot(211)
nDetsArray = np.log10(np.array(nDetsArray))
plt.plot(levelArray, nDetsArray, 'o-')
plt.grid(True, ls=':')
plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('log$_{10}$(# Determinants)')
plt.title('Contribution from Each Excitation Level in Variation')

ax2 = plt.subplot(212, sharex=ax1)
sumWeightsArray = np.log10(np.array(sumWeightsArray))
plt.plot(levelArray, sumWeightsArray, 'o-')
plt.grid(True, ls=':')
plt.xlabel('Excitation Level from HF')
plt.ylabel('log$_{10}(\Sigma_i c_i^2)$')
plt.subplots_adjust(left=0.185)

plt.savefig('excitation.eps', format='eps', dpi=300)
plt.show()

