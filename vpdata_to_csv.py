#! /usr/bin/env python2

# harrisonn griffin 2019
# @harryturr

import numpy as np
import pandas as pd
import glob
import os
import sys

filename=sys.argv[1]
print(filename)

# loading vpdata with numpy
df=np.loadtxt(filename, skiprows= %s) % # depends on header size

# converting to pandas dataframe and exporting
df = pd.DataFrame(df)
df.to_csv(filename + '.csv', index=False, header=False)

