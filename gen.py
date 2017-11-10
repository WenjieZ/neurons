#!/usr/bin/env python
"""
Description: Generate numpy .npy files.

Usage: ./gen.py  config.json  train_size  test_size  noise_level  seed
Example: ./gen.py config.json 1000 1000 0 0

Warning: The directory used to store the data must exist already.
"""

import json
import sys
import numpy as np
import func

## catch command line arguments
with open(sys.argv[1]) as config_file:    
    par = json.load(config_file)    
N1, N2, sigma, seed = (int(i) for i in sys.argv[2:])
N = N1 + N2


## catch problem parameters
n, m = par['xDim'], par['yDim']
f = getattr(func, par['funcName'])

## generate data
np.random.seed(seed)
X = np.random.randn(N, n)
W = np.random.randn(n, m)
b = np.random.randn(m)
Y = f(np.dot(X, W)) + np.tile(b, (N, 1))
if sigma > 0:
    Y += np.std(Y) * sigma * np.random.randn(N, m)
    
## save data
np.save(par['trainInputPath'], X[0:N1, :])
np.save(par['trainOutputPath'], Y[0:N1, :])
np.save(par['testInputPath'], X[N1:, :])
np.save(par['testOutputPath'], Y[N1:, :])
