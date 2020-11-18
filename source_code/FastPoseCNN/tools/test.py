"""
import matplotlib.pyplot as plt
plt.figure()
print('Hello')
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure()
plt.savefig('test.png')