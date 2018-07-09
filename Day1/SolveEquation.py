import numpy as np

'''
3 * x0 + x1 = 9
x0 + 2 * x1 = 8
'''

a = np.array([[3,1], [1,2]])
b = np.array([9,8])
print(np.linalg.solve(a, b))

#(2,3)