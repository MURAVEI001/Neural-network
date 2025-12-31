import numpy as np
import time
import math

# array = np.array([[2,3],
#                   [4,5],
#                   [6,7]])
# vector = np.array([2,3])
# output = np.ones(len(array))
# output = np.sum(vector*array,1)
# print(array,array.shape)
# print(vector, vector.shape)
# print(output, output.shape) 

# array = np.ones(10,)
# scalar = 3
# x = array*scalar
# print(x,x.shape)

# start = time.time()
# x = np.zeros(100000)
# print(f"{time.time() - start:.10f}")


# start = time.time()
# y = np.empty(100000)
# print(f"{time.time() - start:.10f}")

# x = 4

# i = np.array([1,2,3])
# w = np.array([[1,2,3],
#              [4,5,6]])
# b = i*w
# print(b,b.shape)

# d0 = np.array([1])
# d1 = np.array([1,2,3])
# d2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
# d3 = np.array([ [[1,2,3],[4,5,6],[7,8,9]],[[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]] ])

# print(d0.shape,d1.shape,d2.shape,d3.shape)
# y = np.array([[1,2,3,4]]).T
# inputs = np.array([1,2,3])
# w_0_1 = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
# w_1_2 = np.array([[1],[2],[3],[4]])

# l1 = np.dot(inputs,w_0_1)
# l2 = np.dot(l1,w_1_2)
# l2_delta = y[0] - l2
# l1_delta = np.dot(l2_delta,w_1_2.T)*l1
# print(l1, l1.shape)
# print(l2, l2.shape)
# print(y,y.shape)
# print(l2_delta,l2_delta.shape)
# print(l1_delta)

# i = np.array([0.5,2]) # 2,
# print(i.shape)
# w1 = np.array([[0.5,0.1],[0.8,0.2]]) # 2,2
# w2 = np.array([[0.1,0.5]]) # 1,2
# y = np.array([10]) # ()

# z1 = np.dot(w1,i) # 2,2 * 2, =  2,
# print(z1,z1.shape)
# z2 = np.dot(w2,i) # 1,2 * 2, 
# print(z2,z2.shape)

print(np.random.randint(2,size=30))