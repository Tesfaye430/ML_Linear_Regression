import numpy as np
import matplotlib.pyplot as plt

x=[2,4,8,11,14,18,21]
y=[3,5,7,7.5,8,9,12]

# linear regression
b,a = np.polyfit(x,y,1)
x_data1 = np.linspace(0,25,100)
y_data1 = a + b * x_data1

plt.plot(x_data1,y_data1)
plt.scatter(x,y)
plt.title("Degree 1 polynomial fit (linear) : y= a + bx")
plt.show()

# polynomial degree 2
c,b1,a1 = np.polyfit(x,y,2)
x_data2 = np.linspace(0,25,100)
y_data2 = c * (x_data2 ** 2) + b1 * x_data2 + a1
plt.plot(x_data2,y_data2)
plt.scatter(x,y)
plt.title("Degree 2 polynomial fit (quadratic) : y= a + bx + cx^2")
plt.show()

# polynomial degree 3
d,c1,b2,a2 = np.polyfit(x,y,3)
x_data3 = np.linspace(0,25,100)
y_data3 = d * (x_data3 ** 3) + c1 * (x_data3 ** 2) + b2 * x_data3 + a2
plt.plot(x_data3,y_data3)
plt.scatter(x,y)
plt.title("Degree 3 polynomial fit (Cubic) : y= a + bx + cx^2 + dx^3")
plt.show()
