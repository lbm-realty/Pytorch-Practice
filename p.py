x = 4
y = x**2

w = 0.1
for i in range(100000):
    prediction = w * x 
    cost = y - prediction
    derivative = cost / w
    w = w - 0.0001 * derivative
print(derivative, w)