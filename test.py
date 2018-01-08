import numpy, libreg
X = numpy.array ([[1 , 1] , [ -1 , 1] , [ -1 , -1] , [1 , -1]])
y = numpy.array ([0.8 , 1.2 , 1.1 , 1.4])
# X = numpy.array ([[2, 8, 25] , [1, 4, 78] , [5, 3, 42] , [1, 2, 30]])
# y = numpy.array ([1,3,4,6])
lm = libreg.LinearModel()
lm.train(X, y)

print("coefs :" , lm.coefs)
print("prediction at (4, 1):", lm.predict(numpy.array([4 ,1])))
print("sse :", round(lm.sse, 2))
print("sst :", round(lm.sst, 2))
print("r2 on train :", round(lm.r2(X, y), 2))
print("r2 on test :", round(lm.r2(X, y + 0.1), 2))