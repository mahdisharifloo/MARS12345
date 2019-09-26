# -*- coding: utf-8 -*-

#Import Library
 
from sklearn import svm,datasets
 
#Assumed you have, X (predictor) and Y (target) for training data set and x_test(predictor) of test_dataset
 
# Create SVM classification object
 
model = svm.SVC(kernel='linear', C=1, gamma=1)
 
# there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score

# import iris data 
iris = datasets.load_iris()

x = iris.data[:,:2]

y = iris.target

 
model.fit(X, y)
 
model.score(X, y)
 
#Predict Output
 
predicted= model.predict(x_test)


