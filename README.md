# optimization
Repository of the code for the homework of Optimization for Data Science, academic year 2019/2020, UNIPD.


# issue:
1- The loss function seems to be after few step even if the algorithm in reality has not yet converged (the accuracy goes up even if loss  doen't go down). <br>

2- The linear classifier gives value not bounded between 0-1 or -1,1 so at the end of the fitting procedure i implemeted a rescaling such that the decision boundary is 0 (probably best way could be apply softmax to output to transform it to probability).<br>

3- The step size (a.k.a learning rate) seems to work when it is set high (1, 5, 10) and not low (0.01, 0.005). <br>

4- Tested only on a dataset where the feature are N integer numbers (from 0 to 9) and the target is 1 if X1 + ... + XN is above the mean of the sums and it is 0 if X1 + ... + XN is lower or equal the mean of the sums. <br>
