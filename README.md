# Credit-Card-Fraud-Detection-Using-ML
Analyzed the datasheet with 284807 and then taking 10% of total transaction which is 28481 transaction using Unsupervised learning.
Algorithms used Local Outlier Factor and Isolation Forest Algorithm.
Packages used like sys, numpy, pandas, matplotlib, seaborn, scipy, sklearn.
Imported the dataset of 284807 transaction whihc i downloaded from Kaggle.com.
Explored the dataset and print the column.
There are 2 classses, Class 1 for fraud transaction and Class 0 for valid transaction.
Plotting the histogram i was able to find out the amound withdrawn.
Then I printed the outlier factor, fraud transaction out of total transaction and valid transaction.
aftre this i got into corelation matrix and plotted the heat map of corelation matrix.
aftre that i tried to get all columns of my dataset.
then i removed the classes which were specifying the valid and fraud transaction making it a unsupervised learning.
then came the main part using the algorithms, importing Isolation Forest and Local Outlier Factor from sk learn.
classifying the detection tools to be used and fiting the model i predicted the Isolation Forest and Local Outlier Factor.
Local Outlier Factor: 97 (0.9965942207085425)
Isolation Forest: 71 (0.99750711000316)
