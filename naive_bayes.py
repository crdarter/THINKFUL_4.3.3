import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Data file taken from http://www.math.yorku.ca/people/georges/Files/NATS1500/Data/ds/idealwt.txt
# Data file needed no column name manipulation, but if it did...
# Clean up the data file and write it to a new csv file, 'ideal_weight_processed.csv'.
# with open('http://www.math.yorku.ca/people/georges/Files/NATS1500/Data/ds/idealwt.txt', 'r') as f:
    # document = []
    # for line in f:
        # document.append(line.replace("'", ""))
    # f2 = open('ideal_weight_processed.csv', 'w')
    # for line in document:
        # f2.write(line)
    # f2.close()

# Create a DataFrame for the data in the csv file.
df = pd.read_csv('idealwt.csv')

# The histogram for ideal shows 2 distinct peaks. This hints at two different ideals dependent on gender. 
plt.hist(df['actual'], alpha=.3, label='actual', color='b')
plt.hist(df['ideal'], alpha=.3, label='ideal', color='r')
plt.legend(loc='upper right')
plt.show()

# This histogram shows distribution of values for the difference between ideal and actual weights.
plt.hist(df['diff'])
plt.show()

# The greater number of females in the sample makes sense, given the two peaks in the histogram for ideal weight. 
print "number of males: ", len(df.loc[df['sex'] == 'Male'])
print "number of females: ", len(df.loc[df['sex'] == 'Female'])

gender = df['sex']
X = df[['ideal','actual','diff']]

gen_nb = GaussianNB()
gen_nb.fit(X, gender)

# Accuracy of the model is 92%
print "The model's mean accuracy is %f." % (gen_nb.score(X, gender))

# Finding the total number of points.
print "the total number of points in the dataset is %d." % (len(gen_nb.predict_proba(X)))

# finding the predicted values for gender
# Using .predict_proba() will return the probability estimates for the test vector.
predicted_gen = gen_nb.predict(X)

# Where was the model wrong?
print "The Naive Bayes classifier was wrong %d times." % (len(df.loc[df['sex'] != predicted_gen]))
print "These are all the points where the model was wrong:\n"
print df.loc[df['sex'] != predicted_gen]

# Predicting gender based on an actual weight, an ideal weight, and a diff.
print "Someone with an actual weight of 145, an ideal weight of 160, and a diff of -15 is most likely %s." % (gen_nb.predict([160, 145, -15])[0])
print "Someone with an actual weight of 160, an ideal weight of 145, and a diff of 15 is most likely %s." % (gen_nb.predict([145, 160, 15])[0])
