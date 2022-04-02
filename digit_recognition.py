# We'll do some plotting - get NumPy too, for good measure
import matplotlib.pyplot as plt
import numpy as np

# sklearn has some built in datasets, like the
# digits data, these can be accessed via "datasets"
# "svm" = Support vector machine, this is highly preferred
# by many over regression in classification problems as
# it produces significant accuracy with less computation power.
# See image "SVM_quick_description.png" in this directory
# and the accompanying PDF with a more thorough description.
# metrics = a package to test the accuracy of our model
# using f_scores, p_values etc. Generates a report
# More details below.
from sklearn import datasets, svm, metrics

# This function automatically splits our data into
# train and test sets. Ideally, one performs the training
# many times on the same fraction of data, randomising
# which data is actually used for training and which for
# testing.
# More details below
from sklearn.model_selection import train_test_split

def digit_classification(indata,test_frac=0.5,visualise=False):
    '''
    This code takes images of handwritten digits and
    builds a supervised classification model to classify
    which digit was written.
    Based on: https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
    With extra comments and information

    :param indata: Input array of images (will be flattened) for training and testing.
                   *MUST* have attributes "images" and "target". Can be accessed via:
                   sklearn.datasets.load_digits()
    :param test_frac: The fraction of data to use for testing e.g. 0.5
    :param visualise: If True, show the first 4 test images along with their predicted values
    :return:
    '''

    # The digits dataset consists of 8x8 pixel images of digits.
    # The "images" attribute of the dataset stores 8x8 arrays of grayscale values for each image.
    # The "target" attribute contains the real digit that each image is supposed to represent

    images = indata.images
    targets = indata.target

    # Define a text file to store the results of the model
    results_text_file = open('optimal_class_report_{}_percent_test_nongridsearch.txt'.format(int(test_frac*100)),"w")

    # To apply a classifier on this data, we need to flatten the images,
    # turning each 2-D array of grayscale values from shape (8, 8) into shape (64,)
    # In this case the number of "samples" = rows will be the number of images we have
    # while the number of "features" = columns are the number of pixels
    n_samples = len(images)
    # In reshape, one shape dimension can be -1. In this case, the value is
    # inferred from the length of the array and remaining dimensions.
    # Should be given in (rows, columns) order
    # For example:
    # a = np.arange(6).reshape((3, 2))
    # print(a)
    # array([[0, 1],
    #        [2, 3],
    #        [4, 5]])
    data = images.copy().reshape((n_samples, -1))

    # Next we will create a classifier. We'll use an SVM = support vector machine
    # see the .png and .pdf in this directory for detailed information.
    # SVC = support vector classifier.
    # When training an SVM with the Radial Basis Function (RBF) kernel,
    # two parameters must be considered: C and gamma. The parameter C, common to all SVM kernels,
    # trades off misclassification of training examples against simplicity of the decision surface.
    # A low C makes the decision surface smooth, while a high C aims at classifying all training
    # examples correctly.
    # "gamma" defines how much influence a single training example has.
    # The larger gamma is, the closer other examples must be to be affected.
    # Proper choice of C and gamma is critical to the SVM’s performance.
    # One is advised to use GridSearchCV with C and gamma spaced
    # exponentially far apart to choose good values.
    # A support vector machine constructs a hyper-plane or set of hyper-planes in a high or
    # infinite dimensional space, which can be used for classification, regression or other tasks.
    # Intuitively, a good separation is achieved by the hyper-plane that has the largest distance
    # to the nearest training data points of any class (so-called functional margin), since in
    # general the larger the margin the lower the generalization error of the classifier.
    # So, optimise for the largest margin.

    # Define a Support vector classifier with some gamma, call:
    clf = svm.SVC(gamma=0.001,C=100)

    # Split data into train and test subsets. Shuffle = Whether or not to shuffle the data before splitting.
    # y_test is the target value for fraction of data in the test set
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=test_frac, shuffle=False)

    # Now, train the classifier using the simple function: "fit". The machinery is all
    # under the hood of the original svm.SVC() or GridSearchCV() call to initiate a classifier
    clf.fit(X_train, y_train)

    # Now, simply predict the value of the digit on the test subset
    predicted = clf.predict(X_test)

    # Now let's look at how well the model did by using some statistical metrics
    # classification_report builds a text report showing the main classification metrics.
    #
    # Summary of below:
    # Precision = What fraction of the test positives are correct?
    # Recall    = What fraction of the true positives did we find?
    # Fscore    = Weighted harmonic mean of precision and recall, 1.0 is perfect. Beta is the
    #             weight in which recall is given over precision. 1.0 = equally weighted (default).
    # Support   = The support is the number of occurrences of each class in the target values.
    #             This shows us if a specific class was underrepresented, for instance
    #             (y_test is the target value for fraction of data in the test set)
    # A true positive  = The predicted value is the same as the true value
    # A true negative  = The predicted value says the image is not a 9 and it isn't
    # A false positive (Type  I error) = The predicted value says the image is a 9 but it isn't (An incorrect positive)
    # A false negative (Type II error) = The predicted value says the image is not a 9 but it is (A missed positive)
    #
    # The precision is the ratio tp / (tp + fp) where tp is the number of true positives and
    # fp the number of false positives. The precision is intuitively the ability of the
    # classifier not to label as positive a sample that is negative.
    # In an imbalanced classification problem with more than two classes, like this one,
    # precision is calculated as the sum of true positives across all classes
    # divided by the sum of true positives and false positives across all classes.
    #
    # Precision = Sum(TruePositives_allclasses) / Sum(TruePositives_allclasses + FalsePositives_allclasses)
    #
    # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and
    # fn the number of false negatives (missed positives). The recall is intuitively the ability of the classifier
    # to find all the positive samples. Recall is a metric that quantifies the number of correct positive predictions
    # made out of all positive predictions that could have been made. Unlike precision that only comments on the
    # correct positive predictions out of all positive predictions, recall provides an indication of missed
    # positive predictions.
    #
    # In an imbalanced classification problem with more than two classes, like the digits example,
    # recall is calculated as the sum of true positives across all classes divided by the sum of true
    # positives and false negatives across all classes.
    #
    # Recall = Sum(TruePositives_allclasses) / Sum(TruePositives_allclasses + FalseNegatives_allclasses)
    #
    # The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall,
    # where an F-beta score reaches its best value at 1 and worst score at 0.
    #
    # The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and
    # precision are equally important.
    #
    #
    # Support   = The support is the number of occurrences of each class in the target values.
    #             This shows us if a specific class was underrepresented, for instance
    #             (y_test is the target value for fraction of data in the test set)
    #
    #  Output averages:
    # 'micro':
    # Calculate metrics globally by counting the total true positives, false negatives and false positives.
    #
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into
    # account.
    #
    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by support (the number of true instances
    # for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is
    # not between precision and recall.


    print(
        f"\n\nClassification report for classifier {clf}:\n\n"
        f"{metrics.classification_report(y_test, predicted)}"
    )
    results_text_file.write(f"Classification report for classifier {clf}:\n\n"
                            f"{metrics.classification_report(y_test, predicted)}")

    # Optionally show the first 4 test images along with their predicted values
    if visualise:
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for axis, image, label in zip(ax, X_test, predicted):
            axis.set_axis_off()
            image = image.reshape(8, 8)
            axis.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
            axis.set_title("Prediction: {}".format(label))
        plt.show()
        plt.clf()
        plt.close(fig)

    # Plot a confusion matrix of the true digit values (Y-axis) and the predicted digit values (X-axis)
    # with each square being the count. For instance 3 times, the true value was 5 and the predicted value was 7
    # The value at (7,5) = 3.
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    plt.savefig('optimal_ConfusionMatrix_{}_percent_test_nongridsearch.png'.format(int(test_frac*100)),dpi=300)
    plt.clf()
    print(f"\n\nConfusion matrix:\n{disp.confusion_matrix}")
    results_text_file.write(f"\n\nConfusion matrix:\n\n{disp.confusion_matrix}")
    results_text_file.close()

digits = datasets.load_digits()
digit_classification(digits,0.5,visualise=True)