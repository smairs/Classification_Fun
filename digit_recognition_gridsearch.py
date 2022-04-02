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

# Exhaustive search over specified parameter values for an estimator.
# In this case, we will be usinv a support vector classifier
# so the important parameters are gamma and C (see below)
from sklearn.model_selection import GridSearchCV

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
    With an added gridsearch component for model optimisation

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
    results_text_file = open('optimal_class_report_{}_percent_test.txt'.format(int(test_frac*100)),"w")


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

    # To define a Support vector classifier with some gamma, call:
    # clf = svm.SVC(gamma=0.001,C=100)
    # But here, we are going to gridsearch over C and gamma to find the optimal solution.
    # The GridsearchCV (CV = cross-validation) function itself will become the classifier instance.

    grid_params = {'gamma': np.linspace(0.0001,0.01,10), 'C': np.linspace(1, 1000,10)}
    svc = svm.SVC()
    clf = GridSearchCV(svc, grid_params,verbose=3)

    # Split data into train and test subsets. Shuffle = Whether or not to shuffle the data before splitting.
    # y_test is the target value for fraction of data in the test set
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=test_frac, shuffle=False)

    # Now, train the classifier using the simple function: "fit". The machinery is all
    # under the hood of the original svm.SVC() or GridSearchCV() call to initiate a classifier.
    # The cross-validation is based on a k-fold strategy with default k=5. This means,
    # randomly split the training data into 5 groups of training and validation data (NOTE: TEST DATA
    # IS STILL HELD ASIDE FROM THIS WHOLE PROCESS FOR A FINAL EVALUATION), looping over
    # the parameters of interest for each group, fitting the model, testing the prediction
    # on the validation data, and scoring
    # the result.
    #
    # The general procedure is as follows:
    #
    # 1. Shuffle the training dataset randomly.
    # 2. Split the training dataset into k groups
    # 3. Take 1 of the k groups and use it as validation data while the others are used as a mini-training set
    #    (First *split*)
    # 4. Fit a model on the mini-training set and evaluate it on the validation group
    # 5. Retain the evaluation score and discard the model
    # 6. Now, assign another one of the k groups as your validation set and use the others as a new mini-training set
    #    (Second Split)
    # 7. Repeat over all splits. So 5 groups = 5 splits
    # 8. Loop over all parameter combinations. E.g. 10 gammas, 10 C's, 5-folds = 10*10*5 = 500 fits
    # 9. Search evaluations performed for best parameters overall
    # 10. Use the initially-set aside final testing group to perform final evaluation of the model
    clf.fit(X_train, y_train)

    # This generates a "clf.cv_results_" dict of numpy (masked) ndarrays
    # This is a dict with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
    # Columns will include param_gamma and param_C along with test scores.
    # You can also see the "best_estimator_" by using clf.best_estimator_ or the best_index_
    # So, best_index_ is an integer that can be used as follows:
    #
    # The index (of the cv_results_ arrays) which corresponds to the best candidate parameter setting.
    #
    # The dict at clf.cv_results_['params'][search.best_index_] gives the parameter setting for the
    # best model, that gives the highest mean score (clf.best_score_).
    #
    print('\nBest parameters found:\n\n{}'.format(clf.cv_results_['params'][clf.best_index_]))
    results_text_file.write('Best parameters found:\n\n{}'.format(clf.cv_results_['params'][clf.best_index_]))


    # Now, simply predict the value of the digit on the test subset -- this will automatically use the
    # best parameters found in the grid search!
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
    results_text_file.write(f"\n\nClassification report for classifier {clf}:\n\n"
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
    plt.savefig('optimal_ConfusionMatrix_{}_percent_test.png'.format(int(test_frac*100)),dpi=300)
    plt.clf()
    print(f"\n\nConfusion matrix:\n{disp.confusion_matrix}")
    results_text_file.write(f"\n\nConfusion matrix:\n\n{disp.confusion_matrix}")
    results_text_file.close()


digits = datasets.load_digits()
digit_classification(digits,0.5,visualise=True)