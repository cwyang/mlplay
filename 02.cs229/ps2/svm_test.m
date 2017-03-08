% [sparseTestMatrix, tokenlist, testCategory] = readMatrix('spam_data/MATRIX.TEST');

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
Xtest = 1.0 * (Xtest > 0);
%ytest = (2 * testCategory - 1)';
squared_Xtest = sum(Xtest.^2, 2);
m_test = size(Xtest, 1);
gram_Xtest = Xtest * Xtrain'; % !
Ktest = full( exp( -(repmat(squared_Xtest, 1, m_train) ...
                     +repmat(squared_Xtrain', m_test, 1) ...
                     - 2*gram_Xtest) / (2 * tau^2)));

preds = Ktest * avg_alpha;
fprintf(1, 'Test error rate for average alpha: %1.4f\n', ...
        sum(preds .* ytest <= 0) / length(ytest));
test_error =         sum(preds .* ytest <= 0) / length(ytest);
%numTestDocs = size(sparseTestMatrix, 1);
%numTokens = size(sparseTestMatrix, 2);

% Assume svm_train.m has just been executed, and the model trained
% by your classifier is in memory through that execution. You can also assume 
% that the columns in the test set are arranged in exactly the same way as for the
% training set (i.e., the j-th column represents the same token in the test data 
% matrix as in the original training data matrix).

% Write code below to classify each document in the test set (ie, each row
% in the current document word matrix) as 1 for SPAM and 0 for NON-SPAM.

% Note that the predict function for LIBLINEAR uses the sparse matrix 
% representation of the document word  matrix, which is stored in sparseTestMatrix.
% Additionally, it expects the labels to be dimension (numTestDocs x 1).

% Construct the (numTestDocs x 1) vector 'predictions' such that the i-th
% entry of this vector is positive if the predicted class is 1 and negative if
% the predicted class is -1 for the i-th email (i-th row in Xtest) in the test
% set.
%predictions = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE



%---------------


% Compute the error on the test set
%error = sum(ytest .* predictions <= 0) / numTestDocs;
%fprintf(1, 'Test error for SVM: %1.4f\n', error);