%% learning curve svm

clear;
training_sizes = [50, 100, 200, 400, 800, 1400];

test_errors_svm = zeros(length(training_sizes), 1);
total_svms_to_avg = 5;
[M, tokenlist, category] = readMatrix('spam_data/MATRIX.TEST');
Xtest = M;
ytest = (2*category - 1)';

for train_ind = 1:length(training_sizes)
  for iter = 1:total_svms_to_avg
    num_train = training_sizes(train_ind);
    svm_train;
    svm_test;
    test_errors_svm(train_ind) = test_errors_svm(train_ind) + test_error;
  end
end

test_errors_svm = test_errors_svm / total_svms_to_avg
for i=1:length(training_sizes)
  fprintf(1, 'Train_size=%4d, Test error: %1.4f\n', training_sizes(i), test_errors_svm(i));
end
%Print out the classification error on the test set
