%% learning curve

clear;
training_sizes = [50, 100, 200, 400, 800, 1400];

num_sizes=length(training_sizes);
neg_prior=zeros(num_sizes,1);
pos_prior=zeros(num_sizes,1);
for idx=1:num_sizes
  [spmatrix, tokenlist, trainCategory] = readMatrix(sprintf('./spam_data/MATRIX.TRAIN.%d',training_sizes(idx)));
  trainMatrix = full(spmatrix);
  [m,n] = size(trainMatrix);
  numTrainDocs = size(trainMatrix, 1);
  numTokens = size(trainMatrix, 2);
  
  neg = trainMatrix(trainCategory==0, :);
  pos = trainMatrix(trainCategory==1, :);
  
  neg_words(idx,1) = sum(sum(neg));
  pos_words(idx,1) = sum(sum(pos));
  neg_prior(idx,1) = size(neg,1) / numTrainDocs;
  pos_prior(idx,1) = size(pos,1) / numTrainDocs;
  neg_phi(idx,:) = (sum(neg)+1) / (neg_words(idx)+n);
  pos_phi(idx,:) = (sum(pos)+1) / (pos_words(idx)+n);
end

[spmatrix, tokenlist, category] = readMatrix('spam_data/MATRIX.TEST');
testMatrix = full(spmatrix);
numTestDocs = size(testMatrix, 1);
numTokens = size(testMatrix, 2);
output = zeros(num_sizes, numTestDocs);
output = nb_predict(testMatrix, category, neg_prior, pos_prior, neg_phi, pos_phi);

% Compute the error on the test set
y = full(category);
y = y(:);
output = output';
for i=1:num_sizes
  error(i) = sum(y ~= output(:,i)) / numTestDocs;
  fprintf(1, 'Train_size=%4d, Test error: %1.4f\n', training_sizes(i), error(i));
end
%Print out the classification error on the test set
