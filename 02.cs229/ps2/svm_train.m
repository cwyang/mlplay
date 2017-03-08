% Before using this method, set num_train to be the number of training
% examples you wish to read.

[sparseTrainMatrix, tokenlist, trainCategory] = ...
    readMatrix(sprintf('spam_data/MATRIX.TRAIN.%d', num_train));

% Make y be a vector of +/-1 labels and X be a {0, 1} matrix.
ytrain = (2 * trainCategory - 1)';
Xtrain = 1.0 * (sparseTrainMatrix > 0);

% reading solution...
[m_train, n_train] = size(Xtrain);

squared_Xtrain = sum(Xtrain .^ 2, 2);
gram_Xtrain = Xtrain * Xtrain'; % wow
tau = 8;

Ktrain = full( exp( -(repmat(squared_Xtrain, 1, m_train) ...
                      + repmat(squared_Xtrain', m_train, 1) ...
                      - 2 * gram_Xtrain) / (2 * tau^2)));

lambda = 1 / (64 * m_train);
num_outer_loops = 40;
alpha = zeros(m_train,1);
avg_alpha = zeros(m_train, 1);
Imat = eye(m_train);

count = 0;
for ii = 1:(num_outer_loops*m_train)
  count = count + 1;
  ind = ceil(rand * m_train);
  margin = ytrain(ind) * Ktrain(ind, :) * alpha;
  g = -(margin < 1) * ytrain(ind) * Ktrain(:,ind) + ...
      m_train * lambda * (Ktrain(:, ind) * alpha(ind));
  alpha = alpha - g / sqrt(count);
  avg_alpha = avg_alpha + alpha;
end
avg_alpha = avg_alpha / (num_outer_loops * m_train);

% Xtrain is a (numTrainDocs x numTokens) sparse matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents if the j-th token appears in
% email i.

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% For the SVM, we convert these to +1 and -1 to form the numTrainDocs x 1
% vector ytrain.

% This vector should be output by this method
average_alpha = zeros(m_train, 1);

%---------------
% YOUR CODE HERE


%---------------
