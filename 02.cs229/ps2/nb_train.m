clear
[spmatrix, tokenlist, trainCategory] = readMatrix('./spam_data/MATRIX.TRAIN');
trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

tokens = textread('./spam_data/TOKENS_LIST', "%* %s");
% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 

% tokenlist is a long string containing the list of all tokens (words).
% These tokens are easily known by position in the file TOKENS_LIST

% trainCategory is a (1 x numTrainDocs) vector containing the true 
% classifications for the documents just read in. The i-th entry gives the 
% correct class for the i-th email (which corresponds to the i-th row in 
% the document word matrix).

% Spam documents are indicated as class 1, and non-spam as class 0.
% Note that for the SVM, you would want to convert these to +1 and -1.


% YOUR CODE HERE
[m,n] = size(trainMatrix);

neg = trainMatrix(trainCategory==0, :);
pos = trainMatrix(trainCategory==1, :);

neg_words = sum(sum(neg));
pos_words = sum(sum(pos));

neg_prior = size(neg,1) / numTrainDocs;
pos_prior = size(pos,1) / numTrainDocs;

neg_phi = (sum(neg)+1) / (neg_words+n);
pos_phi = (sum(pos)+1) / (pos_words+n);
%neg_phi = (sum(neg)) / (neg_words);
%pos_phi = (sum(pos)) / (pos_words);

res = nb_predict(trainMatrix, trainCategory, neg_prior, pos_prior, neg_phi, pos_phi);
y = full(trainCategory)(:);
res = res(:);
[y(:) res(:)];
error = sum (y ~= res) / numTrainDocs;
fprintf(1, 'Train error: %1.4f\n', error);

token_score = pos_phi ./ neg_phi;
[foo, idx] = sort(token_score, "descend");
for i=1:5
  fprintf(1, "%d: %s\n", i, tokens(idx(i)){1,1});
end
nb_test
