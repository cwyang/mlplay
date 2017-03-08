function estimatedCategory = nb_predict(trainMatrix,trainCategory,neg_prior, pos_prior, neg_phi, pos_phi)
% trainMatrix is now a (numTrainDocs x numTokens) matrix.
% Each row represents a unique document (email).
% The j-th column of the row $i$ represents the number of times the j-th
% token appeared in email $i$. 
%
% pi_1, pi_0 are (1 x numTokens) matrix
%
% estimatedCategory should be (numDocs * 1) matrix
  [m,n] = size(trainMatrix);
  pred_num = size(neg_prior,1);
  estimatedCategory=zeros(pred_num,m);
  log_neg_phi = log(neg_phi);
  log_pos_phi = log(pos_phi);
  log_neg_prior = log(neg_prior);
  log_pos_prior = log(pos_prior);

  for i=1:m
    % for comparision, we do not need to compute denominator
    % if xi occurs k times, the probability is p(xi)^k
    % neg_posterior = product(neg_phi .^ trainMatrix(i,:))* neg_prior;
    reps = repmat(trainMatrix(i,:), pred_num, 1);
    log_neg_posterior = sum(reps .* log_neg_phi, 2) + log_neg_prior;
    log_pos_posterior = sum(reps .* log_pos_phi, 2) + log_pos_prior;
    log_neg_posterior;
    log_pos_posterior;
    estimatedCategory(:,i) = log_neg_posterior < log_pos_posterior;
  end
end
