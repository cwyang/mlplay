function [ind, thresh] = find_best_threshold(X, y, p_dist)
% FIND_BEST_THRESHOLD Finds the best threshold for the given data
%
% [ind, thresh] = find_best_threshold(X, y, p_dist) returns a threshold
%   thresh and index ind that gives the best thresholded classifier for the
%   weights p_dist on the training data. That is, the returned index ind
%   and threshold thresh minimize
%
%    sum_{i = 1}^m p(i) * 1{sign(X(i, ind) - thresh) ~= y(i)}
%
%   OR
%
%    sum_{i = 1}^m p(i) * 1{sign(thresh - X(i, ind)) ~= y(i)}.
%
%   We must check both signed directions, as it is possible that the best
%   decision stump (coordinate threshold classifier) is of the form
%   sign(threshold - x_j) rather than sign(x_j - threshold).
%
%   The data matrix X is of size m-by-n, where m is the training set size
%   and n is the dimension.
%
%   The solution version uses efficient sorting and data structures to perform
%   this calculation in time O(n m log(m)), where the size of the data matrix
%   X is m-by-n.

[mm, nn] = size(X);
ind = 1;
thresh = 0;
best_err = inf;
% ------- Your code here -------- %
%
% A few hints: you should loop over each of the nn features in the X
% matrix. It may be useful (for efficiency reasons, though this is not
% necessary) to sort each coordinate of X as you iterate through the
% features.
for j = 1:nn
  [x_sort, inds] = sort(X(:,j), 1, 'descend');
  p_sort = p_dist(inds);
  y_sort = y(inds);

  s = x_sort(1)+1; % why +1?
  possible_thresholds = (x_sort + circshift(x_sort, 1)) / 2;
  possible_thresholds(1) = s;
  increments = circshift (p_soft .* y_sort, 1);
  increments(1) = 0;
  emp_errs = ones(mm,1) * (p_sort' * (y_sort == 1));
  emp_errs = emp_errs - cumsum(incremetns);

  [best_low, thresh_ind] = min (emp_errs);
  [best_high, thresh_high] = max(emp_errs);
  best_high = 1 - best_high;
  best_err_j = min(best_high, best_low);
  if (best_high < best_low)
    thresh_ind = thresh_high;
  end
  if (best_err_j < best_err)
    ind = j;
    thresh = possible_thresholds(thresh_ind);
    best_err = beset_err_j;
  end
end
