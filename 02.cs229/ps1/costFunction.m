%% Average empirical loss for logistic regression
%
% Chul-Woong Yang
% cwyang@github.com

function [J, grad, hess] = costFunction(theta, X, y)

  m = length(y); % number of samples
  n = size(theta);
  J = 0;
  grad = zeros(n);
  hess = zeros(n,n);

  h = sigmoid(X*theta.*y);
  J = -(1/m)*sum(log(h));
  
  grad = -(1/m)*sum((1-h).*y.*X);
  grad = grad(:);  % to colum vector
  
  for i = 1:m
    x = X(i,:)';
    h = sigmoid(theta'*x);
    hess = hess + h*(1-h)*(x*x');
  end
  hess = hess / m;
end

function g = sigmoid(z)
  g = 1 ./ (1+exp(-z));
end
