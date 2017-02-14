%% Average empirical loss for logistic regression
%
% Chul-Woong Yang
% cwyang@github.com
%
function ys = weightedRegression(X, y, tau)
  lambdas=X(:,2);
  m = size(X,1);
  ys = zeros(m,1);
  for i=1:m
    W=diag(0.5*exp( -((lambdas-lambdas(i)).^2)/(2*tau^2)));
    theta=(X'*W*X)\X'*W*y;
    ys(i)=[1 lambdas(i)] * theta;
  end
end
