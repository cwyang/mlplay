function res = productNonzero(mat)
  mat(mat==0) = 1;
  mat = log(mat);
  res = exp(sum(mat,2));
end
