function [mu, covariance]=parameters_learn(data)
% maximum likelihood estimation
mu=mean(data);
N=size(data,1);
bias=data-mu;
covariance=bias'*bias/(N);
end