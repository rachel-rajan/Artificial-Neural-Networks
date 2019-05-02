function pred_class = determine_output(Theta1, Theta2, X)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
pred_class = zeros(size(X, 1), 1);

h1 = compute_sigmoid([ones(m, 1) X] * Theta1');
h2 = compute_sigmoid([ones(m, 1) h1] * Theta2');
[~, pred_class] = max(h2, [], 2);

end