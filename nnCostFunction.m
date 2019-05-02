function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y)
%2 layer NN
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);

%Initialize J and theta
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%
X=[ones(m,1) X];
a2=compute_sigmoid(X*Theta1');
a2=[ones(size(a2,1),1) a2];
a3=compute_sigmoid(Theta2*a2');
% hx = a3
temp_y=y==[1:num_labels];  
temp_y=temp_y';    
J=sum(sum(-temp_y.*log(a3)-(1-temp_y).*log(1-a3)))*(1/m);

%Backpropagation for m training examples

%Initialise partial derivatives
del1=zeros(size(Theta1)); 
del2=zeros(size(Theta2));

%Forwarded propagation of all 5000 samples 
%matrix of size (5000,25)
z2=X*Theta1';  
a2=compute_sigmoid(z2);
%matrix of size (5000,26)
a2=[ones(m,1) a2]; 
%matrix of size (5000,10)
z3=a2*Theta2'; 
a3=compute_sigmoid(z3);

%Backpropagating the errors
%matrix of size(5000,10)
errk=a3-temp_y'; 
a = (errk*Theta2);
%matrix size (5000,25)
err2=a(:,2:end).*compute_sigmoid_grad(z2); 

%Using matrix multiplication to automaticaly sum up all samples
%matrix of size (10,26)
del2=errk'*a2; 
%matrix of size (25,401)
del1=err2'*X; 

%Finilising gradient
Theta1_grad=(1/m)*del1;
Theta2_grad=(1/m)*del2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


function g = compute_sigmoid_grad(z)
g = zeros(size(z));
g=compute_sigmoid(z).*(1-compute_sigmoid(z));
end

end
