%% Initialization
clear ; 
close all; 
clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% loading and Visualizing Data 

% Load Training Data
load('Sample_MNIST.mat');
m = size(X, 1);

% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);
displayData(X(sel, :));

%% Initialize Parameters 

% Initialize the weights into variables Theta1 and Theta2
eps = 0.12;
%Initialize theta_1
Theta1 = rand(hidden_layer_size, 1+input_layer_size)*(2*eps)-eps;
%Initialize theta_2
Theta2 = rand(num_labels,1+ hidden_layer_size)*(2*eps)-eps;

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% Compute Cost (Feedforward) 

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y);

%% ADVANCED OPTIMIZATION
               
% Initializing Pameters 

initial_Theta1 = rand(hidden_layer_size, 1+input_layer_size)*(2*eps)-eps;
initial_Theta2 = rand(num_labels,1+ hidden_layer_size)*(2*eps)-eps;

% parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Training Neural Network

options = optimset('MaxIter', 55);

% Return Cost and Grad values
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y);

% Learning Process
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%% Visualize Weights 

fprintf('\nVisualizing Neural Network... \n')

figure(2)
displayData(Theta1(:, 2:end));

%% Predict the Output

pred = determine_output(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
