% MATLAB Starter Code Demo

% --- 1. Inputs (Clear environment and define initial variables) ---

% Use 'clc' to clear the command window and 'clear' to clear the workspace.
clc;
clear;

% Define variables using meaningful names.
% Semicolons suppress the output in the command window.
length_val = 10; % Length in meters
width_val = 5;   % Width in meters

% Create a vector (one-dimensional array).
data_points = [1, 2, 3, 4, 5, 6];

% Create a matrix (two-dimensional array), rows separated by semicolons.
matrix_A = [1, 2, 3; 
            4, 5, 6; 
            7, 8, 9];

% --- 2. Process (Perform calculations) ---

% Calculate the area.
area = length_val * width_val;

% Perform element-wise operations on the vector. The '.' operator is for element-wise ops.
squared_data = data_points.^2; 

% Calculate the roots of a polynomial (example from MathWorks documentation).
% Represents the polynomial s^4 + 3s^3 - 15s^2 - 2s + 9.
coefficients = [1, 3, -15, -2, 9];
roots_poly = roots(coefficients);

% --- 3. Output (Display results and visualize data) ---

% Display variables in the Command Window by omitting the semicolon.
disp("--- Results ---");
fprintf("The calculated area is: %.2f square meters.\n", area);

% Display the squared data points.
disp("Squared data points:");
disp(squared_data);

% Display the roots of the polynomial.
disp("Roots of the polynomial:");
disp(roots_poly);

% Create a simple 2D plot to visualize data.
figure; % Open a new figure window
plot(data_points, squared_data, '-o'); % Plot with lines and markers
title("Data Points vs. Squared Data Points");
xlabel("Original Data");
ylabel("Squared Data");
grid on;
