clc;
clear;
close all;

unmodified_training_data = table2dataset(readtable("data\CreditCardClientsTrainDataSimple.xls"));
unmodified_training_labels = table2dataset(readtable("data\CreditCardClientsTrainLabelSimple.xls"));

%Get just the labels
extracted_labels = unmodified_training_labels(:,2);
extracted_data = unmodified_training_data(:,2:end);
data_with_labels = dataset2table([extracted_data extracted_labels]);
table_variables = data_with_labels.Variables;

%% Let's do some data visualization!!

%Means
data_means = zeros(24,1);
for index = 1:24
    current_col = data_with_labels.(index);
    data_means(index) = (mean(data_with_labels.(index)));
    
end
figure(1);
scatter(data_means);

%% Data is unbalanced, split up based on default status

defaulting_data = data_with_labels(data_with_labels.defaultPaymentNextMonth==1,:);
nondefaulting_data =data_with_labels(data_with_labels.defaultPaymentNextMonth==0,:);

%% 50/50 test

default_portion = defaulting_data(1:5000,:);
nondefualting_portion = nondefaulting_data(1:5000,:);

data_even_split = [default_portion; nondefualting_portion];

randomized_5050= data_even_split(randperm(size(data_even_split, 1)), :);


%% 75% non-defauling | 25% defaulting

nondefaulting_portion = nondefaulting_data(1:15000,:);
data_7525_split = [default_portion; nondefaulting_portion];
randomized_7525 = data_7525_split(randperm(size(data_7525_split, 1)), :);

%% 30% non-defaulting | 70% defaulting
default_portion = defaulting_data(1:5000,:);
nondefaulting_portion = nondefaulting_data(1:1667,:);
data_2575 = [default_portion ; nondefaulting_portion];
randomized_7525 = data_2575(randperm(size(data_2575, 1)), :);

%% Data import
function [dataByColumn1] = importfile(fileToRead1)
%IMPORTFILE(FILETOREAD1)
%  Imports data from the specified file
%  FILETOREAD1:  file to read

%  Auto-generated by MATLAB on 24-Nov-2019 02:29:36

% Import the file
sheetName='Data';
[numbers, strings, raw] = xlsread(fileToRead1, sheetName);
if ~isempty(numbers)
    newData1.data =  numbers;
end

if ~isempty(strings) && ~isempty(numbers)
    [strRows, strCols] = size(strings);
    [numRows, numCols] = size(numbers);
    likelyRow = size(raw,1) - numRows;
    % Break the data up into a new structure with one field per column.
    if strCols == numCols && likelyRow > 0 && strRows >= likelyRow
        newData1.colheaders = strings(likelyRow, :);
    end
end

% Create output variables.
for i = 1:size(newData1.colheaders, 2)
    dataByColumn1.(matlab.lang.makeValidName(newData1.colheaders{i})) = newData1.data(:, i);
end
end

