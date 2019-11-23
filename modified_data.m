clc;
clear;
close all;

unmodified_training_data = table2dataset(readtable("data\CreditCardClientsTrainDataSimple.xls"));
unmodified_training_labels = table2dataset(readtable("data\CreditCardClientsTrainLabelSimple.xls"));

%Get just the labels
extracted_labels = unmodified_training_labels(:,2);
extracted_data = unmodified_training_data(:,2:end);

data_with_labels = dataset2table([extracted_data extracted_labels]);


%%
%% Data is unbalanced, need to get a 50/50 split maybe?

defaulting_data = data_with_labels(data_with_labels.defaultPaymentNextMonth==1,:);
nondefaulting_data =data_with_labels(data_with_labels.defaultPaymentNextMonth==0,:);

