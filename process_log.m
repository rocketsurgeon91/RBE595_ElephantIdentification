%% Reading Log Data to Make Plot
clear all; clc; close all;
%% Required Inputs:
output_prefix = '0p00005_';
total_epochs = 100;
log_file_name = 'console_output_27.txt';
%% Main Code:
data = parseLog(log_file_name);
%% Plotting Code
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.time,'-o'); grid on;
    ylabel('Execution Time [sec]'); axis tight;
    xlabel('Number of Epochs');
    saveas(fig,[output_prefix,'exe_time.jpg']);
catch
    fprintf('Unable to created Execution Time Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.rpn_mean,'-o'); grid on;
    ylabel({'Mean overlapping bounding boxes from RPN',...
        ['for the last: ',num2str(data.prev_iter(end)),' iterations']});
    axis tight;
    xlabel('Number of Epochs');
    saveas(fig,[output_prefix,'mn_rpn.jpg']);
catch
    fprintf('Unable to created RPN Overlap Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.mnbb_rpn,'-o'); grid on;
    ylabel({'Mean number of ',...
        'bounding boxes from ',...
        'RPN overlapping ground truth boxes'});
    xlabel('Number of Epochs');
    axis tight;
    saveas(fig,[output_prefix,'mnbb.jpg']);
catch
    fprintf('Unable to created MNBB Time Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.class_acc_rpn,'-o'); grid on;
    ylabel({'Classifier Accuracy',...
        ' for bounding boxes from RPN'});
    xlabel('Number of Epochs'); axis tight;
    saveas(fig,[output_prefix,'class_acc.jpg']);
catch
    fprintf('Unable to created Classifier Accuracy Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.loss_rpn_reg,'-o'); grid on; hold on;
    plot(data.epochs*total_epochs,data.loss_det_class,'-o'); grid on;
    plot(data.epochs*total_epochs,data.loss_det_reg,'-o'); grid on;
    xlabel('Number of Epochs'); axis tight;
    ylabel('Loss Types');
    legend('Loss of RPN Regression','Loss of Detector Class','Loss of Detector Regression');
    saveas(fig,[output_prefix,'loss_types.jpg']);
catch
    fprintf('Unable to created Losses Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs*total_epochs,data.loss_rpn_reg+data.loss_det_class+data.loss_det_reg,'-o'); grid on; hold on;
    xlabel('Number of Epochs'); axis tight;
    ylabel('Total Loss');
    saveas(fig,[output_prefix,'sum_of_losses.jpg']); close all;
catch
    fprintf('Unable to created Total Loss Plot\n');
    close all;
end
pause(1); close all;
try
    fig = figure('units','normalized','outerposition',[0 0 0.5 0.5]);
    plot(data.epochs_num, data.decrease_to,'-o'); grid on;
    ylabel('Total Loss');
    xlabel('Number of Epochs');
    axis tight;
    saveas(fig,[output_prefix,'decrease.jpg']);
catch
    fprintf('Unable to created Total Change throughout Epochs Plot\n');
    close all;
end
pause(1); close all;
fprintf('End\n');

function data = parseLog(file_name)
raw_data = lineByLine(file_name);
count_epoch = 0;
count_rpn = 0;
count_prev_iter = 0;
count_mnbb_rpn = 0;
count_class_acc_rpn = 0;
count_loss_rpn_class = 0;
count_loss_rpn_reg = 0;
count_loss_det_class = 0;
count_loss_det_reg = 0;
count_elapse_time = 0;
count_epoch_num = 0;
count_decr_from = 0;
count_decr_to = 0;

for i = 1:length(raw_data)-1
    %     fprintf([raw_data{i},'\n'])
    if ~isempty(regexp(raw_data{i}, '(?<=Epoch\s\{).*(?=\})', 'match'))
        count_epoch = count_epoch + 1;
        data.epochs(count_epoch) = str2double(regexp(raw_data{i}, '(?<=Epoch\s\{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=RPN\s\=\s\{).*(?=\}\sfor)', 'match')) % Average number of overlapping bounding boxes from RPN = {0.96} for {100} previous iterations
        count_rpn = count_rpn + 1;
        data.rpn_mean(count_rpn) = str2double(regexp(raw_data{i}, '(?<=RPN\s\=\s\{).*(?=\}\sfor)', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=\}\sfor\s\{).*(?=\}\sprevious\siterations)', 'match')) % Average number of overlapping bounding boxes from RPN = {0.96} for {100} previous iterations
        count_prev_iter = count_prev_iter + 1;
        data.prev_iter(count_prev_iter) = str2double(regexp(raw_data{i}, '(?<=\}\sfor\s\{).*(?=\}\sprevious\siterations)', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=boxes: \{).*(?=\})', 'match'))
        count_mnbb_rpn = count_mnbb_rpn + 1;
        data.mnbb_rpn(count_mnbb_rpn) = str2double(regexp(raw_data{i}, '(?<=boxes: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=RPN: \{).*(?=\})', 'match'))
        count_class_acc_rpn = count_class_acc_rpn + 1;
        data.class_acc_rpn(count_class_acc_rpn) = str2double(regexp(raw_data{i}, '(?<=RPN: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=RPN\sclassifier: \{).*(?=\})', 'match'))
        count_loss_rpn_class = count_loss_rpn_class + 1;
        data.loss_rpn_class(count_loss_rpn_class) = str2double(regexp(raw_data{i}, '(?<=RPN\sclassifier: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=RPN\sregression: \{).*(?=\})', 'match'))
        count_loss_rpn_reg = count_loss_rpn_reg + 1;
        data.loss_rpn_reg(count_loss_rpn_reg) = str2double(regexp(raw_data{i}, '(?<=RPN\sregression: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=Detector\sclassifier: \{).*(?=\})', 'match'))
        count_loss_det_class = count_loss_det_class + 1;
        data.loss_det_class(count_loss_det_class) = str2double(regexp(raw_data{i}, '(?<=Detector\sclassifier: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=Detector\sregression: \{).*(?=\})', 'match'))
        count_loss_det_reg = count_loss_det_reg + 1;
        data.loss_det_reg(count_loss_det_reg) = str2double(regexp(raw_data{i}, '(?<=Detector\sregression: \{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=time:\s\{).*(?=\})', 'match'))
        count_elapse_time = count_elapse_time + 1;
        data.time(count_elapse_time) = str2double(regexp(raw_data{i}, '(?<=time:\s\{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=\{).*(?=\}Total)', 'match'))
        count_epoch_num = count_epoch_num + 1;
        data.epochs_num(count_epoch_num) = str2double(regexp(raw_data{i}, '(?<=\{).*(?=\}Total)', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=decreased\sfrom\s\{).*(?=\})', 'match'))
        count_decr_from = count_decr_from + 1;
        data.decrease_from(count_decr_from) = str2double(regexp(raw_data{i}, '(?<=decreased\sfrom\s\{).*(?=\})', 'match'));
    end
    if ~isempty(regexp(raw_data{i}, '(?<=\}\sto\s\{).*(?=\})', 'match'))
        count_decr_to = count_decr_to + 1;
        data.decrease_to(count_decr_to) = str2double(regexp(raw_data{i}, '(?<=\}\sto\s\{).*(?=\})', 'match'));
    end
end
end

function raw_data = lineByLine(filename)
fid = fopen(filename,'r');
counter = 1;
raw_data{counter} = fgetl(fid);
while ischar(raw_data{counter})
    counter = counter + 1;
    raw_data{counter} = fgetl(fid);
end
fclose(fid);
raw_data{end} = [];
end