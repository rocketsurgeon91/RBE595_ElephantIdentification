%% Cleaning MATLAB
function main_rev4()
%% Select Validation or Training
type_list = {'training','test'};
aug_images_dir = 'augmented_images';
aug_dir{1} = ['AED\',aug_images_dir,'\training\'];
aug_dir{2} = ['AED\',aug_images_dir,'\test\'];
file_name_output{1} = 'aug_training_images.csv';
file_name_output{2} = 'aug_testing_images.csv';
for ii = 1:length(type_list)
    type = type_list{ii};
    %% Import Training Images Data
    fileID = fopen(['AED/',type,'_images.csv'],'r');
    images.raw_scan = textscan(fileID,'%s %f %f %f %f %f %f %f','delimiter',',','collectoutput',1);
    images.image_ids = images.raw_scan{1,1};
    images.sortie_ids = images.raw_scan{1,2}(:,1);
    images.image_widths = images.raw_scan{1,2}(:,2);
    images.image_heights = images.raw_scan{1,2}(:,3);
    images.gsds = images.raw_scan{1,2}(:,4);
    images.alts = images.raw_scan{1,2}(:,5);
    images.agls = images.raw_scan{1,2}(:,6);
    images.gps_alts = images.raw_scan{1,2}(:,7);
    fclose(fileID);
    %% Import Training Elephants Data
    fileID = fopen(['AED/',type,'_elephants.csv'],'r');
    elephants.raw_scan = textscan(fileID,'%s %f %f','delimiter',',','collectoutput',1);
    elephants.image_ids = elephants.raw_scan{1,1};
    elephants.coordinates = [elephants.raw_scan{1,2}(:,1), elephants.raw_scan{1,2}(:,2)];
    fclose(fileID);
    %% Importing Images into MATLAB
    image_files = dir(['AED/',type,'_images/*.jpg']); % number of jpg images in dir
    n_files = length(image_files); % number of files found
    %% Defining the size of the Cropped Image
    u = 224;
    v = 224;
    counter = 0;
    for jj = 1:n_files
        current_filename = image_files(jj).name;
        % Lookups from data
        training_lookup_logical = strcmp(elephants.image_ids, current_filename(1:end-4));
        file_lookup_logical = strcmp(images.image_ids, current_filename(1:end-4));
        % determining the elephant image coordinates
        elephant_image_coordinate = elephants.coordinates(training_lookup_logical,:);
        image_width = images.image_widths(file_lookup_logical,:);
        image_height = images.image_heights(file_lookup_logical,:);
        % determining the number of elephants
        size_elephants_coordinates = size(elephant_image_coordinate);
        number_elephants = size_elephants_coordinates(1);
        file_name_list{jj} = [cd,'\AED\',type,'_images\', current_filename];
        counter = 0;
        for i = 1:number_elephants
            if elephant_image_coordinate(i,1)-floor(u/2) > 0 && elephant_image_coordinate(i,2)-floor(v/2) > 0 && elephant_image_coordinate(i,1)-floor(u/2) + (u-1) < image_width && elephant_image_coordinate(i,2)-floor(v/2) + (v-1) < image_height % if the bounding box is completely contained in the image
                counter = counter + 1;
                imcrop_size_list{jj}(counter,1:4) = [elephant_image_coordinate(i,1)-floor(u/2), elephant_image_coordinate(i,2)-floor(v/2), u, v];
            end
        end
    end
    if strcmp(type,'training')
        training_table = cell2table(horzcat(file_name_list',imcrop_size_list'),'VariableNames',{'imageFilename','elephant'});
    else
        test_table = cell2table(horzcat(file_name_list',imcrop_size_list'),'VariableNames',{'imageFilename','elephant'});
    end
    clear file_name_list imcrop_size_list
    save([type,'_table'], [type,'_table']);
end
addpath(genpath(aug_dir{1}));
if exist(aug_dir{1},'dir') ~= 7
    mkdir(aug_dir{1});
end
file_name = [aug_dir{1},file_name_output{1}];
if exist(file_name,'file') == 2
    delete(file_name);
end
file_id = fopen(file_name,'w');
fclose(file_id);

num = height(training_table);
parfor i = 1:num
    data_out = augmentData(training_table{i,:});
    imwriteAll(file_name, aug_dir{1}, data_out,i);
    fprintf('iter:\t%d\n',i);
end
fprintf('Parallel For Loop... End\n');

addpath(genpath(aug_dir{2}));
if exist(aug_dir{2},'dir') ~= 7
    mkdir(aug_dir{2});
end
file_name = [aug_dir{2},file_name_output{2}];
if exist(file_name,'file') == 2
    delete(file_name);
end
file_id = fopen(file_name,'w');
fclose(file_id);

num = height(test_table);
parfor i = 1:num
    data_out = augmentData(test_table{i,:});
    imwriteAll(file_name, aug_dir{2}, data_out,i);
    fprintf('iter:\t%d\n',i);
end
fprintf('Parallel For Loop... End\n');

rmpath(aug_dir{1});
rmpath(aug_dir{2});

end

