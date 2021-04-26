function data = preprocessData(data,targetSize)
% numRows = size(data,1);
% sz = size(data{1},[1 2]);
% for idx = 1:numRows
%     % Resize image and bounding boxes to targetSize.
%     scale = targetSize(1:2)./size(data{idx,1},[1 2]);
%     data{idx,1} = imresize(data{idx,1},targetSize(1:2));
%     % Sanitize box data, if needed.
%     data{idx,2} = helperSanitizeBoxes(data{idx,2}, sz);
%     data{idx,2} = bboxresize(data{idx,2},scale);
% end

[m, n] = size(data{1},[1 2]);
p = targetSize(1);
q = targetSize(2);
height_pad = ceil((p-m)/2);
width_pad = ceil((q-n)/2);
elephant_coordinates = data{2}; 
new_elephant_coordinates = zeros(size(elephant_coordinates)); 
num_elephants = size(elephant_coordinates,1); 
for i = 1:num_elephants
    new_elephant_coordinates(i,:) = [(elephant_coordinates(i,1)+width_pad), (elephant_coordinates(i,2)+height_pad), elephant_coordinates(i,3), elephant_coordinates(i,4)];
end
data{1} = padarray(data{1}, [height_pad width_pad], 0, 'both');
while size(data{1},1) > targetSize(1) 
    data{1} = data{1}(1:end-1,:,:); 
end
while size(data{1},2) > targetSize(2) 
    data{1} = data{1}(:,1:end-1,:); 
end
data{2} = new_elephant_coordinates; 

end