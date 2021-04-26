function data_out = augmentData(data_entry)
numRows = size(data_entry,1);
data_out = cell(numRows*4,2); % original / x reflection / y reflection / xy reflection
for data_idx = 1:numRows
    preprocess_format_data = data_entry; 
    preprocess_format_data{1} = imread(char(data_entry{data_idx,1}));
    padded_data = preprocessData(preprocess_format_data,[3690,5525]);
    data_out{data_idx,1} = padded_data{1};
    data_out{data_idx,2} = padded_data{2};
end
tform1 = affine2d([-1, 0, 0; 0,  1, 0; 0, 0, 1 ]);
tform2 = affine2d([1, 0,  0; 0, -1, 0; 0, 0, 1 ]);

for idx = numRows+1:numRows*2
    data_idx = idx - numRows;
    % Randomly flip images and bounding boxes horizontally.
    sz = size(data_out{data_idx,1});
    rout = affineOutputView(sz,tform1);
    data_out{idx,1} = imwarp(data_out{data_idx,1},tform1,'OutputView',rout);
    % Warp boxes.
    data_out{idx,2} = bboxwarp(data_out{data_idx,2},tform1,rout);
end

for idx = numRows*2+1:numRows*3
    data_idx = idx - numRows*2;
    % Randomly flip images and bounding boxes horizontally.
    sz = size(data_out{data_idx,1});
    rout = affineOutputView(sz,tform2);
    data_out{idx,1} = imwarp(data_out{data_idx,1},tform2,'OutputView',rout);
    % Warp boxes.
    data_out{idx,2} = bboxwarp(data_out{data_idx,2},tform2,rout);
end

for idx = numRows*3+1:numRows*4
    data_idx = idx - numRows*3;
    % Randomly flip images and bounding boxes horizontally.
    sz = size(data_out{data_idx,1});
    rout1 = affineOutputView(sz,tform1);
    rout2 = affineOutputView(sz,tform2);
    step1 = imwarp(data_out{data_idx,1},tform1,'OutputView',rout1);
    data_out{idx,1} = imwarp(step1,tform2,'OutputView',rout2);
    % Warp boxes.
    bbox_step1 = bboxwarp(data_out{data_idx,2},tform1,rout1);
    data_out{idx,2} = bboxwarp(bbox_step1,tform2,rout2);
end

end