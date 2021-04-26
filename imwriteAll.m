function imwriteAll(file_name, aug_dir, data_out, count)
file_id = fopen(file_name,'a'); 
for i = 1:4
    count_str = [sprintf( '%06d', count - 1),'_',num2str(i-1),'.jpg']; 
    imwrite(data_out{i},[aug_dir,count_str])
    sz = size(data_out{i,2}); 
    for j = 1:sz(1)
        fprintf(file_id,'%s,%d,%d,%d,%d\n', [cd,'\AED\augmented_images\',count_str],  data_out{i,2}(j,1), data_out{i,2}(j,2), data_out{i,2}(j,3), data_out{i,2}(j,4)); 
    end
end
fclose(file_id); 
end