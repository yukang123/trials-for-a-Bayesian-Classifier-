function sam_data=ex_data(class,N)
% random extract N data from the dataset
list=randperm(size(class,1));
sam_data=class(list(1:N),:);
end