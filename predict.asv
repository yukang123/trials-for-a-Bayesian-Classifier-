function  y_predict=predict(X,num,table,mean,co,pi)
y_predict=zeros(num,1);
for i=1:num
    x_s=X(i,:);
    loss=zeros(3,1);
    for j=1:3
        for k=1:3
        loss(j)=loss(j)+exp(postdens_calc(x_s,mean(k,:),co(:,:,k),pi(k)))*(table(j,k));
        end
    end
    [loss_min min_po]=min(loss);
     y_predict(i)=min_po;
end
end