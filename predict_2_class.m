function [accuracy, sn, sp]=predict_2_class(X,y,num,mean_pos,mean_neg,co_pos,co_neg,p1,de_table)
% predict
y_predict=zeros(num,1);
for i=1:num
    x_s=X(i,:);
    g_pos=postdens_calc(x_s,mean_pos,co_pos,p1)+log(de_table(2,1)-de_table(1,1));
    g_neg=postdens_calc(x_s,mean_neg,co_neg,1-p1)+log(de_table(1,2)-de_table(2,2));
    if g_pos>g_neg
        y_predict(i)=1;
    else
        y_predict(i)=-1;
    end
end
% evaluate
index_pos=find(y==1);
sn=mean(y_predict(index_pos)==1);
index_neg=find(y==-1);
sp=mean(y_predict(index_neg)==-1);
accuracy=mean(y_predict==y);
fprintf('accuracy is %.2f \n',accuracy*100);
fprintf('sn is %.2f \n',sn*100);
fprintf('sp is %.2f \n',sp*100);