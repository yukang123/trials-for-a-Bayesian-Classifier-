%% 2-dimension 2-class
%% data
clear
% number
d=2;
load('training.mat');
load('test.mat');
N_train=[500 300 200];
N_test=[250 150 100];
num_train=sum(N_train);
num_test=sum(N_test);
train_1=ex_data(class1_train,N_train(1));
train_2=ex_data(class2_train,N_train(2));
train_3=ex_data(class3_train,N_train(3));
test_1=ex_data(class1_test,N_test(1));
test_2=ex_data(class2_test,N_test(2));
test_3=ex_data(class3_test,N_test(3));
train_x=[train_1;train_2;train_3];
test_x=[test_1;test_2;test_3];
train_y=[ones(N_train(1),1);2*ones(N_train(2),1);3*ones(N_train(3),1)];
test_y=[ones(N_test(1),1);2*ones(N_test(2),1);3*ones(N_test(3),1)];
figure(1);
plotdata(train_1,train_2,train_3);
title('training dataset');
figure(2)
plotdata(test_1,test_2,test_3);
title('test dataset');
%% train 
[mu_1, co_1]=parameters_learn(train_1);
[mu_2, co_2]=parameters_learn(train_2);
[mu_3, co_3]=parameters_learn(train_3);
mu=[mu_1;mu_2;mu_3];
co_all=zeros(d,d,3);
co_all(:,:,1)=co_1;
co_all(:,:,2)=co_2;
co_all(:,:,3)=co_3;
%% make decisions and evaluate(consider decision loss)
p_all=[0.5 0.3 0.2];
% de_table=[0 3 1
%           5 0 1
%           5 3 0];
de_table=[0 1 1
          1 0 1
          1 1 0];      
y_predict_train=predict(train_x,num_train,de_table,mu,co_all,p_all);
accuracy_train=mean(y_predict_train==train_y);
fprintf('accuracy_train is %.2f \n',accuracy_train*100);
y_predict_test=predict(test_x,num_test,de_table,mu,co_all,p_all);
accuracy_test=mean(y_predict_test==test_y);
fprintf('accuracy_test is %.2f \n',accuracy_test*100);

%% plot pdf
[x1,y1] = meshgrid(-10:0.05:10);
len=size(x1,1);
hei=size(y1,1);
num_draw=len*hei;
x_draw=[reshape(x1,num_draw,1) reshape(y1,num_draw,1)];
y_times=zeros(num_draw,3);
for i=1:num_draw
    x_s=x_draw(i,:);
    for k=1:3
    y_times(i,k)=exp(postdens_calc(x_s,mu(k,:),co_all(:,:,k),p_all(k))-d/2*log(2*pi));
    end
end
z_1=reshape(y_times(:,1),len,hei);
z_2=reshape(y_times(:,2),len,hei);
z_3=reshape(y_times(:,3),len,hei);
% p(x|w1)*p(w)
cdata_1=cat(3,ones(size(x1)),zeros(size(x1)),zeros(size(x1))); % red
cdata_2=cat(3,zeros(size(x1)),ones(size(x1)),zeros(size(x1)));% green
cdata_3=cat(3,zeros(size(x1)),zeros(size(x1)),ones(size(x1))); %blue
figure(3);
surf(x1,y1,z_1,cdata_1);
hold on;
surf(x1,y1,z_2,cdata_2);
hold on;
surf(x1,y1,z_3,cdata_3);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(X|w1)*p(w1)','p(X|w2)*p(w2)','p(X|w3)*p(w3)');
figure(4);
plot_edge(z_1,z_2,x1,y1,0.0001,0.0002);
plot_edge(z_3,z_2,x1,y1,0.0001,0.0002);
plot_edge(z_3,z_1,x1,y1,0.0003,0.0005);
plotdata(train_1,train_2,train_3);
hold on;
xlabel('x1');ylabel('x2');
hold off
% p(w|x)
figure(5);
surf(x1,y1,z_1./(z_1+z_2+z_3),cdata_1);
hold on;
surf(x1,y1,z_2./(z_1+z_2+z_3),cdata_2);
hold on
surf(x1,y1,z_3./(z_1+z_2+z_3),cdata_3);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(w1|x)','p(w2|x)','p(w3|x)');
% p(x|w)
figure(6);
surf(x1,y1,z_1/p_all(1),cdata_1);
hold on;
surf(x1,y1,z_2/p_all(2),cdata_2);
hold on
surf(x1,y1,z_3/p_all(3),cdata_3);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(x|w1)','p(x|w2)','p(x|w3)');
hold off