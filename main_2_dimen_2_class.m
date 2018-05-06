%% 2-dimension 2-class
%% data
clear
% number
N1=100;
N2=100;
num=N1+N2;
d=2;
% mean
mean_pos=[1,5];
mean_neg=[-3,-3];
%covariance
sigma=4;
co_pos=sigma^2*eye(d);
co_neg=co_pos;
pos_x=mvnrnd(mean_pos,co_pos,N1); %generate data (guassian distribution)
neg_x=mvnrnd(mean_neg,co_neg,N2);
X=[pos_x;neg_x];
y=[ones(N1,1);-ones(N2,1)];
figure(1);
plot(pos_x(:,1),pos_x(:,2),'r+');
hold on
plot(neg_x(:,1),neg_x(:,2),'go');
xlabel('x1');ylabel('x2');
hold off
%% make decisions and evaluate(consider decision loss)
p1=0.6;p2=1-p1;
de_table=[0 5
          1 0];
[accuracy, sn, sp]=predict_2_class(X,y,num,mean_pos,mean_neg,co_pos,co_neg,p1,de_table);
% p_all=0.01:0.0005:0.99;
% p_num=length(p_all);
a12=1:0.05:10;
p_num=length(a12);
ac_all=zeros(1,p_num);
sn_all=zeros(1,p_num);
sp_all=zeros(1,p_num);
for i=1:p_num
de_table_1=[0 a12(i)
          1 0];
[ac_all(i),sn_all(i),sp_all(i)]=predict_2_class(X,y,num,mean_pos,mean_neg,co_pos,co_neg,p1,de_table_1);
end
% for i=1:p_num
% [ac_all(i),sn_all(i),sp_all(i)]=predict_2_class(X,y,num,mean_pos,mean_neg,co_pos,co_neg,p_all(i),de_table);
% end
%x_all=p_all;
x_all=a12;
figure(6);
plot(x_all,ac_all,'r-','LineWidth',2);
hold on;
plot(x_all,sn_all,'g-','LineWidth',2);
hold on;
plot(x_all,sp_all,'b-','LineWidth',2);
legend('accuracy','sn','sp');
%xlabel('p(w1)');ylabel('ÆÀ¼ÛÖ¸±ê');
xlabel('\lambda12');ylabel('ÆÀ¼ÛÖ¸±ê');
hold off


%% plot pdf
[x1,y1] = meshgrid(-10:0.05:10);
len=size(x1,1);
hei=size(y1,1);
num_draw=len*hei;
x_draw=[reshape(x1,num_draw,1) reshape(y1,num_draw,1)];
y_times_pos=zeros(num_draw,1);
y_times_neg=zeros(num_draw,1);
for i=1:num_draw
    y_times_pos(i)=exp(postdens_calc(x_draw(i,:),mean_pos,co_pos,p1)-d/2*log(2*pi))*(de_table(2,1)-de_table(1,1));
    y_times_neg(i)=exp(postdens_calc(x_draw(i,:),mean_neg,co_neg,p2)-d/2*log(2*pi))*(de_table(1,2)-de_table(2,2));
%     y_times_pos(i)=exp(postdens_calc(x_draw(i,:),mean_pos,co_pos,p1)-d/2*log(2*pi));
%     y_times_neg(i)=exp(postdens_calc(x_draw(i,:),mean_neg,co_neg,p2)-d/2*log(2*pi));
end
z_pos=reshape(y_times_pos,len,hei);
z_neg=reshape(y_times_neg,len,hei);
% p(x|w1)*p(w)
cdata_pos=cat(3,ones(size(x1)),zeros(size(x1)),zeros(size(x1))); % red
cdata_neg=cat(3,zeros(size(x1)),ones(size(x1)),zeros(size(x1)));% green
%cdata=cat(3,zeros(size(X)),zeros(size(X)),ones(size(X)));%blue
figure(2);
surf(x1,y1,z_pos,cdata_pos);
hold on;
surf(x1,y1,z_neg,cdata_neg);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(X|w1)*p(w1)*(\lambda21-\lambda11)','p(X|w2)*p(w2)*(\lambda12-\lambda22)');
%legend('p(X|w1)*p(w1)','p(X|w2)*p(w2)');
dec_pos_1=(abs(z_pos-z_neg)<0.0001);
dec_pos_2=(z_pos>0.0002);
p=dec_pos_1&dec_pos_2;
x_hat=x1.*p;
y_hat=y1.*p;
index=find(x_hat~=0);
figure(3);
plot(x_hat(index),y_hat(index));
hold on;
plot(pos_x(:,1),pos_x(:,2),'r+');
hold on
plot(neg_x(:,1),neg_x(:,2),'go');
xlabel('x1');ylabel('x2');
hold off
%
figure(4);
surf(x1,y1,z_pos./(z_pos+z_neg),cdata_pos);
hold on;
surf(x1,y1,z_neg./(z_pos+z_neg),cdata_neg);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(w1|x)','p(w2|x)');
% p(x|w)
figure(5);
surf(x1,y1,z_pos/p1,cdata_pos);
hold on;
surf(x1,y1,z_neg/p2,cdata_neg);
xlabel('x1');ylabel('x2');zlabel('p');
legend('p(x|w1)','p(x|w2)');
