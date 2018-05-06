%% one-dimensional two-class classifier 
%% dataset
clear
pos_x=[-3.9847, -3.5549,  -1.2401,  -0.9780,  -0.7932,  -2.8531,  -2.7605,  -3.7287 ... 
   -3.5414,  -2.2692,  -3.4549,  -3.0752,  -3.9934,  -0.9780, -1.5799,  -1.4885 ...
   -0.7431,  -0.4221,  -1.1186,  -2.3462,  -1.0826,  -3.4196,  -1.3193,  -0.8367, -0.6579,  -2.9683];  %  26 positive samples
pos_x=pos_x';
num_pos=size(pos_x,1);
neg_x= [2.8792, 0.7932, 1.1882, 3.0682, 4.2532, 0.3271,0.9846,2.7648,2.6588];   % 9 negative samples
neg_x=neg_x';
num_neg=size(neg_x,1);
x=[pos_x;neg_x];
y=[ones(num_pos,1);-ones(num_neg,1)];
num=size(x,1);
d=size(x,2);
%% find parameters of the conditional density for each class
[pos_mu, pos_covariance]=parameters_learn(pos_x);
[neg_mu, neg_covariance]=parameters_learn(neg_x);
%% make decisions (consider decision loss)
p1=0.9;p2=0.1;
de_table=[0 6
          1 0];
y_predict=zeros(num,1);
for i=1:num
    x_s=x(i);
    g_pos=postdens_calc(x_s,pos_mu,pos_covariance,p1)+log(de_table(2,1)-de_table(1,1));
    g_neg=postdens_calc(x_s,neg_mu,neg_covariance,p2)+log(de_table(1,2)-de_table(2,2));
    if g_pos>g_neg
        y_predict(i)=1;
    else
        y_predict(i)=-1;
    end
end
%% evaluate 
index_pos=find(y==1);
sn=mean(y_predict(index_pos)==1);
index_neg=find(y==-1);
sp=mean(y_predict(index_neg)==-1);
accuracy=mean(y_predict==y);
fprintf('accuracy is %.2f \n',accuracy*100);
fprintf('sn is %.2f \n',sn*100);
fprintf('sp is %.2f \n',sp*100);
%% make decisions (without decision loss)
y_predict_noloss=zeros(num,1);
for i=1:num
    x_s=x(i);
    g_pos=postdens_calc(x_s,pos_mu,pos_covariance,p1);
    g_neg=postdens_calc(x_s,neg_mu,neg_covariance,p2);
    if g_pos>g_neg
       y_predict_noloss(i)=1;
    else
       y_predict_noloss(i)=-1;
    end
end
%% evaluate 
sn_noloss=mean(y_predict_noloss(index_pos)==1);
sp_noloss=mean(y_predict_noloss(index_neg)==-1);
accuracy_noloss=mean(y_predict_noloss==y);
fprintf('accuracy (without considering decision loss) is %.2f \n',accuracy_noloss*100);
fprintf('sn (without considering decision loss) is %.2f \n',sn_noloss*100);
fprintf('sp (without considering decision loss) is %.2f \n',sp_noloss*100);
%% plot pdf
x1=-5:0.01:5;
for i=1:length(x1)
y_times_pos(i)=exp(postdens_calc(x1(i),pos_mu,pos_covariance,p1)-d/2*log(2*pi));%*(de_table(2,1)-de_table(1,1));
y_times_neg(i)=exp(postdens_calc(x1(i),neg_mu,neg_covariance,p2)-d/2*log(2*pi));%*(de_table(1,2)-de_table(2,2));
end
% p(x|w)*p(w)
figure(1);
plot(x1,y_times_pos,'g-','LineWidth',2);
hold on;
plot(x1,y_times_neg,'r-','LineWidth',2);
hold on;
legend('p(X|w1)*p(w1)','p(X|w2)*p(w2)');
xlabel('x');
ylabel('p');
dec_pos=find(abs(y_times_pos-y_times_neg)<0.0003);
a1=[x1(dec_pos),x1(dec_pos)];
b1=[0,0.1];
plot(a1,b1,'--','LineWidth',1.5);
hold on;
plot(pos_x,zeros(num_pos,1),'r+');
plot(neg_x,zeros(num_neg,1),'go');
% p(w|x)
figure(2);
plot(x1,y_times_pos./(y_times_pos+y_times_neg),'g-','LineWidth',2);
hold on;
plot(x1,y_times_neg./(y_times_pos+y_times_neg),'r-','LineWidth',2);
hold on;
legend('p(w1|x)','p(w2|x)');
xlabel('x');
ylabel('p');
dec_pos_1=find(abs(y_times_pos-y_times_neg)<0.0003);
t=x1(dec_pos_1);
a1=[t,t];
b1=[0,0.8];
plot(a1,b1,'--','LineWidth',1.5);
hold on;
plot(pos_x,zeros(num_pos,1),'r+');
plot(neg_x,zeros(num_neg,1),'go');
hold off;
% p(x|w)
figure(3);
plot(x1,y_times_pos/p1,'g-','LineWidth',2);
hold on;
plot(x1,y_times_neg/p2,'r-','LineWidth',2);
hold on;
legend('p(x|w1)','p(x|w2)');
xlabel('x');
ylabel('p');
hold off;







