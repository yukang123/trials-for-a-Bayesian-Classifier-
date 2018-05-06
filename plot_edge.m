function plot_edge(z_1,z_2,x1,y1,d1,d2)
dec_pos_1=(abs(z_1-z_2)<d1);
dec_pos_2=(z_1>d2);
p=dec_pos_1&dec_pos_2;
x_hat=x1.*p;
y_hat=y1.*p;
index=find(x_hat~=0);
plot(x_hat(index),y_hat(index));
hold on;
end
