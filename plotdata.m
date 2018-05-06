function plotdata(x_1,x_2,x_3)
plot(x_1(:,1),x_1(:,2),'r+');
hold on
plot(x_2(:,1),x_2(:,2),'go');
hold on
plot(x_3(:,1),x_3(:,2),'b*');
xlabel('x1');ylabel('x2');
hold off