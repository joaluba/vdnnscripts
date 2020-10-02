figure;
subplot(3,1,1)
plot(y_predi(5000:6000,1),'b', 'Linewidth',1.5)
hold on
plot(y_test(5000:6000,1),'r', 'Linewidth',2)
title('estimation of F0')
subplot(3,1,2)
plot(y_predi(5000:6000,2),'b', 'Linewidth',1.5)
hold on
plot(y_test(5000:6000,2),'r', 'Linewidth',2)
title('estimation of F1')
subplot(3,1,3)
plot(y_predi(5000:6000,3),'b', 'Linewidth',1.5)
hold on 
plot(y_test(5000:6000,3),'r', 'Linewidth',2)
title('estimation of F2')
