%% ADMM_XUUpdate_Animated.m
% Two-point nonlinear system, ADMM with explicit U-update + live animation

clear; clc; close all;

%% Parameters
mA = 1; mB = 1;
qA = 1; qB = 1;
k = 1;
dt = 0.02; T = 2;
N = T/dt;
rho = 10;
alpha = 0.01; 
eta = 0.1; % gradient step for U

%% Initial conditions
xA = [-1;0]; vA = [0;0];
xB = [1;0];  vB = [0;0];
xB_goal = [3;2];

%% ADMM Initialization
X = zeros(8,N+1);
Z = X;
Lambda = X;
U = zeros(2,N);

X(:,1) = [xA;vA;xB;vB];
Z(:,1) = X(:,1);

max_iter = 50;

for iter = 1:max_iter
    %% X-update (physics)
    X(:,1) = [xA;vA;xB;vB];
    for k=1:N
        r = X(5:6,k) - X(1:2,k);
        F_B = k*qA*qB*r / (norm(r)^3 + 1e-6);
        vA_next = X(3:4,k) + dt*(U(:,k) - F_B)/mA;
        xA_next = X(1:2,k) + dt*X(3:4,k);
        vB_next = X(7:8,k) + dt*F_B/mB;
        xB_next = X(5:6,k) + dt*X(7:8,k);
        X(:,k+1) = [xA_next; vA_next; xB_next; vB_next];
    end
    
    %% Z-update (task projection)
    for k=1:N+1
        Z(5:6,k) = (2*xB_goal + rho*(X(5:6,k)+Lambda(5:6,k)))/(2+rho);
        Z(7:8,k) = (rho*(X(7:8,k)+Lambda(7:8,k)))/(2+rho);
        Z(1:4,k) = X(1:4,k) + Lambda(1:4,k);
    end
    
%% U-update (gradient step)
for k=1:N
    r = X(5:6,k) - X(1:2,k);
    F_B = k*qA*qB*r / (norm(r)^3 + 1e-6);
    % Gradient w.r.t U (only velocity of A matters)
    gradU = 2*alpha*U(:,k) + rho*(X(3:4,k)-Z(3:4,k)+Lambda(3:4,k));
    U(:,k) = U(:,k) - eta*gradU;
end

    
    %% Dual update
    Lambda = Lambda + X - Z;
end

%% Live animation
figure; hold on; grid on; axis equal;
xlim([-2 2]); ylim([-1 6]);
plot(xB_goal(1), xB_goal(2), 'kx','MarkerSize',10,'LineWidth',2,'DisplayName','B goal');
hA = plot(X(1,1), X(2,1),'ro','MarkerSize',8,'DisplayName','A');
hB = plot(X(5,1), X(6,1),'bo','MarkerSize',8,'DisplayName','B');
hU = quiver(X(1,1),X(2,1),U(1,1),U(2,1),'r','MaxHeadSize',2);
legend;

for k=1:N+1
    set(hA,'XData',X(1,k),'YData',X(2,k));
    set(hB,'XData',X(5,k),'YData',X(6,k));
    if k<=N
        set(hU,'XData',X(1,k),'YData',X(2,k),'UData',U(1,k)/2,'VData',U(2,k)/2);
    end
    drawnow;
    pause(0.01);
end
title('ADMM with explicit U-update: 2-point nonlinear system with animation');
xlabel('X'); ylabel('Y');
