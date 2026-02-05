clear; clc; close all;

%% Parameters
mA = 1.0;
mB = 1.0;
qA = 1.0;
qB = 1.0;
k  = 1.0;

dt = 0.01;
T  = 10;
N  = T/dt;

alpha = 1e-3;   % control penalty
lr = 1e-2;      % learning rate

%% Initial Conditions
xA = [-1; 0];
vA = [0; 0];
xB = [ 1; 0];
vB = [0; 0];

xB_goal = [3; 0];

%% Control Initialization
U = zeros(2, N);  % force on A

%% Storage
XA = zeros(2,N); VA = zeros(2,N);
XB = zeros(2,N); VB = zeros(2,N);

%% Optimization Loop
for iter = 1:20
    
    % Reset state
    xA_i = xA; vA_i = vA;
    xB_i = xB; vB_i = vB;
    
    J = 0;
    
    for kstep = 1:N
        
        r = xB_i - xA_i;
        F_B = (k*qA*qB/norm(r)^3) * r;
        
        % Dynamics
        vA_i = vA_i + dt*(U(:,kstep)/mA - F_B/mA);
        xA_i = xA_i + dt*vA_i;
        
        vB_i = vB_i + dt*(F_B/mB);
        xB_i = xB_i + dt*vB_i;
        
        % Cost
        J = J + norm(xB_i - xB_goal)^2 ...
              + norm(vB_i)^2 ...
              + alpha*norm(U(:,kstep))^2;
        
        % Save for visualization
        XA(:,kstep) = xA_i; VA(:,kstep) = vA_i;
        XB(:,kstep) = xB_i; VB(:,kstep) = vB_i;
    end
    
    % Gradient (finite difference – simple and robust)
    gradU = zeros(size(U));
    eps = 1e-4;
    
    for i = 1:2
        for kstep = 1:N
            U(i,kstep) = U(i,kstep) + eps;
            Jp = simulate_cost(U, xA, vA, xB, vB, xB_goal, dt, mA, mB, qA, qB, alpha);
            U(i,kstep) = U(i,kstep) - eps;
            gradU(i,kstep) = (Jp - J)/eps;
        end
    end
    
    U = U - lr * gradU;
    
    fprintf("Iter %d | Cost %.3f\n", iter, J);
end

%% Live Visualization
figure;
for kstep = 1:10:N
    clf;
    plot(XA(1,1:kstep), XA(2,1:kstep), 'b'); hold on;
    plot(XB(1,1:kstep), XB(2,1:kstep), 'r');
    plot(xB_goal(1), xB_goal(2), 'kx', 'MarkerSize',10);
    legend('A','B','Goal');
    axis equal; grid on;
    drawnow;
end

function J = simulate_cost(U, xA, vA, xB, vB, xB_goal, dt, mA, mB, qA, qB, alpha)

N = size(U,2);
k = 1.0;   % interaction constant (must match main script)

J = 0;

for kstep = 1:N

    r = xB - xA;
    F_B = (k*qA*qB / (norm(r)^3 + 1e-6)) * r;

    % Dynamics
    vA = vA + dt*(U(:,kstep)/mA - F_B/mA);
    xA = xA + dt*vA;

    vB = vB + dt*(F_B/mB);
    xB = xB + dt*vB;

    % Cost
    J = J + norm(xB - xB_goal)^2 ...
           + norm(vB)^2 ...
           + alpha*norm(U(:,kstep))^2;

end
end
