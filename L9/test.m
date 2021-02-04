function test
clc;
clear all;
close all;

save_solution = false;

DYNCST  = @(x,u,i) dyn_func(x,u);
iter       = 500;                  
x0      = [pi; 0];              
u0      = 0.1 * randn(1, iter);    
Op.max_action = [-5, 5];             
Op.plot = -1;                  

figure(9);
set(gcf,'name','Pendulum','Menu','none','NumberT','off')
set(gca,'xlim',[-4 4],'ylim',[-4 4],'DataAspectRatio',[1 1 1])
box on
line_handle = line([0 0],[0 0],'color','w','linewidth',1);
Op.plotFn = @(x) set(line_handle,'Xdata', sin(x(1,:)),'Ydata',cos(x(1,:)));


% //////////////// optimization! //////////////
[x,u]= iLQG(DYNCST, x0, u0, Op);

if save_solution
    save('trajectory.mat', 'x', 'u');
end

% animate the resulting trajectory
figure(9)
handles = [];
for i=1:iter
   set(0,'currentfigure',9);
   delete(handles)
   handles = plot_viz(x(:,i), u(:,i), i);
   pause(0.01)
end


function c = pendulum_cost(x)
c = abs(x(1, :));

function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = dyn_func(x,u)

if nargout == 2
    f = f_dyn(x,u);
    c = abs(x(1, :));
else
    ix = 1:2;
    iu = 3;
    
    xu_dyn  = @(xu) f_dyn(xu(ix,:),xu(iu,:));
    J       = diff_func(xu_dyn, [x; u]);
    fx      = J(:,ix,:);
    fu      = J(:,iu,:);
    [fxx,fxu,fuu] = deal([]);
  
    
    x_cost = @(xu) pendulum_cost(xu(ix,:));
    J       = squeeze(diff_func(x_cost, [x; u]));
    cx      = J(ix,:);
    cu      = J(iu,:);
    
    % cost second derivatives
    xu_Jcst = @(xu) squeeze(diff_func(x_cost, xu));
    JJ      = diff_func(xu_Jcst, [x; u]);
    JJ      = 0.5*(JJ + permute(JJ,[2 1 3])); %symmetrize
    cxx     = JJ(ix,ix,:);
    cxu     = JJ(ix,iu,:);
    cuu     = JJ(iu,iu,:);
    
    [f,c] = deal([]);
end

function y = f_dyn(x, u)
    int = 0.01;
    m = 1;
    l = 1;
    g = 9.8;
    mu = 0.01;
    dt = 0.01;
    max_ang = [-pi, pi];
    max_vel = [-2 * pi, 2 * pi];    
    num_steps = int / dt;

    A = [0, 1; 0, -mu / (m * l^2)];
    b = [0; 1 / (m * l^2)];
    e = [0; g / l];

    for i = 1:num_steps
        dx = A * x + b * u + e * sin(x(1, :));
        y = x + dt * dx;
        y(1, :) = y(1, :) + 0.5 * dt^2 * dx(2, :);
        limit_down = y(1, :) < max_ang(1);
        y(1, :) = y(1, :) + limit_down * 2 * pi;  
        limit_up = y(1, :) > max_ang(2);
        y(1, :) = y(1, :) - limit_up * 2 * pi;
        y(2, :) = max(max_vel(1), min(y(2, :), max_vel(2)));
        x = y;
    end


function J = diff_func(fun, x, h)

if nargin < 3
    h = 2^-17;
end

[n, K]  = size(x);
H       = [zeros(n,1) h*eye(n)];
H       = permute(H, [1 3 2]);
X       = pp(x, H);
X       = reshape(X, n, K*(n+1));
Y       = fun(X);
m       = numel(Y)/(K*(n+1));
Y       = reshape(Y, m, K, n+1);
J       = pp(Y(:,:,2:end), -Y(:,:,1)) / h;
J       = permute(J, [1 3 2]);

function c = pp(a,b)
c = bsxfun(@plus,a,b);




% ///////////// graphics function ///////////////////////
% source:
% https://www.mathworks.com/help/symbolic/examples/simulate-physics-pendulum-swing.html
function h = plot_viz(x, u, iteration)

h = [];
x_pos = sin(x(1));
y_pos = cos(x(1));
hold on;
h(end+1) = plot(x_pos, y_pos, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
h(end+1) = plot([0 .96 * x_pos], [0 .96 * y_pos], 'k-');
xlim([-1.5, 1.5]);
ylim([-1.5, 1.5]);
if ~isempty(iteration)
    h(end+1) = text(-0.2, 1.25, "Timer: " + num2str(iteration * 0.01, 2) + " s");
    h(end+1) = quiver(x_pos, y_pos, u * cos(x(1)) / 10, - u * sin(x(1)) / 10, 'r', 'linewidth', 2, 'AutoScaleFactor', 2);
end
daspect([1 1 1]);
title({'iLQR Pendulum', 'iLQR Pendulum'})
xlabel('x')
ylabel('y')
