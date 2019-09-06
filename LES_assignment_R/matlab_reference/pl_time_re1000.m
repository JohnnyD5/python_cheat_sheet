close all
clear all
% ***** read u
load u_v_time_4nodes_re1000.dat
fil=u_v_time_4nodes_re1000;
u1=fil(:,1);
v1=fil(:,2);
u2=fil(:,3);
v2=fil(:,4);
u3=fil(:,5);
v3=fil(:,6);
u4=fil(:,7);
v4=fil(:,8);

% time step
dt=8.177E-04;
n=length(u1);
% compute time array
t=dt:dt:n*dt;

%%%%%%%%%%%%%%%% plotting section %%%%%%%%%%%%%%%%%%%%%%%%%%
% plot u
plot(t,u2)
hold
plot(t,u3,'r--')
xlabel('t','fontsize',[20])
ylabel('u','fontsize',[20])
handle=gca
set(handle,'fontsize',[20])
print u_time.ps -deps
%