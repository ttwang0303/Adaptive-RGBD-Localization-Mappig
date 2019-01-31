%% ORB
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_ORB-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_ORB-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_ORB-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_ORB-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

file=csvread('Landmarks_ORB-LATCH.csv');
x=file(:,1);
y=file(:,2);
p5=plot(x,y,'LineWidth',LineWidth,'Color','c', 'Marker','>', 'LineStyle',LineStyle);

title('ORB');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5],'BRISK','BRIEF','FREAK','ORB','LATCH','Location','NorthEastOutside');
hold off

%% BRISK
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_BRISK-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_BRISK-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_BRISK-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_BRISK-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

file=csvread('Landmarks_BRISK-LATCH.csv');
x=file(:,1);
y=file(:,2);
p5=plot(x,y,'LineWidth',LineWidth,'Color','c', 'Marker','p', 'LineStyle',LineStyle);

file=csvread('Landmarks_BRISK-SURF.csv');
x=file(:,1);
y=file(:,2);
p6=plot(x,y,'LineWidth',LineWidth,'Color',[0.5,0.3,0.5], 'Marker','x', 'LineStyle',LineStyle);

title('BRISK');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6],'BRISK','BRIEF','FREAK','ORB','LATCH', 'SURF','Location','NorthEastOutside');
hold off

%% FAST
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_FAST-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-LATCH.csv');
x=file(:,1);
y=file(:,2);
p5=plot(x,y,'LineWidth',LineWidth,'Color','c', 'Marker','p', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-SURF.csv');
x=file(:,1);
y=file(:,2);
p6=plot(x,y,'LineWidth',LineWidth,'Color',[0.5,0.3,0.5], 'Marker','x', 'LineStyle',LineStyle);

file=csvread('Landmarks_FAST-SIFT.csv');
x=file(:,1);
y=file(:,2);
p7=plot(x,y,'LineWidth',LineWidth,'Color',[0.9,0.6,0.9], 'Marker','h', 'LineStyle',LineStyle);

title('FAST');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6 p7],'BRISK','BRIEF','FREAK','ORB','LATCH', 'SURF','SIFT','Location','NorthEastOutside');
hold off

%% SHI_TOMASI
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_SHI_TOMASI-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-LATCH.csv');
x=file(:,1);
y=file(:,2);
p5=plot(x,y,'LineWidth',LineWidth,'Color','c', 'Marker','p', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-SURF.csv');
x=file(:,1);
y=file(:,2);
p6=plot(x,y,'LineWidth',LineWidth,'Color',[0.5,0.3,0.5], 'Marker','x', 'LineStyle',LineStyle);

file=csvread('Landmarks_SHI_TOMASI-SIFT.csv');
x=file(:,1);
y=file(:,2);
p7=plot(x,y,'LineWidth',LineWidth,'Color',[0.9,0.6,0.9], 'Marker','h', 'LineStyle',LineStyle);

title('SHI_TOMASI');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6 p7],'BRISK','BRIEF','FREAK','ORB','LATCH', 'SURF','SIFT','Location','NorthEastOutside');
hold off

%% STAR
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_STAR-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_STAR-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_STAR-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_STAR-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

file=csvread('Landmarks_STAR-LATCH.csv');
x=file(:,1);
y=file(:,2);
p5=plot(x,y,'LineWidth',LineWidth,'Color','c', 'Marker','p', 'LineStyle',LineStyle);

file=csvread('Landmarks_STAR-SURF.csv');
x=file(:,1);
y=file(:,2);
p6=plot(x,y,'LineWidth',LineWidth,'Color',[0.5,0.3,0.5], 'Marker','x', 'LineStyle',LineStyle);

title('STAR');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4 p5 p6],'BRISK','BRIEF','FREAK','ORB','LATCH', 'SURF','Location','NorthEastOutside');
hold off

%% SURF
clear all;
close all;
clc;

LineWidth=1.3;
LineStyle='-.';

hold on
file=csvread('Landmarks_SURF-BRISK.csv');
x=file(:,1);
y=file(:,2);
p1=plot(x,y,'LineWidth',LineWidth,'Color','r', 'Marker','+', 'LineStyle',LineStyle);

file=csvread('Landmarks_SURF-BRIEF.csv');
x=file(:,1);
y=file(:,2);
p2=plot(x,y,'LineWidth',LineWidth,'Color','g', 'Marker','o', 'LineStyle',LineStyle);

file=csvread('Landmarks_SURF-FREAK.csv');
x=file(:,1);
y=file(:,2);
p3=plot(x,y,'LineWidth',LineWidth,'Color','b', 'Marker','*', 'LineStyle',LineStyle);

file=csvread('Landmarks_SURF-ORB.csv');
x=file(:,1);
y=file(:,2);
p4=plot(x,y,'LineWidth',LineWidth,'Color','k', 'Marker','d', 'LineStyle',LineStyle);

title('SURF');
xlabel('Observations');
ylabel('Landmarks');
grid on

a=axes('position',get(gca,'position'),'visible','off');
legend(a,[p1 p2 p3 p4],'BRISK','BRIEF','FREAK','ORB','Location','NorthEastOutside');
hold off