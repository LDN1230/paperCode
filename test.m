clc;
clear all;
addpath('C:/Users/LDN/Desktop/cpp/vsMex/vsMex/x64/Debug');
t1 = cputime;
vsMex(10000, 10000, 100);
t2 = cputime - t1;

t3 = cputime;
for i = 1:10000
    for j = 1:10000
        for k = 1:100
        
        end
    end
end
t4 = cputime - t3;