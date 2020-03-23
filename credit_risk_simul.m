clear all;
clc
format long;

Nout  = 100000; % number of out-of-sample scenarios
Nin   = 5000;   % number of in-sample scenarios
Ns    = 5;      % number of idiosyncratic scenarios for each systemic

C = 8;          % number of credit states

% Filename to save out-of-sample scenarios
filename_save_out  = 'scen_out';

% Read and parse instrument data
instr_data = dlmread('instrum_data.csv', ',');
instr_id   = instr_data(:,1);           % ID
driver     = instr_data(:,2);           % credit driver
beta       = instr_data(:,3);           % beta (sensitivity to credit driver)
recov_rate = instr_data(:,4);           % expected recovery rate
value      = instr_data(:,5);           % value
prob       = instr_data(:,6:6+C-1);     % credit-state migration probabilities (default to A)
exposure   = instr_data(:,6+C:6+2*C-1); % credit-state migration exposures (default to A)
retn       = instr_data(:,6+2*C);       % market returns

K = size(instr_data, 1); % number of  counterparties

% Read matrix of correlations for credit drivers
rho = dlmread('credit_driver_corr.csv', '\t');
sqrt_rho = (chol(rho))'; % Cholesky decomp of rho (for generating correlated Normal random numbers)

disp('======= Credit Risk Model with Credit-State Migrations =======')
disp('============== Monte Carlo Scenario Generation ===============')
disp(' ')
disp(' ')
disp([' Number of out-of-sample Monte Carlo scenarios = ' int2str(Nout)])
disp([' Number of in-sample Monte Carlo scenarios = ' int2str(Nin)])
disp([' Number of counterparties = ' int2str(K)])
disp(' ')

% Find credit-state for each counterparty
% 8 = AAA, 7 = AA, 6 = A, 5 = BBB, 4 = BB, 3 = B, 2 = CCC, 1 = default
[Ltemp, CS] = max(prob, [], 2);
clear Ltemp

% Account for default recoveries
exposure(:, 1) = (1-recov_rate) .* exposure(:, 1);

% Compute credit-state boundaries
CS_Bdry = norminv( cumsum(prob(:,1:C-1), 2) );

%% Out of Sample (Nout = 100000)
% -------- Insert your code here -------- %

if(~exist('scenarios_out.mat','file'))
    
    % -------- Insert your code here -------- %
    w_out = [];
    CR_out = [];
    for s = 1:Nout
        % -------- Insert your code here -------- %
        A_out = normrnd(0,1,length(sqrt_rho),1);
        credidrivers_out = sqrt_rho * A_out;
        for i = 1:K
            for j = 1:driver(end)
                if driver(i) == j
                    counteryparties_out(i) = credidrivers_out(j);%creating 100 counterparties
                end
            end
        end
        y_out = counteryparties_out';
        z_out = normrnd(0,1,length(beta),1);
        sigma_out = sqrt(1-beta.^2);
        w_out = [w_out sort(-(beta .* y_out + sigma_out .* z_out))];
        
        credirate_out = ones(1,100);%credit rate holder
        credirate_out_platform = ones(100,Nout);%credits rate holder
        bdry_size = size(CS_Bdry);
        
        for k = 1:bdry_size(1)
            AA_bdry = CS_Bdry(k,end);%finding the AAA boundary
            CCC_bdry = CS_Bdry(k,1);%finding the D boundary
            if w_out(k,s) < CCC_bdry
                credirate_out(k) = 1;
            elseif w_out(k,s) >= AA_bdry
                credirate_out(k) = 8;
            else
                for space = 1:bdry_size(2)-1
                    low_bdry = CS_Bdry(k,space);%setting lower boundary
                    up_bdry = CS_Bdry(k,space+1);%setting upper boundary
                    if ((w_out(k,s) >= low_bdry) && (w_out(k,s) < up_bdry))
                        credirate_out(k) = space; %finding creditworthniess the relative credit rate range
                    else
                        continue
                    end
                end 
            end
        end
        crediR_out = credirate_out_platform(:,s) .* credirate_out';
        CR_out = [CR_out crediR_out]; 
    end
    
    for n = 1:100
        for o = 1:Nout
             crediloss_out(n,o) =  exposure(n,(CR_out(n,o)));%calculate losses according to exposure given credit rating
        end
    end
    Losses_out = crediloss_out';
    Losses_out = sort(-Losses_out,2);
    save('scenarios_out', 'Losses_out')
else
    load('scenarios_out', 'Losses_out')
end

% Normal approximation computed from out-of-sample scenarios
mu_l = mean(Losses_out)';
var_l = cov(Losses_out);

% Compute portfolio weights
portf_v = sum(value);     % portfolio value
w0{1} = value / portf_v;  % asset weights (portfolio 1)
w0{2} = ones(K, 1) / K;   % asset weights (portfolio 2)
x0{1} = (portf_v ./ value) .* w0{1};  % asset units (portfolio 1)
x0{2} = (portf_v ./ value) .* w0{2};  % asset units (portfolio 2)

% Quantile levels (99%, 99.9%)
alphas = [0.99 0.999];

% Compute VaR and CVaR (non-Normal and Normal) for 100000 scenarios
for(portN = 1:2)
    for(q=1:length(alphas))
        alf = alphas(q);
        % -------- Insert your code here -------- %
        portf_loss_out = [sort(-(x0{1}' * Losses_out'),2);sort(-(x0{2}' * Losses_out'),2)];
        mu_p_out = [mean(portf_loss_out(1,:)),mean(portf_loss_out(2,:))];
        sigma_p_out = [std(portf_loss_out(1,:)),std(portf_loss_out(2,:))];
        VaRout(portN,q)  = portf_loss_out(portN,(ceil(Nout*alf)));%non-normal
        VaRinN(portN,q)  = mu_p_out(portN)  + norminv(alf,0,1)*(sigma_p_out(portN));%normal
        CVaRout(portN,q) = (1/(Nout*(1-alf))) * ( (ceil(Nout*alf)-Nout*alf) * VaRout(portN,q) + sum(portf_loss_out(portN,(ceil(Nout*alf)+1:Nout))));%non-normal
        CVaRinN(portN,q) = mu_p_out(portN) + (normpdf(norminv(alf,0,1))/(1-alf))*(sigma_p_out(portN));%normal
        % -------- Insert your code here -------- %        
 	end
end

%% Perform 100 trials
N_trials = 100;

for(tr=1:N_trials)
 %% Monte Carlo approximation 1   
    % -------- Insert your code here -------- %
    w_MC1 = [];
    CR_MC1 = [];
    for s = 1:ceil(Nin/Ns) % systemic scenarios
        % -------- Insert your code here -------- %
        A_MC1 = normrnd(0,1,length(sqrt_rho),1);
        credidrivers_MC1 = sqrt_rho * A_MC1;
        for i = 1:length(driver)
            for j = 1:driver(end)
                if driver(i) == j
                    counteryparties_MC1(i) = credidrivers_MC1(j);%creating 100 counterparties
                end
            end
        end
        y_MC1 = counteryparties_MC1';
        for si = 1:Ns % idiosyncratic scenarios for each systemic
            % -------- Insert your code here -------- %
            z_MC1 = normrnd(0,1,length(beta),1);
            w_MC1 = [w_MC1 sort(-(beta .* y_MC1 + sqrt(1-beta.^2) .* z_MC1))]; 
        end
    end
    for s_MC1_5000 = 1:Nin
        credirate_MC1 = ones(1,100);%credit rate holder
        credirate_MC1_platform = ones(100,Nout);%credits rate holder
        bdry_size = size(CS_Bdry);
        
        for k = 1:bdry_size(1)
            AA_bdry = CS_Bdry(k,end);%finding the AAA boundary
            CCC_bdry = CS_Bdry(k,1);%finding the D boundary
            if w_MC1(k,s_MC1_5000) < CCC_bdry
                credirate_MC1(k) = 1;
            elseif w_MC1(k,s_MC1_5000) >= AA_bdry
                credirate_MC1(k) = 8;
            else
                for space = 1:bdry_size(2)-1
                    low_bdry = CS_Bdry(k,space);%setting lower boundary
                    up_bdry = CS_Bdry(k,space+1);%setting upper boundary
                    if ((w_MC1(k,s_MC1_5000) >= low_bdry) && (w_MC1(k,s_MC1_5000) < up_bdry))
                        credirate_MC1(k) = space; %finding creditworthniess the relative credit rate range
                    else
                        continue
                    end
                end 
            end
        end
        crediR_MC1 = credirate_MC1_platform(:,s) .* credirate_MC1';
        CR_MC1 = [CR_MC1 crediR_MC1]; 
    end
    % Calculated losses for MC1 approximation (5000 x 100)
    for n = 1:100
        for o = 1:Nin
             crediloss_MC1(n,o) =  exposure(n,(CR_MC1(n,o)));%calculate losses accroding to exposure given credit rating
        end
    end
    % Losses_inMC1
    Losses_inMC1 = crediloss_MC1';
    Losses_inMC1 = sort(-Losses_inMC1,2);
    
    
    
    
%% Monte Carlo approximation 2
    
    % -------- Insert your code here -------- %
    w_MC2 = [];
    CR_MC2 = [];
    for s = 1:Nin % systemic scenarios (1 idiosyncratic scenario for each systemic)
        
  
        % -------- Insert your code here -------- %
        A_MC2 = normrnd(0,1,length(sqrt_rho),1);
        credidrivers_MC2 = sqrt_rho * A_MC2;
        for i = 1:K
            for j = 1:driver(end)
                if driver(i) == j
                    counteryparties_MC2(i) = credidrivers_MC2(j);%creating 100 counterparties
                end
            end
        end
        y_MC2 = counteryparties_MC2';
        z_MC2 = normrnd(0,1,length(beta),1);
        sigma_MC2 = sqrt(1-beta.^2);
        w_MC2 = [w_MC2 sort(-(beta .* y_MC2 + sigma_MC2 .* z_MC2))];
        
        credirate_MC2 = ones(1,100);%credit rate holder
        credirate_MC2_platform = ones(100,Nout);%credits rate holder
        bdry_size = size(CS_Bdry);
        
        for k = 1:bdry_size(1)
            AA_bdry = CS_Bdry(k,end);%finding the AAA boundary
            CCC_bdry = CS_Bdry(k,1);%finding the D boundary
            if w_MC2(k,s) < CCC_bdry
                credirate_MC2(k) = 1;
            elseif w_MC2(k,s) >= AA_bdry
                 credirate_MC2(k) = 8;
            else
                 for space = 1:bdry_size(2)-1
                     
                     low_bdry = CS_Bdry(k,space);%setting lower boundary
                     up_bdry = CS_Bdry(k,space+1);%setting upper boundary
                     if ((w_MC2(k,s) >= low_bdry) && (w_MC2(k,s) < up_bdry))
                        credirate_MC2(k) = space; %finding creditworthniess the relative credit rate range
                     else
                        continue
                     end
                end 
             end
         end
         crediR_MC2 = credirate_MC2_platform(:,s) .* credirate_MC2';
         CR_MC2 = [CR_MC2 crediR_MC2]; 
     end
    % Calculated losses for MC2 approximation (5000 x 100)
     for n = 1:100
         for o = 1:Nin
             crediloss_MC2(n,o) =  exposure(n,(CR_MC2(n,o)));%calculate losses accroding to exposure given credit rating
         end
     end
     % Losses_inMC2
     Losses_inMC2 = crediloss_MC2';
     Losses_inMC2 = sort(-Losses_inMC2,2);
        % -------- Insert your code here -------- %
        
    
    
    
    % Compute VaR and CVaR
    for(portN = 1:2)
        for(q=1:length(alphas))
            alf = alphas(q);
            % -------- Insert your code here -------- %            
            % Compute portfolio loss 
            portf_loss_inMC1 = [sort(-(x0{1}' * Losses_inMC1'));sort(-(x0{2}' * Losses_inMC1'))];
            portf_loss_inMC2 = [sort(-(x0{1}' * Losses_inMC2'));sort(-(x0{2}' * Losses_inMC2'))];
            mu_MCl = mean(Losses_inMC1)';
            var_MCl = cov(Losses_inMC1);
            mu_MC2 = mean(Losses_inMC2)';
            var_MC2 = cov(Losses_inMC2);
            % Compute portfolio mean loss mu_p_MC1 and portfolio standard deviation of losses sigma_p_MC1
            mu_p_MC1 = [mean(portf_loss_inMC1(1,:)),mean(portf_loss_inMC1(2,:))];
            sigma_p_MC1 = [std(portf_loss_inMC1(1,:)),std(portf_loss_inMC1(2,:))];
            % Compute portfolio mean loss mu_p_MC2 and portfolio standard deviation of losses sigma_p_MC2
            mu_p_MC2 = [mean(portf_loss_inMC2(1,:)),mean(portf_loss_inMC2(2,:))];
            sigma_p_MC2 = [std(portf_loss_inMC2(1,:)),std(portf_loss_inMC2(2,:))];
            
            % Compute VaR and CVaR for the current trial
            VaRinMC1{portN,q}(tr) = portf_loss_inMC1(portN,(ceil(Nin*alf)));%non-normal 
            VaRinMC2{portN,q}(tr) = portf_loss_inMC2(portN,(ceil(Nin*alf)));%non-normal 
            VaRinN1{portN,q}(tr) =  mu_p_MC1(portN)  + norminv(alf,0,1)*(sigma_p_MC1(portN));%normal
            VaRinN2{portN,q}(tr) =  mu_p_MC2(portN)  + norminv(alf,0,1)*(sigma_p_MC2(portN));%normal
            CVaRinMC1{portN,q}(tr) = (1/(Nin*(1-alf))) * ( (ceil(Nin*alf)-Nin*alf) * VaRinMC1{portN,q}(tr) + sum(portf_loss_inMC1(portN,(ceil(Nin*alf)+1:Nin))));%non-normal
            CVaRinMC2{portN,q}(tr) = (1/(Nin*(1-alf))) * ( (ceil(Nin*alf)-Nin*alf) * VaRinMC2{portN,q}(tr) + sum(portf_loss_inMC2(portN,(ceil(Nin*alf)+1:Nin))));%non-normal
            CVaRinN1{portN,q}(tr) = mu_p_MC1(portN) + (normpdf(norminv(alf,0,1))/(1-alf))*(sigma_p_MC1(portN));%normal
            CVaRinN2{portN,q}(tr) = mu_p_MC2(portN) + (normpdf(norminv(alf,0,1))/(1-alf))*(sigma_p_MC2(portN));%normalre -------- %
        end
    end
end

%Display portfolio VaR and CVaR
for(portN = 1:2)
fprintf('\nPortfolio %d:\n\n', portN)    
 for(q=1:length(alphas))
    alf = alphas(q);
    fprintf('Out-of-sample: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRout(portN,q), 100*alf, CVaRout(portN,q))
    fprintf('In-sample MC1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC1{portN,q}), 100*alf, mean(CVaRinMC1{portN,q}))
    fprintf('In-sample MC2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinMC2{portN,q}), 100*alf, mean(CVaRinMC2{portN,q}))
    fprintf(' In-sample No: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRinN(portN,q), 100*alf, CVaRinN(portN,q))
    fprintf(' In-sample N1: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n', 100*alf, mean(VaRinN1{portN,q}), 100*alf, mean(CVaRinN1{portN,q}))
    fprintf(' In-sample N2: VaR %4.1f%% = $%6.2f, CVaR %4.1f%% = $%6.2f\n\n', 100*alf, mean(VaRinN2{portN,q}), 100*alf, mean(CVaRinN2{portN,q}))
 end
end

%% Plot results
%figure 3
p1_mc1_varin = mean(VaRinMC1{1,q});
p1_mc1_cvarin = mean(CVaRinMC1{1,q});
p1_n1_varin = mean(VaRinN1{1});
p1_n1_cvarin = mean(CVaRinN1{1});
%figure 4
p2_mc1_varin = mean(VaRinMC1{2,q});
p2_mc1_cvarin = mean(CVaRinMC1{2,q});
p2_n1_varin = mean(VaRinN1{2});
p2_n1_cvarin = mean(CVaRinN1{2});
%figure 5
p1_mc2_varin = mean(VaRinMC2{1,q});
p1_mc2_cvarin = mean(CVaRinMC2{1,q});
p1_n2_varin = mean(VaRinN2{1});
p1_n2_cvarin = mean(CVaRinN2{1});
%figure 6
p2_mc2_varin = mean(VaRinMC2{2,q});
p2_mc2_cvarin = mean(CVaRinMC2{2,q});
p2_n2_varin = mean(VaRinN2{2});
p2_n2_cvarin = mean(CVaRinN2{2});
%figure 7 numerical
p1_mc2_stdvarin = std(VaRinMC2{1,q});
p1_mc2_meanvarin = mean(VaRinMC2{1,q});
p1_stdvarout = sigma_p_out(1);
p1_meanvarout = mu_p_out(1);
%figure 8 numerical
p1_n_stdvarin = std(VaRinN(1));
p1_n_meanvarin = mean(VaRinN(1));
p1_n2_stdvarin = std(VaRinN2{1,q});
p1_n2_meanvarin = mean(VaRinN2{1,q});
%% Out Sample Normal and Non-normal VaR and CVaR for Portfolio 1&2
% figure(1): For Portfolio 1
% Including: VaRout, CVaRout, VaRinN, CVaRinN
% -------- Insert your code here -------- %
figure(1)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_out(1,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([VaRout(1) VaRout(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([CVaRout(1) CVaRout(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_out(1))*sqrt(2*pi)) * exp( -0.5*((binLocations-mu_p_out(1))/sigma_p_out(1)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([VaRinN(1) VaRinN(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([CVaRinN(1) CVaRinN(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(VaRout(1), max(frequencyCounts)/6, 'VaRout')
text(VaRinN(1), max(frequencyCounts)/2, 'VaRinN')
text(CVaRout(1), max(frequencyCounts)/8, 'CVaRout')
text(CVaRinN(1), max(frequencyCounts)/4, 'CVaRinN')
xlabel('1-year loss in $ value on portfolio #1 (Out-of-sample)')
ylabel('Frequency')
title('Figure1:Out Sample Normal and Non-normal VaR and CVaR for Portfolio 1');

% figure(2):% figure(1): For Portfolio 2
% Including: VaRout, CVaRout, VaRinN, CVaRinN
% -------- Insert your code here -------- %
figure(2)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_out(2,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([VaRout(2) VaRout(2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([CVaRout(2) CVaRout(2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_out(2)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mu_p_out(2))/sigma_p_out(2)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([VaRinN(2) VaRinN(2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([CVaRinN(2) CVaRinN(2)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(VaRout(2), max(frequencyCounts)/6, 'VaRout')
text(VaRinN(2), max(frequencyCounts)/2, 'VaRinN')
text(CVaRout(2), max(frequencyCounts)/8, 'CVaRout')
text(CVaRinN(2), max(frequencyCounts)/4, 'CVaRinN')
xlabel('1-year loss in $ value on portfolio #2 (Out-of-sample)')
ylabel('Frequency')
title('Figure2: Out Sample Normal and Non-normal VaR and CVaR for Portfolio 2');
%% MC1 Normal and Non-normal VaR and CVaR for Portfolio 1&2
% figure(3): For Portfolio 1
% Including: p1_mc1_varin, p1_mc1_cvarin, p1_n1_varin, p1_n1_cvarin
% -------- Insert your code here -------- %
figure(3)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_inMC1(1,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([p1_mc1_varin p1_mc1_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([p1_mc1_cvarin p1_mc1_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_MC1(1)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mu_p_MC1(1))/sigma_p_MC1(1)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([p1_n1_varin p1_n1_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([p1_n1_cvarin p1_n1_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(p1_mc1_varin, max(frequencyCounts)/6, 'VaRinMC1')
text(p1_n1_varin, max(frequencyCounts)/2, 'VaRinN1')
text(p1_mc1_cvarin, max(frequencyCounts)/8, 'CVaRinMC1')
text(p1_n1_cvarin, max(frequencyCounts)/4, 'CVaRinN1')
xlabel('1-year loss in $ value on portfolio #1 (MC1)')
ylabel('Frequency')
title('Figure3: MC1 Normal and Non-normal VaR and CVaR for Portfolio 1');

% figure(4):% For Portfolio 2
% Including: p2_mc1_varin, p2_mc1_cvarin, p2_n1_varin, p2_n1_cvarin
% -------- Insert your code here -------- %
figure(4)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_inMC1(2,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([p2_mc1_varin p2_mc1_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([p2_mc1_cvarin p2_mc1_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_MC1(2)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mu_p_MC1(2))/sigma_p_MC1(2)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([p2_n1_varin p2_n1_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([p2_n1_cvarin p2_n1_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(p2_mc1_varin, max(frequencyCounts)/6, 'VaRinMC1')
text(p2_n1_varin, max(frequencyCounts)/2, 'VaRinN1')
text(p2_mc1_cvarin, max(frequencyCounts)/8, 'CVaRinMC1')
text(p2_n1_cvarin, max(frequencyCounts)/4, 'CVaRinN1')
xlabel('1-year loss in $ value on portfolio #2 (MC1)')
ylabel('Frequency')
title('Figure4: MC1 Normal and Non-normal VaR and CVaR for Portfolio 2');
%% MC2 Normal and Non-normal VaR and CVaR for Portfolio 1&2
% figure(5): For Portfolio 1
% Including: p1_mc2_varin, p1_mc2_cvarin, p1_n2_varin, p1_n2_cvarin
% -------- Insert your code here -------- %
figure(5)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_inMC2(1,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([p1_mc2_varin p1_mc2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([p1_mc2_cvarin p1_mc2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_MC2(1)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mu_p_MC2(1))/sigma_p_MC2(1)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([p1_n2_varin p1_n2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([p1_n2_cvarin p1_n2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(p1_mc2_varin, max(frequencyCounts)/6, 'VaRinMC2')
text(p1_n2_varin, max(frequencyCounts)/2, 'VaRinN2')
text(p1_mc2_cvarin, max(frequencyCounts)/8, 'CVaRinMC2')
text(p1_n2_cvarin, max(frequencyCounts)/4, 'CVaRinN2')
xlabel('1-year loss in $ value on portfolio #1 (MC2)')
ylabel('Frequency')
title('Figure5: MC2 Normal and Non-normal VaR and CVaR for Portfolio 1');

% figure(6):% For Portfolio 2
% Including: p2_mc2_varin, p2_mc2_cvarin, p2_n2_varin, p2_n2_cvarin
% -------- Insert your code here -------- %
figure(6)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_inMC2(2,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([p2_mc2_varin p2_mc2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([p2_mc2_cvarin p2_mc2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(sigma_p_MC2(2)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mu_p_MC2(2))/sigma_p_MC2(2)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([p2_n2_varin p2_n2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([p2_n2_cvarin p2_n2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
hold off;
text(p2_mc2_varin, max(frequencyCounts)/6, 'VaRinMC2')
text(p2_n2_varin, max(frequencyCounts)/2, 'VaRinN2')
text(p2_mc2_cvarin, max(frequencyCounts)/8, 'CVaRinMC2')
text(p2_n2_cvarin, max(frequencyCounts)/4, 'CVaRinN2')
xlabel('1-year loss in $ value on portfolio #2 (MC2)')
ylabel('Frequency')
title('Figure6: MC2 Normal and Non-normal VaR and CVaR for Portfolio 2');
%% SAMPLING ERROR(MC2 VS TRUE OUT OF SAMPLE LOSSES)PORTFOLIO #1 
%figure(7) MC2 PORTFOLIO #1 
figure(7)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_inMC2(1,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([p1_mc2_varin p1_mc2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([p1_mc2_cvarin p1_mc2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/((sigma_p_MC2(1))*sqrt(2*pi)) ) * exp( -0.5*((binLocations-(mu_p_MC2(1)))/sigma_p_MC2(1)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([p1_n2_varin p1_n2_varin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([p1_n2_cvarin p1_n2_cvarin], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold off;
text((alf)*p1_mc2_varin, max(frequencyCounts)/6, 'VaRinMC2')
text((alf)*p1_n2_varin, max(frequencyCounts)/2, 'VaRinN2')
text((alf)*p1_mc2_cvarin, max(frequencyCounts)/8, 'CVaRinMC2')
text((alf)*p1_n2_cvarin, max(frequencyCounts)/4, 'CVaRinN2')
xlabel('1-year loss in $ value on portfolio 1 (MC2)')
ylabel('Frequency')
title('Figure7: MC2 Portfolio #1 (Sampling Error)');
%% ANALYZING MODEL ERROR
%figure(8): OUT OF SAMPLE PORTFOLIO #1 
figure(8)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(portf_loss_out(1,:), 300);
bar(binLocations, frequencyCounts);
hold on;
line([VaRout(1) VaRout(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
line([CVaRout(1) CVaRout(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/((sigma_p_out(1))*sqrt(2*pi)) ) * exp( -0.5*((binLocations-(mu_p_out(1)))/sigma_p_out(1)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([VaRinN(1) VaRinN(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold on;
line([CVaRinN(1) CVaRinN(1)], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
hold off;
text((alf)*VaRout(1), max(frequencyCounts)/6, 'VaRout')
text((alf)*VaRinN(1), max(frequencyCounts)/2, 'VaRinN')
text((alf)*CVaRout(1), max(frequencyCounts)/8, 'CVaRout')
text((alf)*CVaRinN(1), max(frequencyCounts)/4, 'CVaRinN')
xlabel('1-year loss in $ value on portfolio 1(Out-of-sample)')
ylabel('Frequency')
title('Figue8: Out-of-sample Portfolio #1 (Model Error)');
