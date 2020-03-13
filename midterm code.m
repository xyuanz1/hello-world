clc;
clear all;
format long

% CSV file with price data
input_file_prices  = 'Daily_closing_prices.csv';

% Read daily prices
if(exist(input_file_prices,'file'))
  fprintf('\nReading daily prices datafile - %s\n', input_file_prices)
  fid = fopen(input_file_prices);
     % Read instrument tickers
     hheader  = textscan(fid, '%s', 1, 'delimiter', '\n');
     headers = textscan(char(hheader{:}), '%q', 'delimiter', ',');
     tickers = headers{1}(2:end);
     % Read time periods
     vheader = textscan(fid, '%[^,]%*[^\n]');
     dates = vheader{1}(1:end);
  fclose(fid);
  data_prices = dlmread(input_file_prices, ',', 1, 1);
else
  error('Daily prices datafile does not exist')
end

% Convert dates into array [year month day]
format_date = 'mm/dd/yyyy';
dates_array = datevec(dates, format_date);
dates_array = dates_array(:,1:3);

% Remove datapoints for year 2014
day_ind_start0 = 1;
day_ind_end0 = length(find(dates_array(:,1)==2014));
data_prices = data_prices(day_ind_end0+1:end,:);
dates_array = dates_array(day_ind_end0+1:end,:);
dates = dates(day_ind_end0+1:end,:);

% Compute means and covariances for Question 2
day_ind_start = 1;
day_ind_end = 39;
cur_returns = data_prices(day_ind_start+1:day_ind_end,:) ./ data_prices(day_ind_start:day_ind_end-1,:) - 1;
mu = mean(cur_returns)';  % Expected returns for Question 2
Q = cov(cur_returns);     % Covariances for Question 2


%% Question 1

% Specify quantile level for VaR/CVaR
alf = 0.95;

% Positions in the portfolio
positions = [100 0 0 0 0 0 0 0 200 500 0 0 0 0 0 0 0 0 0 0]';

% Number of assets in universe
Na = size(data_prices,2);

% Number of historical scenarios
Ns = size(data_prices,1);

cur_prices = data_prices(:,:);
%calculate total value of the portfolio
total_value = cur_prices*positions;

%% 1 day loss
%calculate daily return difference
PLData = diff(total_value);
% Sort loss data in increasing order
loss_1d = sort(-PLData);
%specify number of scenarios
N = size(PLData,1);
% Compute Historical 1-day VaR from the data
VaR  = loss_1d(ceil(N*alf));
% Compute Historical 1-day CVaR from the data
CVaR = (1/(N*(1-alf))) * ( (ceil(N*alf)-N*alf) * VaR + sum(loss_1d(ceil(N*alf)+1:N)) );

% Compute Normal 1-day VaR from the data
w_stock = (cur_prices(1,:)'.*positions)./total_value(1);%20x1
stock_return_1d = (data_prices(2:end,:)./data_prices(1:end-1,:)-1);%503x20
Q_stock_1d = cov(stock_return_1d);%20x20
port_std_1d = sqrt(w_stock'*Q_stock_1d*w_stock);
port_return_1d = mean(stock_return_1d,1)*w_stock;
VaRn = (-port_return_1d + norminv(alf,0,1)*port_std_1d)*total_value(1);
% Compute Normal 1-day CVaR from the data
CVaRn = (-port_return_1d + (normpdf(norminv(alf,0,1))/(1-alf))*port_std_1d)*total_value(1);

fprintf('Historical 1-day VaR %4.1f%% = $%6.2f,   Historical 1-day CVaR %4.1f%% = $%6.2f\n', 100*alf, VaR, 100*alf, CVaR)
fprintf('    Normal 1-day VaR %4.1f%% = $%6.2f,       Normal 1-day CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRn, 100*alf, CVaRn)

%% 10-day moving window
%calculate daily return difference
PLData_10d = total_value(10:end)-total_value(1:(end-9));
% Sort loss data in increasing order
loss_10d = sort(-PLData_10d);
%specify number of scenarios
N = size(PLData_10d,1);
% Compute Historical 1-day VaR from the data
VaR_10d  = loss_10d(ceil(N*alf));
% Compute Historical 1-day CVaR from the data
CVaR_10d = (1/(N*(1-alf))) * ( (ceil(N*alf)-N*alf) * VaR_10d + sum(loss_10d(ceil(N*alf)+1:N)) );

% Compute Normal 10-day VaR from the data
w_stock = (cur_prices(1,:)'.*positions)./total_value(1);%20x1
stock_return_10d = (data_prices(10:end,:)./data_prices(1:end-9,:)-1);%503x20
Q_stock_10d = cov(stock_return_10d);%20x20
port_std_10d = sqrt(w_stock'*Q_stock_10d*w_stock);
port_return_10d = mean(stock_return_10d,1)*w_stock;
VaRn_10d = (-port_return_10d + norminv(alf,0,1)*port_std_10d)*total_value(1);
% Compute Normal 10-day CVaR from the data
CVaRn_10d = (-port_return_10d + (normpdf(norminv(alf,0,1))/(1-alf))*port_std_10d)*total_value(1);

%print results
fprintf('Historical 10-day VaR %4.1f%% = $%6.2f,   Historical 10-day CVaR %4.1f%% = $%6.2f\n', 100*alf, VaR_10d, 100*alf, CVaR_10d);
fprintf('    Normal 10-day VaR %4.1f%% = $%6.2f,       Normal 10-day CVaR %4.1f%% = $%6.2f\n', 100*alf, VaRn_10d, 100*alf, CVaRn_10d);

%% Plot a histogram of the distribution of losses in portfolio value for 1 day 
% figure(1)
figure(1)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(loss_1d, 100);
bar(binLocations, frequencyCounts);
hold on;
line([VaR VaR], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(loss_1d)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(loss_1d))/std(loss_1d)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([CVaR CVaR], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.9*VaR, max(frequencyCounts)/1.9, 'VaR')
text(1.05*CVaR, max(frequencyCounts)/1.9, 'CVaR')
xlabel('1-day loss in $ value on 1 unit of stock')
ylabel('Frequency')

%% Plot a histogram of the distribution of losses in portfolio value for 10 days
% figure(2)
figure(2)
set(gcf, 'color', 'white');
[frequencyCounts, binLocations] = hist(loss_10d, 100);
bar(binLocations, frequencyCounts);
hold on;
line([VaR_10d VaR_10d], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '--');
hold on;
normf = ( 1/(std(loss_10d)*sqrt(2*pi)) ) * exp( -0.5*((binLocations-mean(loss_10d))/std(loss_10d)).^2 );
normf = normf * sum(frequencyCounts)/sum(normf);
plot(binLocations, normf, 'r', 'LineWidth', 3);
hold on;
line([CVaR_10d CVaR_10d], [0 max(frequencyCounts)/2], 'Color', 'r', 'LineWidth', 1, 'LineStyle', '-.');
hold off;
text(0.9*VaR_10d, max(frequencyCounts)/1.9, 'VaR')
text(1.05*CVaR_10d, max(frequencyCounts)/1.9, 'CVaR')
xlabel('10-day loss in $ value on 1 unit of stock')
ylabel('Frequency')
%% part 2-MSFT
%calculate daily return difference
PLData_MSFT = diff(100*cur_prices(:,1));
% Sort loss data in increasing order
loss_1d_MSFT = sort(-PLData_MSFT);
%specify number of scenarios
N = size(PLData_MSFT,1);
% Compute Historical 1-day VaR from the data
VaR_MSFT  = loss_1d_MSFT(ceil(N*alf));
% Compute Historical 1-day CVaR from the data
CVaR_MSFT = (1/(N*(1-alf))) * ( (ceil(N*alf)-N*alf) * VaR_MSFT + sum(loss_1d_MSFT(ceil(N*alf)+1:N)) );
% Compute Normal VaR and Normal CVaR
VaRn_MSFT = mean(loss_1d_MSFT) + norminv(alf,0,1)*std(loss_1d_MSFT);
CVaRn_MSFT = mean(loss_1d_MSFT) + (normpdf(norminv(alf,0,1))/(1-alf))*std(loss_1d_MSFT);
fprintf('Historical 1-day VaR_MSFT %4.1f%% = $%6.2f,   Historical 1-day CVaR_MSFT %4.1f%% = $%6.2f\n', 100*alf, VaR_MSFT, 100*alf, CVaR_MSFT)

%% part 2 - AAPL
%calculate daily return difference
PLData_AAPL = diff(200*cur_prices(:,9));
% Sort loss data in increasing order
loss_1d_AAPL = sort(-PLData_AAPL);
%specify number of scenarios
N = size(PLData_AAPL,1);
% Compute Historical 1-day VaR from the data
VaR_AAPL  = loss_1d_AAPL(ceil(N*alf));
% Compute Historical 1-day CVaR from the data
CVaR_AAPL = (1/(N*(1-alf))) * ( (ceil(N*alf)-N*alf) * VaR_AAPL + sum(loss_1d_AAPL(ceil(N*alf)+1:N)) );
% Compute Normal
VaRn_AAPL = mean(loss_1d_AAPL) + norminv(alf,0,1)*std(loss_1d_AAPL);
CVaRn_AAPL = mean(loss_1d_AAPL) + (normpdf(norminv(alf,0,1))/(1-alf))*std(loss_1d_AAPL);
fprintf('Historical 1-day VaR_AAPL %4.1f%% = $%6.2f,   Historical 1-day CVaR_AAPL %4.1f%% = $%6.2f\n', 100*alf, VaR_AAPL, 100*alf, CVaR_AAPL);

%% part 2 - IBM
%calculate daily return difference
PLData_IBM = diff(500*cur_prices(:,10));
% Sort loss data in increasing order
loss_1d_IBM = sort(-PLData_IBM);
%specify number of scenarios
N = size(PLData_IBM,1);
% Compute Historical 1-day VaR from the data
VaR_IBM  = loss_1d_IBM(ceil(N*alf));
% Compute Historical 1-day CVaR from the data
CVaR_IBM = (1/(N*(1-alf))) * ( (ceil(N*alf)-N*alf) * VaR_IBM + sum(loss_1d_IBM(ceil(N*alf)+1:N)) );
% Compute Norm
VaRn_IBM = mean(loss_1d_IBM) + norminv(alf,0,1)*std(loss_1d_IBM);
CVaRn_IBM = mean(loss_1d_IBM) + (normpdf(norminv(alf,0,1))/(1-alf))*std(loss_1d_IBM);
fprintf('Historical 1-day VaR_IBM %4.1f%% = $%6.2f,   Historical 1-day CVaR_IBM %4.1f%% = $%6.2f\n', 100*alf, VaR_IBM, 100*alf, CVaR_IBM);

fprintf('Historical 1-day VaR_SUM %4.1f%% = $%6.2f,   Historical 1-day CVaR_SUM %4.1f%% = $%6.2f\n', 100*alf, VaR_IBM+VaR_MSFT+VaR_AAPL, 100*alf, CVaR_IBM+CVaR_MSFT+CVaR_AAPL);
fprintf('Normal 1-day VaR_SUM %4.1f%% = $%6.2f,   Normal 1-day CVaR_SUM %4.1f%% = $%6.2f\n', 100*alf, VaRn_IBM+VaRn_MSFT+VaRn_AAPL, 100*alf, CVaRn_IBM+CVaRn_MSFT+CVaRn_AAPL);

%find the correlation (To prove there are correlation among the three
%assets
correlation = corr(stock_return_1d);

%% Question 2

% Annual risk-free rate for years 2015-2016 is 2.5%
r_rf = 0.025;
% Cplex solving Mean-variance
n = 20;

% Optimization problem data
lb = zeros(n,1);
ub = inf*ones(n,1);
A  = ones(1,n);
b  = 1;

% Compute minimum variance portfolio
cplex1 = Cplex('min_Variance');
cplex1.addCols(zeros(n,1), [], lb, ub);
cplex1.addRows(b, A, b);
cplex1.Model.Q = 2*Q;
cplex1.Param.qpmethod.Cur = 6; % concurrent algorithm
cplex1.Param.barrier.crossover.Cur = 1; % enable crossover
cplex1.DisplayFunc = []; % disable output to screen
cplex1.solve();

% Display minimum variance portfolio
w_minVar = cplex1.Solution.x;
var_minVar = w_minVar' * Q * w_minVar;
ret_minVar = mu' * w_minVar;
fprintf ('Minimum variance portfolio:\n');
fprintf ('Solution status = %s\n', cplex1.Solution.statusstring);
fprintf ('Solution value = %f\n', cplex1.Solution.objval);
fprintf ('Return = %f\n', ret_minVar);
fprintf ('Standard deviation = %f\n\n', sqrt(var_minVar));

% Compute maximum return portfolio
cplex2 = Cplex('max_Return');
cplex2.Model.sense = 'maximize';
cplex2.addCols(mu, [], lb, ub);
cplex2.addRows(b, A, b);
cplex2.Param.lpmethod.Cur = 6; % concurrent algorithm
cplex2.Param.barrier.crossover.Cur = 1; % enable crossover
cplex2.DisplayFunc = []; % disable output to screen
cplex2.solve();

% Display maximum return portfolio
w_maxRet = cplex2.Solution.x;
var_maxRet = w_maxRet' * Q * w_maxRet;
ret_maxRet = mu' * w_maxRet;
fprintf ('Maximum return portfolio:\n');
fprintf ('Solution status = %s\n', cplex2.Solution.statusstring);
fprintf ('Solution value = %f\n', cplex2.Solution.objval);
fprintf ('Return = %f\n', ret_maxRet);
fprintf ('Standard deviation = %f\n\n', sqrt(var_maxRet));

% Target returns
targetRet = linspace(ret_minVar,ret_maxRet,20);

% Compute efficient frontier
cplex3 = Cplex('Efficient_Frontier');
cplex3.addCols(zeros(n,1), [], lb, ub);
cplex3.addRows(targetRet(1), mu', inf);
cplex3.addRows(b, A, b);
cplex3.Model.Q = 2*Q;
cplex3.Param.qpmethod.Cur = 6; % concurrent algorithm
cplex3.Param.barrier.crossover.Cur = 1; % enable crossover
cplex3.DisplayFunc = []; % disable output to screen

w_front = [];
for i=1:length(targetRet)
    cplex3.Model.lhs(1) = targetRet(i);
    cplex3.solve();
    w_front = [w_front cplex3.Solution.x];
    var_front(i) = w_front(:,i)' * Q * w_front(:,i);
    ret_front(i) = mu' * w_front(:,i);
end

%% Initial portfolio 
init_positions = [5000 950 2000 0 0 0 0 2000 3000 1500 0 0 0 0 0 0 1001 0 0 0]';
init_value = data_prices(day_ind_end+1,:) * init_positions;
w_init = (data_prices(day_ind_end+1,:) .* init_positions')' / init_value;
var_init = w_init' * Q * w_init;
ret_init = mu' * w_init;

% Equal weight
w_equal = ones(20,1)*(1/20);
var_equal = w_equal' * Q * w_equal;
ret_equal = mu' * w_equal;
% Max Sharpe Ratio portfolio weights
w_Sharpe = [ 0 0 0 0 0 0 0 0.385948690661642 0.172970428625544 0 0 0 0 0 0.003409676869715 0.260942060896445 0 0.185966939781285 0 0]';
var_Sharpe = w_Sharpe' * Q * w_Sharpe;
ret_Sharpe = mu' * w_Sharpe;
% Equal Risk Contribution portfolio weights
w_ERC = [0.049946771209069 0.049951626261681 0.049955739901370 0.049998404150207 0.050000297368719 0.050004255546315 0.050006307026730 0.050007308995726 0.050010525832832 0.050013840015521 0.050014404492514 0.050015932843104 0.050016630302524 0.050017212457105 0.050017600497611 0.050017998351827 0.050018997074443 0.050019598350121 0.050019778113513 0.049946771209069]';
var_ERC = w_ERC' * Q * w_ERC;
ret_ERC = mu' * w_ERC;

%Leveraged equal risk
w_lever = w_ERC;
var_lever = w_lever' * Q * w_lever;
ret_lever = (2*mu'-((1+r_rf)^(1/6)-1))* w_lever;

%% Plotting Q2-part 1
figure(9);
set(gcf, 'color', 'white');

plot(sqrt(var_front), ret_front, 'k-', 'LineWidth', 3)%efficient frontier
hold on;
plot(sqrt(var_minVar), ret_minVar, 'rd', 'MarkerSize', 10)%minVar
hold on;
plot(sqrt(var_maxRet), ret_maxRet, 'ms', 'MarkerSize', 10)%maxReturn
hold on;
plot(sqrt(var_init), ret_init, 'bo', 'MarkerSize', 10)%initial
hold on;
plot(sqrt(var_equal), ret_equal, 'rs', 'MarkerSize', 10)%equal weight
hold on;
plot(sqrt(var_Sharpe), ret_Sharpe, 'g+', 'MarkerSize', 10)%Sharpe
hold on;
plot(sqrt(var_ERC), ret_ERC, 'b*', 'MarkerSize', 10)%Equal Risk
hold on;
plot(sqrt(var_lever), ret_lever, 'bs', 'MarkerSize', 10)%Leverage
hold on;
plot(0, r_rf/252, 'kd', 'MarkerSize', 10)%risk free
hold on;

plot([0 sqrt(var_Sharpe)*3],[r_rf/252 ret_Sharpe*3],'--b','LineWidth',1);

xlabel('Standard deviation');
ylabel('Expected return');
title('Efficient Frontier')
legend('efficient frontier', 'minVar', 'maxReturn', 'initalPort','EqualW','MaxSharpe','ERP','Lever','Rf','Tangent Line', 'Location', 'NorthWest')


%% Plot for Question 2, Part 2
figure(10);
set(gcf, 'color', 'white');
port = unifrnd(0,1,1000,20);
%ensure the sum of weight = 1
w_random = port./sum(port,2);
return_random = w_random*mu;
var_random = diag(w_random*Q*w_random');

plot(sqrt(var_front), ret_front, 'k-', 'LineWidth', 3)%efficient frontier
hold on;
plot(sqrt(diag(Q)), mu, 'b.', 'MarkerSize', 18)
hold on;
plot(sqrt(var_random), return_random, 'g.', 'MarkerSize', 8)%Equal Risk

legend('efficient frontier','individual asset','random portfolio', 'Location', 'SouthEast')
%%Conclusion
%all of the randomly genereated portfolio are not optimum (e.g.below efficient frontier) and all of the 18
%stocks are below efficient frontier; so any one-asset portfolio will not
%be optimum either
