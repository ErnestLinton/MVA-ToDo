[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MVApcabank** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: MVApcabank

Published in: Applied Multivariate Statistical Analysis

Description: 'Performs a PCA for the Swiss bank notes and shows the first three principal components in two-dimensional scatterplots. Additionally, a screeplot of the eigenvalues is displayed.'

Keywords: 'principal-components, pca, eigenvalues, screeplot, scatterplot, plot, graphical representation, data visualization, sas'

See also: MVAnpcabanki, MVAnpcabank, MVAnpcahousi, MVAnpcatime, MVAnpcafood, MVAnpcausco, MVAnpcausco2, MVAnpcausco2i, MVAcpcaiv, MVAnpcahous, MVApcabanki, MVApcabankr, MVApcasimu

Author: Zografia Anastasiadou
Author[SAS]: Svetlana Bykovskaya
Author[Matlab]: Wolfgang Haerdle, Jorge Patron, Vladimir Georgescu, Song Song

Submitted: Mon, December 12 2016 by Piedad Castro
Submitted[SAS]: Tue, April 05 2016 by Svetlana Bykovskaya
Submitted[Matlab]: Mon, December 12 2016 by Piedad Castro

Datafile: bank2.dat

Note: 'Matlab and SAS decompose matrices differently than R, and therefore some 
      of the eigenvectors may have different signs.'

```

![Picture1](MVApcabank-1.png)

![Picture2](MVApcabank-1_matlab.png)

![Picture3](MVApcabank-1_sas.png)

![Picture4](MVApcabank-2_sas.png)

![Picture5](MVApcabank-3_sas.png)

![Picture6](MVApcabank-4_sas.png)

### MATLAB Code
```matlab

%% clear all variables and console and close windows
clear
clc
close all

%% load data
x = load('bank2.dat');
[n,p]  = size(x);

groups = vertcat(ones(100,1),zeros(100,1)); % Creates a vector with ones in the first 100 entries and zeros in the rest
                                            % This vector enables us to use the 'gscatter'command to plot the data in groups.
adjust = (n-1)*cov(x)/n;
[v,e]  = eigs(adjust,p,'la');    % Calculates eigenvalues and eigenvectors and sorts them by size
e1     = diag(e);                % Creates column vector of eigenvalues

% change the sign of the eigenvectors . This is done only to make easier
% the comparison with R results.
v = -v;                     

x = x*v;                    % Data multiplied by eigenvectors

%% Plots
nr = 1:6;
%% plot of the eigenvalues
subplot(2,2,4,'FontSize',20)
scatter(nr,e1,25,'k')
xlabel('Index')
ylabel('Lambda')
title('Eigenvalues of S')
xlim([0.5 6.5])
ylim([-0.2 3.5])
 
%% Plot of the first vs. second PC
subplot(2,2,1,'FontSize',20)
gscatter(x(:,1),x(:,2),groups,'br','+o',5,'off')
xlabel('PC1 ')
ylabel('PC2 ')
title('First vs. Second PC')

%% Plot of the second vs. third PC
subplot(2,2,2,'FontSize',20)
gscatter(x(:,2),x(:,3),groups,'br','+o',5,'off')
xlabel('PC2 ')
ylabel('PC3 ')
title('Second vs. Third PC')

%% Plot of the first vs. third PC
subplot(2,2,3,'FontSize',20)
gscatter(x(:,1),x(:,3),groups,'br','+o',5,'off')
xlabel('PC1 ')
ylabel('PC3 ')
title('First vs. Third PC')
```

automatically created on 2018-05-28

### R Code
```r


# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

# load data
x = read.table("bank2.dat")
n = nrow(x)

e  = eigen((n - 1) * cov(x)/n)   # calculates eigenvalues and eigenvectors and sorts them by size
e1 = e$values
x  = as.matrix(x) %*% e$vectors  # data multiplied by eigenvectors

# plot of the first vs. second PC
par(mfrow = c(2, 2))
plot(x[, 1], x[, 2], pch = c(rep(1, 100), rep(3, 100)), col = c(rep("blue", 100), 
    rep("red", 100)), xlab = "PC1", ylab = "PC2", main = "First vs. Second PC", cex.lab = 1.2, 
    cex.axis = 1.2, cex.main = 1.8)

# plot of the second vs. third PC
plot(x[, 2], x[, 3], pch = c(rep(1, 100), rep(3, 100)), col = c(rep("blue", 100), 
    rep("red", 100)), ylim = c(-242, -238), xlab = "PC2", ylab = "PC3", main = "Second vs. Third PC", 
    cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8)

# plot of the first vs. third PC
plot(x[, 1], x[, 3], pch = c(rep(1, 100), rep(3, 100)), col = c(rep("blue", 100), 
    rep("red", 100)), ylim = c(-241, -238), xlab = "PC1", ylab = "PC3", main = "First vs. Third PC", 
    cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8)

# plot of the eigenvalues
plot(e1, ylim = c(0, 3), xlab = "Index", ylab = "Lambda", main = "Eigenvalues of S", 
    cex.lab = 1.2, cex.axis = 1.2, cex.main = 1.8) 

```

automatically created on 2018-05-28

### SAS Code
```sas


* Import the data;
data bank2;
  infile '/folders/myfolders/data/bank2.dat';
  input temp1-temp6;
run;

proc iml;
  * Read data into a matrix;
  use bank2;
    read all var _ALL_ into x; 
  close bank2;
  
  n  = nrow(x);
  e  = (n - 1) * cov(x)/n; * calculates eigenvalues and eigenvectors and sorts them by size; 
  e1 = 1:6;
  e2 = eigval(e);
  id = (repeat ({1}, 1, 100) || repeat ({2}, 1, 100))`;
  x  = x * eigvec(e);      * data multiplied by eigenvectors;
  
  x1 = x[,1];
  x2 = x[,2];
  x3 = x[,3];
  create plot var {"x1" "x2" "x3" "e1" "e2" "id"};
    append;
  close plot;
quit;

proc sgplot data = plot
    noautolegend;
  title 'First vs. Second PC';
  scatter x = x1 y = x2 / colorresponse = id colormodel = (blue red);
  xaxis label = 'PC1';
  yaxis label = 'PC2';
run;

proc sgplot data = plot
    noautolegend;
  title 'Second vs. Third PC';
  scatter x = x2 y = x3 / colorresponse = id colormodel = (blue red);
  xaxis label = 'PC2';
  yaxis label = 'PC3';
run;

proc sgplot data = plot
    noautolegend;
  title 'First vs. Third PC';
  scatter x = x1 y = x3 / colorresponse = id colormodel = (blue red);
  xaxis label = 'PC1';
  yaxis label = 'PC3';
run;

proc sgplot data = plot
    noautolegend;
  title 'Eigenvalues of S';
  scatter x = e1 y = e2 / markerattrs = (color = blue);
  xaxis label = 'Index';
  yaxis label = 'Lambda';
run;


```

automatically created on 2018-05-28