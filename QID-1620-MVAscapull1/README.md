[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MVAscapull1** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: MVAscapull1

Published in: Applied Multivariate Statistical Analysis

Description: Computes a two dimensional scatterplot of price and sales from the pullovers data set.

Keywords: data visualization, graphical representation, scatterplot, financial, plot, sas

See also: MVAcorrnorm, MVAregbank, MVAregpull, MVAscabank45, MVAscabank56, MVAscabank456, MVAscacar, MVAscapull2, MVAdraftbank4, MVAdrafthousing, MVAdrafthousingt

Author: Vladimir Georgescu, Jorge Patron, Song Song, Awdesch Melzer
Author[SAS]: Svetlana Bykovskaya

Submitted: Mon, September 15 2014 by Awdesch Melzer
Submitted[SAS]: Wen, April 6 2016 by Svetlana Bykovskaya
Submitted[Matlab]: Tue, November 22 2016 by Lily Medina

Datafile: pullover.dat

```

![Picture1](MVAscapull1_1.png)

![Picture2](MVAscapull1_matlab.png)

![Picture3](MVAscapull1_sas.png)

### MATLAB Code
```matlab

%% clear all variables
clear
clc
close all

%% load data
x = load('pullover.dat');    

%% plot
scatter(x(:,2),x(:,1),25,'k','SizeData',75)
title('Pullovers Data')
xlabel('Price (X2)')
ylabel('Sales (X1)')
xlim([78 127])
 
```

automatically created on 2018-05-28

### R Code
```r


# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

# load data
x = read.table("pullover.dat")

# plot
plot(x[, 2], x[, 1], main = "Pullovers Data", ylab = "Sales (X1)", xlab = "Price (X2)", 
    xlim = c(78, 127), ylim = c(80, 240), frame = TRUE, axes = FALSE)
axis(side = 2, seq(80, 240, 40), seq(80, 240, 40))
axis(side = 1, seq(80, 130, 10), seq(80, 130, 10))

```

automatically created on 2018-05-28

### SAS Code
```sas


* Import the data;
data pull;
  infile '/folders/myfolders/Sas-work/data/pullover.dat';
  input x1-x4; 
run;

proc sgplot data = pull
    noautolegend;
  title 'Pullovers Data';
  scatter x = x2 y = x1 / markerattrs = (color = blue);
  xaxis min = 78 max = 127 label = 'Price (X2)';
  yaxis min = 80 max = 240 label = 'Sales (X1)';
run;
```

automatically created on 2018-05-28