[<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/banner.png" width="888" alt="Visit QuantNet">](http://quantlet.de/)

## [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/qloqo.png" alt="Visit QuantNet">](http://quantlet.de/) **MVAnmdscar2** [<img src="https://github.com/QuantLet/Styleguide-and-FAQ/blob/master/pictures/QN2.png" width="60" alt="Visit QuantNet 2.0">](http://quantlet.de/)

```yaml

Name of QuantLet: MVAnmdscar2

Published in: Applied Multivariate Statistical Analysis

Description: Illustrates the PAV algorithm for nonmetric MDS for car brands data.

Keywords: MDS, non-metric-MDS, multi-dimensional, scaling, PAV, violators, plot, graphical representation, scatterplot, sas

See also: MVAMDScity1, MVAMDScity2, MVAMDSnonmstart, MVAMDSpooladj, MVAmdscarm, MVAnmdscar1, MVAnmdscar3, MVAcarrankings, PAVAlgo

Author: Zografia Anastasiadou
Author[SAS]: Svetlana Bykovskaya
Author[Matlab]: Wolfgang Haerdle, Vladimir Georgescu, Song Song

Submitted: Tue, January 11 2011 by Zografia Anastasiadou   
Submitted[SAS]: Tue, April 05 2016 by Svetlana Bykovskaya
Submitted[Matlab]: Mon, December 19 by Piedad Castro

Example: Scatterplot of dissimilarities against distances.

```

![Picture1](MVAnmdscar2_matlab.png)

![Picture2](MVAnmdscar2_r.png)

![Picture3](MVAnmdscar2_sas.png)

### MATLAB Code
```matlab

%% clear all variables and console and close windows
clear
clc
close all

x = [[3 2 1 10]; [2 7 3 4]];

d = dist(x);

delta = [1 2 3 d(1,2)
         1 3 2 d(1,3)
         1 4 5 d(1,4)
         2 3 1 d(2,3)
         2 4 4 d(2,4)
         3 4 6 d(3,4)];

fig = [1 d(2,3)   
       2 d(1,3)   
       3 d(1,2)  
       4 d(2,4)   
       5 d(1,4)   
       6 d(3,4)];
  
%% plot
scatter(fig(:, 1), fig(:, 2), 'b', 's', 'fill')
xlim([0 7])
ylim([0 10])
title('Dissimilarities and Distances')
xlabel('Dissimilarity')
ylabel('Distance')
for i=1:5
    line([fig(i, 1) fig(i + 1, 1)], [fig(i, 2) fig(i + 1, 2)],...
         'Color', 'k', 'LineWidth',1.5)
end
labels = {'(2,3)', '(1,3)', '(1,2)', '(2,4)', '(1,4)', '(3,4)'};
text(fig(:, 1) + 0.2, fig(:, 2), labels, 'Color', 'r')

```

automatically created on 2018-05-28

### R Code
```r


# clear all variables
rm(list = ls(all = TRUE))
graphics.off()

x = cbind(c(3, 2, 1, 10), c(2, 7, 3, 4))

d = as.matrix(dist(x))

d1 = c(1, 2, 3, d[1, 2])
d2 = c(1, 3, 2, d[1, 3])
d3 = c(1, 4, 5, d[1, 4])
d4 = c(2, 3, 1, d[2, 3])
d5 = c(2, 4, 4, d[2, 4])
d6 = c(3, 4, 6, d[3, 4])
delta = cbind(d1, d2, d3, d4, d5, d6)

f1 = c(1, d[2, 3])
f2 = c(2, d[1, 3])
f3 = c(3, d[1, 2])
f4 = c(4, d[2, 4])
f5 = c(5, d[1, 4])
f6 = c(6, d[3, 4])
fig = rbind(f1, f2, f3, f4, f5, f6)

# plot
plot(fig, pch = 15, col = "blue", xlim = c(0, 7), ylim = c(0, 10), xlab = "Dissimilarity", 
    ylab = "Distance", main = "Dissimilarities and Distances", cex.axis = 1.2, cex.lab = 1.2, 
    cex.main = 1.8)
lines(fig, lwd = 3)
text(fig, labels = c("(2,3)", "(1,3)", "(1,2)", "(2,4)", "(1,4)", "(3,4)"), pos = 4, 
    col = "red") 

```

automatically created on 2018-05-28

### SAS Code
```sas

proc iml;
  x   = {3, 2, 1, 10} || {2, 7, 3, 4};
  d   = distance(x);
  
  f1  = {1} || d[2, 3];
  f2  = {2} || d[1, 3];
  f3  = {3} || d[1, 2];
  f4  = {4} || d[2, 4];
  f5  = {5} || d[1, 4];
  f6  = {6} || d[3, 4];
  fig = f1 // f2 // f3 // f4 // f5 // f6;
  
  x1  = fig[,1];
  x2  = fig[,2];
  points = {'(2,3)', '(1,3)', '(1,2)', '(2,4)', '(1,4)', '(3,4)'};
	
  create plot var {"x1" "x2" "points"};
    append;
  close plot;
quit;

proc sgplot data = plot
    noautolegend;
  title 'Dissimilarities and Distances';
  scatter x = x1 y = x2 / datalabel = points 
    datalabelattrs = (color = red) datalabelpos = right
    markerattrs = (symbol = squarefilled);
  series  x = x1 y = x2 / lineattrs = (color = black THICKNESS = 2);
  xaxis min = 0 max = 7  label = 'Dissimilarity';
  yaxis min = 0 max = 10 label = 'Distance'; 
run;
  
```

automatically created on 2018-05-28