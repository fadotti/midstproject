x1a = rnorm(5000, 0, 1)
x2a = rnorm(5000, 0, 1)
plot(x1a,x2a, ylim=c(-7,7), xlim=c(-7,7))

x1b = rnorm(5000, 3.5, 1)
x2b = rnorm(5000, 3.5, 1)
points(x1b,x2b,col=2)

g1=cbind(x1a,x2a, rep(1,length(x1a)))
g1
g2=cbind(x1b,x2b, rep(-1,length(x1a)))
g2

g = data.frame(x1=numeric(0),x2=numeric(0),y=numeric(0))
for(i in 1:nrow(g1)){
  g = rbind(g,g1[i,],g2[i,])
}
write.table(g, "./data1.txt", sep="\t", col.names=F, row.names = F)
##################################################

xa=c(rbinom(10, 1, 0.2),1)
xb=c(rbinom(10, 1, 0.4),-1)
for (i in 1:4999){
  ra = c(rbinom(10, 1, 0.2),1)
  rb = c(rbinom(10, 1, 0.4),-1)
  xa=rbind(xa,ra)
  xb=rbind(xb,rb)
}

d=rbind(xa,xb)
#write.table(d, "C:/Users/Filippo/Desktop/perceptron/sparse.txt", sep="\t", col.names=F, row.names=F)
