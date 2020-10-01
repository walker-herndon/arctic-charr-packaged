#### Retrieving which re-identified fish were previously PIT-tagged and which were never PIT-tagged
# 19th September 2020
# Lizy

#### Data for CJS model
library(readxl)
# Capture histories
C10_old <- read_excel("Arctic_charr_re_id_caves_a_and_b.xlsx", sheet = "Wide form C10 (old tags)")
C21_old <- read_excel("Arctic_charr_re_id_caves_a_and_b.xlsx", sheet = "Wide form C21 (old tags)")

C10_new <- read_excel("Arctic_charr_re_id_caves_a_and_b.xlsx", sheet = "Wide form C10 (new tags)")
C21_new <- read_excel("Arctic_charr_re_id_caves_a_and_b.xlsx", sheet = "Wide form C21 (new tags)")

old <- as.data.frame(rbind(C10_old,C21_old))
new <- as.data.frame(rbind(C10_new,C21_new))

# Make sure that the tag that was corrected since receiving the file from Ignacy is corrected here
new$Tag[which(new$Tag=="CAL15-2427")] <- "655177"
new[which(new$Tag=="655177"),]
# Merge capture histories
new[237,"2015 June"] <- 1
new <- new[-227,]

old <- old[which(!is.na(old$Tag)),]
new <- new[which(!is.na(new$Tag)),]
new <- new[-c(153,313),]
old <- old[-171,]
rownames(new) <- NULL

duplicated(new$Tag) # There is one.
new[which(duplicated(new$Tag)==TRUE),]
new[which(new$Tag == "123845"),]
new[117,"2017 August"] <- 1
new <- new[-265,]

# Save with June 2012 for third version run 
new2 <- new
old2 <- old
# Remove that June 2012 from both
new <- new[,-2]
new <- new[rowSums(new[,2:16])!=0,] # 265 individuals
old <- old[,-2]
old <- old[rowSums(old[,2:16])!=0,] # 291 individuals

# when each individual was first captured
first.old <- array(dim=dim(old)[1])
first.new <- array(dim=dim(new)[1])
first.new2 <- array(dim=dim(new2)[1])
for(i in 1:(dim(old)[1])){
	first.old[i] <- min(which(old[i,2:(dim(old)[2])]==1))
}
for(i in 1:(dim(new)[1])){
	first.new[i] <- min(which(new[i,2:(dim(new)[2])]==1))
}
for(i in 1:(dim(new2)[1])){
	first.new2[i] <- min(which(new2[i,2:(dim(new2)[2])]==1))
}

# There is no information about survival from individuals that are captured for the first time at the last occasion so remove them
k <- dim(old)[2]-1
k2 <- dim(new2)[2]-1
old <- old[which(first.old<k),]
first.old <- first.old[which(first.old<k)]
new <- new[which(first.new<k),]
first.new <- first.new[which(first.new<k)]
new2 <- new2[which(first.new2<k2),]
first.new2 <- first.new2[which(first.new2<k2)]

# To have a non-zero likelihood, we need starting configurations of a that are consistent with y
a.init.old <- array(NA,dim=dim(old))
a.init.new <- array(NA,dim=dim(new))
a.init.new2 <- array(NA,dim=dim(new2))
for(i in 1:(dim(old)[1])){
	a.init.old[i,(first.old[i]:max(which(old[i,2:(dim(old)[2])]==1)))+1]<-1
}
for(i in 1:(dim(new)[1])){
	a.init.new[i,(first.new[i]:max(which(new[i,2:(dim(new)[2])]==1)))+1]<-1
}
for(i in 1:(dim(new2)[1])){
	a.init.new2[i,(first.new2[i]:max(which(new2[i,2:(dim(new2)[2])]==1)))+1]<-1
}

# CMR analysis to estimate survival and capture rates of old data
library(rjags)
CJS<-"model{
# uniform priors
phi~dunif(0,1)
p~dunif(0,1)
# likelihood is over all individuals
for(i in 1:n){
	a[i,first[i]]~dbern(1)
	for(j in (first[i]+1):k){
		# process model for survival, given previously being alive
		a[i,j]~dbern(a[i,j-1]*phi)
		# state model for being observed, given being alive
		y[i,j]~dbern(a[i,j]*p)
		}
	}
}"
writeLines(CJS,con=file("./CJS.jags"))
d1 <- list(y=old[,2:(k+1)],n=length(first.old),first=first.old,k=k)
m1 <- jags.model(file="./CJS.jags",data=d1,inits=list(a=as.matrix(a.init.old[,2:(k+1)])))
s1 <- jags.samples(m1,variable.names=c("phi","p"),n.iter=10000,thin=10)

# Check model is well mixed
par(mfrow=c(1,2))
plot(s1$phi[1,,1])
plot(s1$p[1,,1])

# CMR analysis to estimate survival and capture rates of new data
d2<-list(y=new[,2:(k+1)],n=length(first.new),first=first.new,k=k)
m2<-jags.model(file="./CJS.jags",data=d2,inits=list(a=as.matrix(a.init.new[,2:(k+1)])))
s2<-jags.samples(m2,variable.names=c("phi","p"),n.iter=10000,thin=10)
par(mfrow=c(1,2))
plot(s2$phi[1,,1])
plot(s2$p[1,,1])

# CMR analysis to estimate survival and capture rates of new data including first occasion
d3<-list(y=new2[,2:(k+1)],n=length(first.new2),first=first.new2,k=k)
m3<-jags.model(file="./CJS.jags",data=d3,inits=list(a=as.matrix(a.init.new2[,2:(k+1)])))
s3<-jags.samples(m3,variable.names=c("phi","p"),n.iter=10000,thin=10)
par(mfrow=c(1,2))
plot(s3$phi[1,,1])
plot(s3$p[1,,1])

meanAndCI<-function(x){c(mean(x),HPDinterval(as.mcmc(x)))}
meanAndCI(s1$phi[1,,1])
meanAndCI(s1$p[1,,1])
meanAndCI(s2$phi[1,,1])
meanAndCI(s2$p[1,,1])
meanAndCI(s3$phi[1,,1])
meanAndCI(s3$p[1,,1])