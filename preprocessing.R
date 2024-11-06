
#===============================================================================================
#=========================================================================================

#       Homogenisation of external validation dataset

#=========================================================================================
#===============================================================================================

train_expMat  # load train expression dataframe, with HUGO gene names in rows and samples in columns
extVal_expMat  # load external validation expression dataframe (other platform), with HUGO gene names in rows and samples in columns

#=====================
# GQ functions
#========================

# from publication: Bioinformatics 2020   https://doi.org/10.1093/bioinformatics/btz974
# MatchMixeR: a cross-platform normalization method for gene expression data integration
# Serin Zhang, Jiang Shao, Disa Yu, Xing Qiu,and Jinfeng Zhang

# https://github.com/dy16b/Cross-Platform-Normalization 


# gq function
#-------------

gq = function(platform1.data, platform2.data, p1.names=0, p2.names=0, skip.match=FALSE){
  #This function is basically a wrapper for normalizeGQ
  
  #Match names
  input = processplatforms(list(x=platform1.data,y=platform2.data),namesvec = c(p1.names, p2.names), skip.match=skip.match)
  
  #Prepare for normalizeGQ
  combined = cbind(input$x,input$y)
  pf = c(seq(1,1,length.out=dim(input$x)[2]),seq(2,2,length.out=dim(input$y)[2]))
  
  #Call normalizeGQ
  ngq = normalizeGQ(combined,pf)
  
  #Split the results and return
  out=split(seq(pf),pf)
  out[[1]] = ngq[,out[[1]]]
  out[[2]] = ngq[,out[[2]]]
  names(out) <- c("x","y")
  return(out)
}

# normalizeGQ function
#------------------------

normalizeGQ <- function(M, pf) { 
  #This function was provided by Xiao-Qin Xia, one of the authors of 
  #webarraydb.
  # modified MRS
  # M is the data matrix
  # pf is the vector to specify the platform for each column of M.
  idx <- split(seq(pf), pf)
  if (length(pf)<=1) return(M)
  imax <- which.max(sapply(idx, length)) # for reference
  ref_med <- apply(M[, idx[[imax]]], 1, function(x) median(x, na.rm=TRUE))
  ref_med_srt <- sort(ref_med)
  idx[imax] <- NULL
  lapply(idx, function(i) {
    MTMP <- sapply(i, function(x) ref_med_srt[rank(M[,x])]); 
    M[,i] <<- MTMP - apply(MTMP, 1, median) + ref_med 
  } )
  invisible(M)
}

# processplatforms function
#--------------------------
         
processplatforms = function(datalist, namesvec=NULL, skip.match=FALSE){
  #Convert data from various formats to the proper format for use 
  #with all the crossnorm normalization functions
  
  for(i in 1:length(datalist)){
    if(is.matrix(datalist[[i]])){
      datalist[[i]] <- as.data.frame(datalist[[i]])
    }
  }
  
  if (is.null(namesvec)){
    namesvec <- numeric(length(datalist))
    for (i in 1:length(datalist)){
      namesvec[i] <- 0
    }
  }
  
  #Put the row names in their places
  for (i in 1:length(namesvec)){
    if(namesvec[i] != 0){
      rownames(datalist[[i]]) = datalist[[i]][,namesvec[i]]
      datalist[[i]] = datalist[[i]][,-1*namesvec[i],drop=FALSE]
    }	
  }
  
  if(!skip.match){
    #Create the common genes list
    commongenes <- rownames(datalist[[1]])
    for (i in 2:length(datalist)){
      commongenes <- intersect(commongenes,rownames(datalist[[i]]))
    }
    
    
    #Put it all together
    for (i in 1:length(datalist)){
      datalist[[i]] <- datalist[[i]][commongenes,,drop=FALSE]
    }
  }
  return(datalist)
}



#=================================
# cross_platfrom normalization 
#===============================

ComGenes <- Reduce(intersect, lapply(list(train_expMat,extVal_expMat) , rownames))
train_expMat <- train_expMat[rownames(train_expMat) %in% ComGenes,]
extVal_expMat <- extVal_expMat[rownames(extVal_expMat) %in% ComGenes,]

extVal_expMat_GQ<- gq(platform1.data= train_expMat, platform2.data=extVal_expMat,p1.names=0, p2.names=0, skip.match=T)        
extVal_expMat_GQ <- extVal_expMat_GQ$y






         
         
