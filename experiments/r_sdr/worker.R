args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 9L) stop("expected METHOD X Y BASIS META SEED DIM ITER TOL/SLICES")
source(file.path(dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE)[1]))),"methods.R"))
suppressPackageStartupMessages(library(MAVE))
method <- args[1]; x <- as.matrix(read.csv(args[2],header=FALSE))
y <- as.numeric(read.csv(args[3],header=FALSE)[[1]])
m <- as.integer(args[7]); iterations <- as.integer(args[8])
set.seed(as.integer(args[6]))
if (length(y) != nrow(x) || !all(is.finite(x)) || !all(is.finite(y))) stop("invalid data")
started <- proc.time()[["elapsed"]]
result <- switch(method,
  "MAVE-package" = {
    fit <- MAVE::mave.compute(x,y,method="MEANMAVE",max.dim=m,screen=ncol(x))
    list(basis=ortho(coef(fit,dim=m),m),iterations=NA,converged=NA,
         stop_reason="package_not_exposed", detail="MAVE::MEANMAVE;screen=d")
  },
  MAVE = refined_fit(x,y,m,FALSE,iterations,as.numeric(args[9]),ambient=TRUE),
  SIR = sir_fit(x,y,m,as.integer(args[9])),
  rMAVE = refined_fit(x,y,m,FALSE,iterations,as.numeric(args[9])),
  dMAVE = refined_fit(x,y,m,TRUE,iterations,as.numeric(args[9])),
  stop("unknown method")
)
elapsed <- proc.time()[["elapsed"]] - started
write.table(result$basis,args[4],sep=",",row.names=FALSE,col.names=FALSE,quote=FALSE)
if (!is.null(result$trace)) {
  write.table(result$trace,paste0(args[5],".trace.csv"),sep=",",row.names=FALSE)
}
metadata <- data.frame(fit_seconds=elapsed,iterations=result$iterations,
  converged=result$converged,stop_reason=result$stop_reason,detail=result$detail)
write.table(metadata,args[5],sep="\t",row.names=FALSE,col.names=TRUE,quote=TRUE,na="")
