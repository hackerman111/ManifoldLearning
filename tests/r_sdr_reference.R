source("benchmarks/experiments/r_sdr/methods.R")
set.seed(29)
x <- matrix(rnorm(240), 60, 4) %*% diag(c(0.3, 1, 2, 4))
y <- x[, 1] + 0.3 * x[, 2] + rnorm(60, sd=0.2)

# Independent SIR reference from full covariance and slice means.
xc <- scale(x, scale=FALSE)
e <- eigen(crossprod(xc) / nrow(x), symmetric=TRUE)
w <- e$vectors %*% diag(1 / sqrt(e$values)) %*% t(e$vectors)
z <- xc %*% w
ends <- ceiling(seq_len(8) * length(y) / 8)
groups <- split(order(y), cut(seq_along(y), c(0,ends)))
means <- sapply(groups, function(ids) colMeans(z[ids,,drop=FALSE]))
kernel <- means %*% diag(lengths(groups) / length(y)) %*% t(means)
expected <- w %*% eigen(kernel, symmetric=TRUE)$vectors[,1,drop=FALSE]
actual <- sir_fit(x, y, 1, 8)$basis
overlap <- sum(crossprod(ortho(expected, 1), actual)^2)
stopifnot(abs(overlap - 1) < 1e-10)
bad <- try(sir_fit(cbind(x, x[,1]), y, 1, 8), silent=TRUE)
stopifnot(inherits(bad, "try-error"))

a1 <- matrix(rnorm(21), 7, 3); a2 <- matrix(rnorm(18), 6, 3)
r1 <- rnorm(7); r2 <- rnorm(6)
state <- qr_merge(NULL, a1, r1)
state <- qr_merge(state, a2, r2)
stopifnot(max(abs(qr.solve(state$a, state$y) - qr.solve(rbind(a1,a2), c(r1,r2)))) < 1e-11)

# Joint density-response update must equal one stacked LS over all j,k.
x <- x[1:14,1:3]; b <- ortho(matrix(rnorm(6),3,2),2)
f <- outer(y[1:14], y[1:5], function(u,v) dnorm((u-v)/0.7)/0.7)
response <- function(ids) f[,ids,drop=FALSE]
h <- 1.4; aa <- list(); yy <- list(); index <- 0L
for (j in seq_len(nrow(x))) {
  dx <- sweep(x, 2, x[j,], "-")
  wj <- exp(-rowSums((dx %*% b)^2) / (2*h*h))
  wj <- sqrt(wj / sum(wj))
  coeff <- qr.solve(cbind(1,dx %*% b) * wj, f * wj)
  for (k in seq_len(ncol(f))) {
    index <- index + 1L
    aa[[index]] <- kronecker(t(coeff[-1,k]), dx) * wj
    yy[[index]] <- (f[,k] - coeff[1,k]) * wj
  }
}
expected <- qr.solve(do.call(rbind,aa), unlist(yy))
actual <- mave_step(x,b,response,ncol(f),h,block=2)$raw
stopifnot(max(abs(as.vector(actual) - expected)) < 1e-9)
stopifnot(max(abs(mave_step(x,b,response,ncol(f),h,block=5)$raw - actual)) < 1e-9)
# Ordinary MAVE must localize in full X, independently of the current B.
aa <- list(); yy <- list(); index <- 0L
for (j in seq_len(nrow(x))) {
  dx <- sweep(x,2,x[j,],"-")
  weights <- exp(-rowSums(dx^2)/(2*h*h))
  weights <- sqrt(weights/sum(weights))
  coefficients <- qr.solve(cbind(1,dx %*% b)*weights,f*weights)
  for (k in seq_len(ncol(f))) {
    index <- index+1L
    aa[[index]] <- kronecker(t(coefficients[-1,k]),dx)*weights
    yy[[index]] <- (f[,k]-coefficients[1,k])*weights
  }
}
expected_ambient <- matrix(qr.solve(do.call(rbind,aa),unlist(yy)),3,2)
ambient <- mave_step(x,b,response,ncol(f),h,block=2,ambient=TRUE)$raw
stopifnot(max(abs(expected_ambient-ambient)) < 1e-9)
stopifnot(max(abs(ambient-actual)) > 1e-4)
cat("SIR, rank failure, streaming QR, joint density LS and block invariance: PASS\n")
