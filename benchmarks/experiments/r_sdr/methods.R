# Benchmark-only references; formulas, variants and boundaries: README.md.
ortho <- function(b, m) {
  if (!all(is.finite(b)) || ncol(b) != m) stop("invalid basis")
  s <- svd(b, nu=m, nv=0)
  if (length(s$d) < m || min(s$d) <= max(dim(b)) * .Machine$double.eps * s$d[1]) {
    stop("rank-deficient basis")
  }
  s$u[,seq_len(m),drop=FALSE]
}

sir_fit <- function(x, y, m, slices) {
  n <- nrow(x); p <- ncol(x)
  if (n <= p) stop("classical SIR requires n > d and nonsingular covariance")
  if (slices <= m || slices > n) stop("SIR requires m < slices <= n")
  xc <- scale(x, scale=FALSE)
  s <- svd(xc / sqrt(n), nu=p, nv=p)
  if (min(s$d) <= max(dim(x)) * .Machine$double.eps * s$d[1]) {
    stop("SIR covariance is rank deficient; no ridge or PCA substitution")
  }
  z <- s$u * sqrt(n)
  # Equal-count slices; ties remain together, even when counts then differ.
  cuts <- unique(as.numeric(quantile(y, seq(0, 1, length.out=slices+1), type=1)))
  if (length(cuts) <= m+1) stop("too few distinct response values for SIR")
  groups <- split(seq_len(n), cut(y, cuts, include.lowest=TRUE))
  groups <- groups[lengths(groups) > 0]
  if (length(groups) <= m) stop("too few nonempty SIR slices")
  means <- sapply(groups, function(ids) colMeans(z[ids,,drop=FALSE]))
  means <- sweep(means, 2, sqrt(lengths(groups) / n), "*")
  eig <- svd(means, nu=m, nv=0)
  basis <- s$v %*% sweep(eig$u[,seq_len(m),drop=FALSE], 1, s$d, "/")
  list(basis=ortho(basis,m), iterations=1L, stop_reason="closed_form",
       converged=NA, detail=paste("nonempty_slices",length(groups),sep="="))
}

qr_merge <- function(state, a, y) {
  if (!is.null(state)) {
    a <- rbind(state$a, a)
    y <- c(state$y, y)
  }
  factor <- qr(a, tol=1e-12, LAPACK=FALSE)
  rows <- seq_len(min(nrow(a),ncol(a)))
  # Store R in original variable order; subsequent QR may repivot it.
  list(a=qr.R(factor)[rows,order(factor$pivot),drop=FALSE],
       y=qr.qty(factor,y)[rows])
}

mave_step <- function(x, b, response, targets, h, block=16L, ambient=FALSE) {
  n <- nrow(x); p <- ncol(x); m <- ncol(b)
  state <- NULL
  for (j in seq_len(n)) {
    dx <- sweep(x, 2, x[j,], "-")
    projected <- dx %*% b
    localization <- if (ambient) dx else projected
    weights <- exp(-rowSums(localization^2)/(2*h*h))
    weights <- sqrt(weights / sum(weights))
    factor <- qr(cbind(1,projected) * weights, tol=1e-12)
    if (factor$rank != m+1) stop("rank-deficient local weighted regression")
    coefficients <- matrix(0, m+1, targets)
    for (start in seq.int(1L,targets,by=block)) {
      ids <- start:min(start+block-1L,targets)
      coefficients[,ids] <- qr.coef(factor, response(ids) * weights)
    }
    slopes <- coefficients[-1,,drop=FALSE]
    # Exact response-space compression: the orthogonal residual is B-independent.
    # C=U D V'; ||F-dx B C||_F^2 depends on B only through F V and C V.
    s <- svd(slopes, nu=0, nv=min(m,targets))
    v <- s$v
    rotated_response <- matrix(0,n,ncol(v))
    for (start in seq.int(1L,targets,by=block)) {
      ids <- start:min(start+block-1L,targets)
      rotated_response <- rotated_response + response(ids) %*% v[ids,,drop=FALSE]
    }
    rotated_response <- sweep(rotated_response,2,drop(coefficients[1,,drop=FALSE] %*% v),"-")
    c <- slopes %*% v
    for (k in seq_len(ncol(v))) {
      a <- kronecker(t(c[,k]), dx) * weights
      state <- qr_merge(state,a,rotated_response[,k]*weights)
    }
  }
  if (is.null(state)) stop("empty global regression")
  raw <- matrix(qr.solve(state$a,state$y,tol=1e-12),p,m)
  list(raw=raw, basis=ortho(raw,m))
}

refined_fit <- function(x, y, m, density, iterations, tolerance, block=16L, ambient=FALSE) {
  n <- nrow(x); p <- ncol(x)
  if (n <= p+1) stop("OPG initialization requires n > d+1")
  scales <- apply(x,2,sd)
  if (any(!is.finite(scales)) || any(scales <= 0)) stop("constant predictor")
  xs <- scale(x)
  ys <- as.numeric(scale(y))
  if (!all(is.finite(ys))) stop("constant response")
  init_method <- if (density) "CSOPG" else "MEANOPG"
  init <- MAVE::mave.compute(xs,ys,method=init_method,max.dim=m,screen=p)
  b <- ortho(coef(init,dim=m),m)
  response_h <- 1.06 * n^(-1/5)
  response <- if (density) {
    function(ids) outer(ys,ys[ids],function(u,v) dnorm((u-v)/response_h)/response_h)
  } else {
    function(ids) matrix(ys,n,1)
  }
  targets <- if (density) n else 1L
  h <- 2.34 * n^(-1/(max(p,3)+6))
  h_min <- 2.34 * n^(-1/(m+4))
  shrink <- n^(-1/(2*(max(p,3)+6)))
  trace <- data.frame()
  converged <- FALSE
  for (it in seq_len(iterations)) {
    next_b <- mave_step(xs,b,response,targets,h,block,ambient=ambient)$basis
    delta <- sqrt(sum((next_b - b %*% crossprod(b,next_b))^2)/m)
    trace <- rbind(trace,data.frame(iteration=it,h=h,subspace_step=delta))
    b <- next_b
    if ((ambient || h <= h_min*(1+1e-12)) && delta <= tolerance) {
      converged <- TRUE
      break
    }
    if (!ambient) h <- max(shrink*h,h_min)
  }
  list(basis=ortho(sweep(b,1,scales,"/"),m), iterations=it,
       converged=converged, stop_reason=if(converged) "subspace_tol" else "max_iterations",
       detail=sprintf("init=%s;targets=%d;response_h=%.17g;ambient=%s;trim=none;ridge=0",init_method,targets,response_h,ambient),
       trace=trace)
}
