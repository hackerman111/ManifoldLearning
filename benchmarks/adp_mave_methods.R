#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(MAVE))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9L) {
  stop("usage: adp_mave_methods.R METHOD X.csv Y.csv BASIS.csv META.tsv SEED DIM MAX_ITER N_Y_ANCHORS")
}
method <- args[[1L]]
x_path <- args[[2L]]
y_path <- args[[3L]]
basis_path <- args[[4L]]
meta_path <- args[[5L]]
seed <- as.integer(args[[6L]])
dim <- as.integer(args[[7L]])
max_iter <- as.integer(args[[8L]])
density_anchors <- as.integer(args[[9L]])

x <- as.matrix(read.csv(x_path, header = FALSE, check.names = FALSE))
y <- as.numeric(read.csv(y_path, header = FALSE, check.names = FALSE)[[1L]])
n <- nrow(x)
d <- ncol(x)
m <- dim
if (length(y) != n || n <= d || d < m) stop("invalid input dimensions")

proc_hwm_kib <- function() {
  status <- "/proc/self/status"
  if (!file.exists(status)) return(NA_real_)
  line <- grep("^VmHWM:", readLines(status), value = TRUE)
  if (length(line) == 0L) return(NA_real_)
  as.numeric(strsplit(trimws(line[[1L]]), "[[:space:]]+")[[1L]][[2L]])
}

extract_basis <- function(fit, dim, p) {
  if (is.null(fit$dir) || length(fit$dir) < dim) stop("MAVE returned no requested direction")
  b <- as.matrix(fit$dir[[dim]])
  if (nrow(b) != p && ncol(b) == p) b <- t(b)
  if (!all(dim(b) == c(p, dim)) || any(!is.finite(b))) {
    stop(sprintf(
      "MAVE basis invalid: got %s, expected %s, finite=%s",
      paste(base::dim(b), collapse = "x"),
      paste(c(p, dim), collapse = "x"),
      all(is.finite(b))
    ))
  }
  if (qr(b)$rank < dim) stop("MAVE returned a rank-deficient basis")
  qr.Q(qr(b))[, seq_len(dim), drop = FALSE]
}

weighted_mave_refinement <- function(x, y, dim, max_iter) {
  # rMAVE-style refinement: OPG initialization followed by alternating
  # projected-kernel local fits and a shared ambient-space weighted LS update.
  init <- MAVE::mave.compute(
    x, y, method = "CSOPG", max.dim = dim, screen = ncol(x)
  )
  b <- extract_basis(init, dim, ncol(x))
  n <- nrow(x)
  d <- ncol(x)
  p <- max(d, 3L)
  c0 <- 2.34
  h <- c0 * n^(-1 / (p + 6))
  h_min <- c0 * n^(-1 / (dim + 4))
  shrink <- n^(-1 / (2 * (p + 6)))
  projector <- function(q) q %*% t(q)
  iterations <- 0L
  final_step <- NA_real_

  for (iteration in seq_len(max_iter)) {
    z <- x %*% b
    z2 <- rowSums(z * z)
    dist2 <- outer(z2, z2, "+") - 2 * tcrossprod(z)
    dist2 <- pmax(dist2, 0)
    weights <- exp(-0.5 * dist2 / (h * h))
    weights <- weights / rowSums(weights)

    intercept <- numeric(n)
    gradient <- matrix(0, n, dim)
    for (i in seq_len(n)) {
      dx <- sweep(x, 2L, x[i, ], "-")
      design <- cbind(1, dx %*% b)
      local <- stats::lm.wfit(design, y, weights[i, ])$coefficients
      if (length(local) != dim + 1L || any(!is.finite(local))) {
        stop(sprintf("local weighted fit failed at anchor %d", i))
      }
      intercept[i] <- local[[1L]]
      gradient[i, ] <- local[-1L]
    }

    qdim <- d * dim
    hessian <- matrix(0, qdim, qdim)
    rhs <- numeric(qdim)
    for (i in seq_len(n)) {
      dx <- sweep(x, 2L, x[i, ], "-")
      wi <- weights[i, ]
      xtwx <- crossprod(dx, dx * wi)
      c_i <- gradient[i, ]
      ci_cross <- tcrossprod(c_i)
      hessian <- hessian + kronecker(ci_cross, xtwx)
      q_i <- drop(crossprod(dx, wi * (y - intercept[i])))
      rhs <- rhs + as.vector(outer(q_i, c_i))
    }
    ridge <- max(max(diag(hessian)), 1) * 1e-8
    b_raw <- matrix(solve(hessian + diag(ridge, qdim), rhs), nrow = d, ncol = dim)
    b_new <- qr.Q(qr(b_raw))[, seq_len(dim), drop = FALSE]
    final_step <- sqrt(sum((projector(b_new) - projector(b))^2))
    b <- b_new
    iterations <- iteration
    if (final_step < 1e-4 || h <= h_min * (1 + 1e-12)) break
    h <- max(shrink * h, h_min)
  }
  list(basis = b, iterations = iterations, final_step = final_step)
}

set.seed(seed)
rss_before <- proc_hwm_kib()
started <- proc.time()[[3L]]
iterations <- NA_integer_
detail <- ""

if (method == "MAVE") {
  fit <- MAVE::mave.compute(x, y, method = "CSMAVE", max.dim = m, screen = d)
  basis <- extract_basis(fit, m, d)
} else if (method == "dMAVE") {
  # dMAVE uses K_h(Y - t) as response. Quantile anchors discretize the
  # response integral; average fitted subspace projectors across anchors.
  h_y <- 1.06 * stats::sd(y) * n^(-1 / 5)
  if (!is.finite(h_y) || h_y <= 0) stop("response bandwidth is not positive")
  anchors <- as.numeric(stats::quantile(
    y, probs = seq(0.1, 0.9, length.out = density_anchors), names = FALSE
  ))
  projector_sum <- matrix(0, d, d)
  for (k in seq_along(anchors)) {
    set.seed(seed + k - 1L)
    response <- exp(-0.5 * ((y - anchors[[k]]) / h_y)^2)
    fit <- MAVE::mave.compute(x, response, method = "CSMAVE", max.dim = m, screen = d)
    b_k <- extract_basis(fit, m, d)
    projector_sum <- projector_sum + tcrossprod(b_k)
  }
  eig <- eigen(projector_sum / length(anchors), symmetric = TRUE)
  basis <- eig$vectors[, seq_len(m), drop = FALSE]
  detail <- sprintf(
    "density_response_gaussian_h=%.17g;quantile_anchors=%d",
    h_y, length(anchors)
  )
} else if (method == "rMAVE") {
  fit <- weighted_mave_refinement(x, y, m, max_iter)
  basis <- fit$basis
  iterations <- fit$iterations
  detail <- sprintf("projected_kernel_refinement_final_step=%.17g", fit$final_step)
} else {
  stop(sprintf("unknown method: %s", method))
}

elapsed <- proc.time()[[3L]] - started
rss_after <- proc_hwm_kib()
write.table(basis, basis_path, sep = ",", row.names = FALSE, col.names = FALSE, quote = FALSE)
metadata <- data.frame(
  method = method,
  seconds = elapsed,
  peak_rss_kib = if (is.na(rss_after)) rss_before else rss_after,
  iterations = iterations,
  detail = detail,
  stringsAsFactors = FALSE
)
write.table(metadata, meta_path, sep = "\t", row.names = FALSE, col.names = TRUE, quote = FALSE)
