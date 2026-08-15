def require_cupy():
    """Return CuPy when a CUDA device is usable; never fall back to NumPy."""
    try:
        import cupy as cp

        count = cp.cuda.runtime.getDeviceCount()
    except Exception as error:
        raise RuntimeError(f"GPU execution requires working CuPy/CUDA: {error}") from error
    if count < 1:
        raise RuntimeError("GPU execution requires an available CUDA device")
    return cp
