import numpy as np

# --- Optional PyTorch Support ---
try:
    import torch
    import torch.fft as torch_fft
    torch_available = True
    # Auto-detect the best available PyTorch device
    if torch.cuda.is_available():
        torch_default_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        torch_default_device = torch.device('mps')
    else:
        torch_default_device = torch.device('cpu')
except ImportError:
    torch_available = False
    torch = None
    torch_fft = None
    torch_default_device = None

# --- Optional JAX Support ---
try:
    import jax
    import jax.numpy as jnp
    jax_available = True
    # JAX automatically defaults to the best available accelerator (GPU/TPU)
    jax_default_device = jax.devices()[0] if jax.devices() else None
except ImportError:
    jax_available = False
    jax = None
    jnp = None
    jax_default_device = None

def fftconvolve(in1, in2, backend='numpy', device=None, return_numpy=True):
    """
    Convolve two N-dimensional arrays using FFT on a specified backend.

    Parameters
    ----------
    in1, in2 : array_like
        Input arrays. Must have the same rank.
    backend : str, optional
        The compute backend to use: 'numpy', 'torch', or 'jax'. Default is 'numpy'.
    device : str, torch.device, or jax.Device, optional
        The target device. If None, it uses the auto-detected default device 
        for the chosen backend (e.g., 'cuda' or 'mps' for Torch).

    Returns
    -------
    out : array_like
        The convolved array, returned in the format of the requested backend.
    """
    backend = backend.lower()

    # --- 1. PyTorch Backend ---
    if backend == 'torch':
        if not torch_available:
            raise ImportError("PyTorch backend requested, but 'torch' is not installed.")
        
        target_device = device if device is not None else torch_default_device
        
        # 1. Load onto CPU first to avoid MPS float64 crash
        in1_t = torch.as_tensor(in1)
        in2_t = torch.as_tensor(in2)
        
        # 2. Safely downcast if the target is Apple Silicon (MPS)
        if str(target_device).startswith('mps'):
            if in1_t.dtype == torch.float64:
                in1_t = in1_t.to(torch.float32)
            elif in1_t.dtype == torch.complex128:
                in1_t = in1_t.to(torch.complex64)
                
            if in2_t.dtype == torch.float64:
                in2_t = in2_t.to(torch.float32)
            elif in2_t.dtype == torch.complex128:
                in2_t = in2_t.to(torch.complex64)
        
        # 3. Now safely move to the target device (GPU/MPS/CPU)
        in1_t = in1_t.to(target_device)
        in2_t = in2_t.to(target_device)
        
        if in1_t.ndim != in2_t.ndim:
            raise ValueError("in1 and in2 should have the same rank")
        if 0 in in1_t.shape or 0 in in2_t.shape:
            return torch.tensor([], device=target_device)

        fsize = tuple(in1_t.shape)
        is_complex = in1_t.is_complex() or in2_t.is_complex()

        if not is_complex:
            t1 = torch_fft.rfftn(in1_t, s=fsize)
            t2 = torch_fft.rfftn(in2_t, s=fsize)
            ret = torch_fft.irfftn(t1 * t2, s=fsize)
        else:
            t1 = torch_fft.fftn(in1_t, s=fsize)
            t2 = torch_fft.fftn(in2_t, s=fsize)
            ret = torch_fft.ifftn(t1 * t2, s=fsize)

        shifts = tuple((-np.floor(np.array(fsize) * 0.5)).astype(int).tolist())
        roll_axes = tuple(range(in1_t.ndim))
        out = torch.roll(ret, shifts=shifts, dims=roll_axes)
        return out.cpu().numpy() if return_numpy else out 

    # --- 2. JAX Backend ---
    elif backend == 'jax':
        if not jax_available:
            raise ImportError("JAX backend requested, but 'jax' is not installed.")
        
        in1_j = jnp.asarray(in1)
        in2_j = jnp.asarray(in2)
        
        if device is not None:
            in1_j = jax.device_put(in1_j, device)
            in2_j = jax.device_put(in2_j, device)

        if in1_j.ndim != in2_j.ndim:
            raise ValueError("in1 and in2 should have the same rank")
        if 0 in in1_j.shape or 0 in in2_j.shape:
            return jnp.array([])

        fsize = tuple(in1_j.shape)
        is_complex = jnp.iscomplexobj(in1_j) or jnp.iscomplexobj(in2_j)

        if not is_complex:
            r1 = jnp.fft.rfftn(in1_j, s=fsize)
            r2 = jnp.fft.rfftn(in2_j, s=fsize)
            ret = jnp.fft.irfftn(r1 * r2, s=fsize)
        else:
            f1 = jnp.fft.fftn(in1_j, s=fsize)
            f2 = jnp.fft.fftn(in2_j, s=fsize)
            ret = jnp.fft.ifftn(f1 * f2, s=fsize)

        shifts = tuple((-np.floor(np.array(fsize) * 0.5)).astype(int).tolist())
        roll_axes = tuple(range(in1_j.ndim))
        out = jnp.roll(ret, shift=shifts, axis=roll_axes)
        return np.asarray(out) if return_numpy else out 

    # --- 3. Standard NumPy Backend (Default) ---
    elif backend == 'numpy':
        in1_n = np.asarray(in1)
        in2_n = np.asarray(in2)
        
        if in1_n.ndim != in2_n.ndim:
            raise ValueError("in1 and in2 should have the same rank")
        if in1_n.size == 0 or in2_n.size == 0:
            return np.array([])

        fsize = tuple(in1_n.shape)
        is_complex = np.iscomplexobj(in1_n) or np.iscomplexobj(in2_n)

        if not is_complex:
            r1 = np.fft.rfftn(in1_n, s=fsize)
            r2 = np.fft.rfftn(in2_n, s=fsize)
            ret = np.fft.irfftn(r1 * r2, s=fsize)
        else:
            f1 = np.fft.fftn(in1_n, s=fsize)
            f2 = np.fft.fftn(in2_n, s=fsize)
            ret = np.fft.ifftn(f1 * f2, s=fsize)

        shifts = tuple((-np.floor(np.array(fsize) * 0.5)).astype(int).tolist())
        roll_axes = tuple(range(in1_n.ndim))
        return np.roll(ret, shift=shifts, axis=roll_axes)

    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'numpy', 'torch', or 'jax'.")
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # from skimage.morphology import ball
    from time import time

    ncells = 200
    in1 = np.zeros((ncells,ncells,ncells))
    in1[100,100,10] = 1
    in1[100,10,100] = 1
    xx = np.arange(ncells)
    mu = [ncells/2, ncells/2, ncells/2]
    xmesh = np.meshgrid(xx, xx, xx)
    sigma = 40
    in2 = -0.5*np.sum([(xmesh[i]-mu[i])**2/sigma**2 for i in range(3)], axis=0)
    in2 = np.exp(in2)
    # in2 = np.zeros((ncells,ncells,ncells))
    # in2[100-sigma:100+sigma+1,100-sigma:100+sigma+1,100-sigma:100+sigma+1] = ball(sigma)
    out = {}
    for backend in ['numpy', 'torch', 'jax']:
        print(f"backend = {backend} ", end="")
        t0 = time()
        out[backend] = fftconvolve(in1, in2, backend=backend, device=None)
        print(f": {time()-t0:.3f} s")

    fig, axs = plt.subplots(2,2,figsize=(10,9))
    axs = axs.flatten()
    for ii,arr in enumerate([in1,in2,out['numpy'],out['jax']]):
        print(f"array shape: {arr.shape}, type: {type(arr)}")
        im = axs[ii].pcolor(
                xx, xx,
                arr[100,...],
                cmap='jet'
            )
        axs[ii].set_xlabel('X', fontsize=14)
        axs[ii].set_ylabel('Y', fontsize=14)
        fig.colorbar(im, ax=axs[ii])
    plt.tight_layout()
    plt.show()