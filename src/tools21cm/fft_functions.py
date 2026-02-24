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

def fftconvolve(in1, in2, mode='same', boundary='circular', backend='numpy', device=None):
    """
    Convolve two N-dimensional arrays using FFT.

    Parameters
    ----------
    in1, in2 : array_like
        Input arrays.
    mode : {'same', 'full', 'valid'}, optional
        Determines the output size based on array overlap.
    boundary : {'constant', 'circular', 'reflect', 'nearest', 'mirror', 'wrap'}, optional
        Determines how the boundaries of in1 are handled. 
        Note: Ignored if mode='valid'. If 'circular', mode is forced to 'same'.
    backend : {'numpy', 'torch', 'jax'}, optional
    device : optional
    """
    backend = backend.lower()
    if mode not in ['same', 'full', 'valid']: raise ValueError("mode must be 'same', 'full', or 'valid'")
    if boundary not in ['constant', 'circular', 'reflect', 'nearest', 'mirror', 'wrap']:
        raise ValueError("Invalid boundary condition.")

    s1, s2 = np.array(in1.shape), np.array(in2.shape)
    
    # Define the axes explicitly for NumPy 2.0 compliance
    calc_axes = tuple(range(len(s1)))

    # --- 1. Logic Overrides & Padding Math ---
    if mode == 'valid':
        if np.any(s1 < s2): raise ValueError("For 'valid' mode, in1 must be >= in2 in all dimensions.")
        boundary = 'constant'  
        
    if boundary == 'circular':
        mode = 'same'
        fsize = tuple(np.maximum(s1, s2).tolist())
        do_pad = False
    elif boundary == 'constant':
        fsize = tuple((s1 + s2 - 1).tolist())
        do_pad = False
    else: 
        do_pad = True
        pad_before = s2 - 1 if mode == 'full' else s2 // 2
        pad_after = s2 - 1 if mode == 'full' else s2 - 1 - (s2 // 2)
        fsize = tuple((s1 + pad_before + pad_after + s2 - 1).tolist())

    # --- 2. PyTorch Backend ---
    if backend == 'torch':
        if not torch_available: raise ImportError("PyTorch not installed.")
        target_device = device if device is not None else torch_default_device
        in1_t, in2_t = torch.as_tensor(in1), torch.as_tensor(in2)
        
        if str(target_device).startswith('mps'): 
            if in1_t.dtype == torch.float64: in1_t = in1_t.to(torch.float32)
            elif in1_t.dtype == torch.complex128: in1_t = in1_t.to(torch.complex64)
            if in2_t.dtype == torch.float64: in2_t = in2_t.to(torch.float32)
            elif in2_t.dtype == torch.complex128: in2_t = in2_t.to(torch.complex64)

        if do_pad:
            t_mode = {'reflect':'reflect', 'mirror':'reflect', 'nearest':'replicate', 'wrap':'circular'}[boundary]
            pad_torch = [int(p) for b, a in zip(reversed(pad_before), reversed(pad_after)) for p in (b, a)]
            import torch.nn.functional as F
            in1_t = F.pad(in1_t.unsqueeze(0).unsqueeze(0), pad_torch, mode=t_mode).squeeze(0).squeeze(0)

        in1_t, in2_t = in1_t.to(target_device), in2_t.to(target_device)
        is_complex = in1_t.is_complex() or in2_t.is_complex()

        # Fixed: Explicit dim passed to all FFTs
        if not is_complex:
            r1 = torch_fft.rfftn(in1_t, s=fsize, dim=calc_axes)
            r2 = torch_fft.rfftn(in2_t, s=fsize, dim=calc_axes)
            ret = torch_fft.irfftn(r1 * r2, s=fsize, dim=calc_axes)
        else:
            f1 = torch_fft.fftn(in1_t, s=fsize, dim=calc_axes)
            f2 = torch_fft.fftn(in2_t, s=fsize, dim=calc_axes)
            ret = torch_fft.ifftn(f1 * f2, s=fsize, dim=calc_axes)

        if boundary == 'circular':
            return torch.roll(ret, shifts=tuple((-s2 // 2).tolist()), dims=calc_axes)

    # --- 3. JAX Backend ---
    elif backend == 'jax':
        if not jax_available: raise ImportError("JAX not installed.")
        in1_j = jax.device_put(jnp.asarray(in1), device) if device else jnp.asarray(in1)
        in2_j = jax.device_put(jnp.asarray(in2), device) if device else jnp.asarray(in2)

        if do_pad:
            j_mode = {'reflect':'reflect', 'mirror':'symmetric', 'nearest':'edge', 'wrap':'wrap'}[boundary]
            in1_j = jnp.pad(in1_j, list(zip(pad_before.tolist(), pad_after.tolist())), mode=j_mode)

        is_complex = jnp.iscomplexobj(in1_j) or jnp.iscomplexobj(in2_j)
        
        # Fixed: Explicit axes passed to all FFTs
        if not is_complex:
            r1 = jnp.fft.rfftn(in1_j, s=fsize, axes=calc_axes)
            r2 = jnp.fft.rfftn(in2_j, s=fsize, axes=calc_axes)
            ret = jnp.fft.irfftn(r1 * r2, s=fsize, axes=calc_axes)
        else:
            f1 = jnp.fft.fftn(in1_j, s=fsize, axes=calc_axes)
            f2 = jnp.fft.fftn(in2_j, s=fsize, axes=calc_axes)
            ret = jnp.fft.ifftn(f1 * f2, s=fsize, axes=calc_axes)

        if boundary == 'circular':
            return jnp.roll(ret, shift=tuple((-s2 // 2).tolist()), axis=calc_axes)

    # --- 4. NumPy Backend ---
    elif backend == 'numpy':
        in1_n, in2_n = np.asarray(in1), np.asarray(in2)
        if do_pad:
            n_mode = {'reflect':'reflect', 'mirror':'symmetric', 'nearest':'edge', 'wrap':'wrap'}[boundary]
            in1_n = np.pad(in1_n, list(zip(pad_before.tolist(), pad_after.tolist())), mode=n_mode)

        is_complex = np.iscomplexobj(in1_n) or np.iscomplexobj(in2_n)
        
        # Fixed: Explicit axes passed to all FFTs to squash NumPy 2.0 warning
        if not is_complex:
            r1 = np.fft.rfftn(in1_n, s=fsize, axes=calc_axes)
            r2 = np.fft.rfftn(in2_n, s=fsize, axes=calc_axes)
            ret = np.fft.irfftn(r1 * r2, s=fsize, axes=calc_axes)
        else:
            f1 = np.fft.fftn(in1_n, s=fsize, axes=calc_axes)
            f2 = np.fft.fftn(in2_n, s=fsize, axes=calc_axes)
            ret = np.fft.ifftn(f1 * f2, s=fsize, axes=calc_axes)

        if boundary == 'circular':
            return np.roll(ret, shift=tuple((-s2 // 2).tolist()), axis=calc_axes)

    # --- 5. Cropping Logic (Applies to all backends) ---
    if do_pad:
        start = s2 - 1
    else:
        if mode == 'full': start = np.zeros_like(s1)
        elif mode == 'same': start = s2 // 2
        elif mode == 'valid': start = s2 - 1

    out_size = s1 + s2 - 1 if mode == 'full' else (s1 if mode == 'same' else s1 - s2 + 1)
    
    slices = tuple(slice(st, st + sz) for st, sz in zip(start, out_size))
    return ret[slices]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time
    import numpy as np
    import torch # Ensure torch is imported for the .cpu() call later

    # --- 1. Define the Larger Image (in1) ---
    ncells = 200
    in1 = np.zeros((ncells, ncells, ncells))
    # Place impulses near the edges to clearly see the 'wrap' boundary effect
    in1[100, 100, 10] = 1  
    in1[100, 10, 100] = 1  

    # --- 2. Define the Smaller Kernel (in2) ---
    ksize = 261  # Making the kernel significantly smaller
    sigma = 50
    xx_k = np.arange(ksize)
    mu_k = ksize // 2
    
    # Use indexing='ij' to ensure perfectly symmetric 3D distance calculations
    xmesh_k = np.meshgrid(xx_k, xx_k, xx_k, indexing='ij')
    in2 = np.exp(-0.5 * np.sum([(xmesh_k[i] - mu_k)**2 / sigma**2 for i in range(3)], axis=0))
    in2 /= in2.sum()  # Normalize the kernel to sum to 1 for better visualization of convolution effects

    # --- 3. Run the Convolutions ---
    out = {}
    for backend in ['numpy', 'torch', 'jax']:
        print(f"backend = {backend:<6} ", end="")
        t0 = time()
        # Using mode='same' (default) so the output matches in1's shape of 200^3
        out[backend] = fftconvolve(in1, in2, mode='same', boundary='circular', backend=backend, device=None)
        print(f": {time()-t0:.3f} s")

    # --- 4. Plotting ---
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))

    # Plot the inputs (Dynamic to handle different shapes)
    for ii, arr in enumerate([in1, in2]):
        print(f"Input {ii+1} shape: {arr.shape}, type: {type(arr)}")
        
        # Dynamically find the center and axis arrays
        mid_idx = arr.shape[0] // 2
        ax_len = arr.shape[1]
        x_ax = np.arange(ax_len)
        
        im = axs[0, ii].pcolor(
                x_ax, x_ax,
                arr[mid_idx, ...],
                cmap='jet'
            )
        axs[0, ii].set_title(f"Input {ii+1} ({arr.shape[0]}$^3$)", fontsize=14)
        axs[0, ii].set_xlabel('X', fontsize=12)
        axs[0, ii].set_ylabel('Y', fontsize=12)
        fig.colorbar(im, ax=axs[0, ii])
        
    axs[0, 2].axis('off') # Hide the empty top-right subplot

    # Plot the outputs
    for ii, name in enumerate(['numpy', 'jax', 'torch']):
        # Safely convert to numpy CPU array
        arr = out[name] if name != 'torch' else out[name].cpu().numpy()
        print(f"Output {name} shape: {arr.shape}, type: {type(arr)}")
        
        mid_idx = arr.shape[0] // 2
        x_ax = np.arange(arr.shape[1])
        
        axs[1, ii].set_title(f"backend = {name}", fontsize=14)
        im = axs[1, ii].pcolor(
                x_ax, x_ax,
                arr[mid_idx, ...],
                cmap='jet'
            )
        axs[1, ii].set_xlabel('X', fontsize=12)
        axs[1, ii].set_ylabel('Y', fontsize=12)
        fig.colorbar(im, ax=axs[1, ii])
        
    plt.tight_layout()
    plt.show()
