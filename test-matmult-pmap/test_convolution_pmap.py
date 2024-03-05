import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Check if the correct number of arguments are passed
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <number1>")
    sys.exit(1)

input_size = int(sys.argv[1])
print(f"input size: ", input_size)

out_dir = os.getcwd()+f"/results_{input_size}"
print(f"output directory: {out_dir}")

with jax.profiler.trace(out_dir, create_perfetto_trace=True):
    x = np.arange(input_size)
    w = np.array([2., 3., 4.])

    def convolve(x, w):
      output = []
      for i in range(1, len(x)-1):
        output.append(jnp.dot(x[i-1:i+2], w))
      return jnp.array(output)

    n_devices = jax.local_device_count()
    xs = np.arange(input_size * n_devices).reshape(-1, input_size)


    y = jax.pmap(convolve, in_axes=(0, None))(xs, w)
    y.block_until_ready()
    print(y)

