import os
import sys
import jax
import jax.numpy as jnp

# Check if the correct number of arguments are passed
if len(sys.argv) != 3:
    print("Usage: test_matmult.py <number1> <number2>")
    sys.exit(1)

# Try to convert arguments to integers
try:
    rows = int(sys.argv[1])
    cols = int(sys.argv[2])
except ValueError:
    print("Both arguments must be integers.")
    sys.exit(1)

out_dir=os.getcwd()+ f"/results/matmult_{rows}x{cols}"

with jax.profiler.trace(out_dir, create_perfetto_trace=True):
    # Run the operations to be profiled
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (rows, cols))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print("program ends, y=", y)

