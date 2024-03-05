import os
import numpy as np
import pytest

import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map as shmap
from jax.experimental.pjit import pjit

# on gpu instead use create_hybrid_device_mesh
from jax.experimental.mesh_utils import create_device_mesh, create_hybrid_device_mesh

m = 9072
n = 2048
k = 3072

b = n
d_in = k
d_out_shard = m

type_precision = jnp.float16

mp_len = 1
dp_len = 1
dcn_len = 1
print("Jax device=",jax.devices())
hw_mesh = create_device_mesh([dcn_len, dp_len, mp_len], jax.devices())

mesh = Mesh(hw_mesh, ["dcn", "dp", "mp"])

x_mp_pspec = PartitionSpec(("dcn", "dp"), "mp")
w_mp_expanding_pspec = PartitionSpec("mp", None)

def init_tensors_x_y(key):
        kx, kw, kdy, kdw = jax.random.split(key, 4)
        x = jax.random.normal(kx, (b, d_in), dtype=type_precision)
        dy = jax.random.normal(kdy, (b, d_out_shard), dtype=type_precision)
        return x, dy

def init_tensors_w(key):
    kx, kw, kdy, kdw = jax.random.split(key, 4)
    w = jax.random.normal(kw, (d_out_shard, d_in), dtype=type_precision)
    return w

def fwdbwd_pjit(x, w):
    y = jnp.einsum("bi,oi->bo",x, w)
    return y

k = jax.random.PRNGKey(0)

(x_mp, dy_mp,) = jax.jit(
    init_tensors_x_y,
    in_shardings=(NamedSharding(mesh, PartitionSpec()),),
    out_shardings=(
        NamedSharding(mesh, x_mp_pspec),
        NamedSharding(mesh, x_mp_pspec),
    ),
)(k)


w_mp_expanding = jax.jit(
    init_tensors_w,
    in_shardings=(NamedSharding(mesh, PartitionSpec()),),
    out_shardings=NamedSharding(mesh, w_mp_expanding_pspec),
)(k)

out_dir=os.getcwd()+"/results"
print("output dir: "+out_dir)

with jax.profiler.trace(out_dir, create_perfetto_trace=True):
    y = jnp.einsum("bi,oi->bo",x_mp, w_mp_expanding)
    y.block_until_ready()
