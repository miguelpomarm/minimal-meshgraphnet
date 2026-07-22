# Copyright 2026 Miguel Pomar Martinez. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Preprocess the CylinderFlow dataset with the Sinkhorn-Knopp step.

For each mesh in the dataset, this script computes the angular bin
allocation (edge_bin_start, edge_bin_width) required by the proposed
architecture and writes an augmented TFRecord containing the original
fields plus the two new bin-range tensors.

The dataset is processed in chunks that are uploaded to S3 and deleted
locally on the fly, to avoid keeping the full dataset on local disk.
A per-mesh cache avoids recomputing the Sinkhorn-Knopp allocation for
trajectories that share the same mesh (as is the case in CylinderFlow).

Usage examples:

  # Process the train split into 4 chunks of 250 trajectories each
  python preprocess_dataset.py \\
      --bucket my-bucket \\
      --input-prefix dataset_base \\
      --out-prefix dataset_v2 \\
      --input-tfrecord train.tfrecord \\
      --output-name train_v2 \\
      --total-trajs 1000 \\
      --chunk-size 250

  # Process the test split into a single file (5 trajectories, no chunking)
  python preprocess_dataset.py \\
      --bucket my-bucket \\
      --input-prefix dataset_base \\
      --out-prefix dataset_v2 \\
      --input-tfrecord test.tfrecord \\
      --output-name test_v2 \\
      --total-trajs 5 \\
      --chunk-size 5

When `--chunk-size >= --total-trajs`, a single file `<output-name>.tfrecord`
is produced. Otherwise, chunks are named `<output-name>_chunk_<i>.tfrecord`.

Bachelor's Thesis - Universidad de Zaragoza (EINA) - 2026
Reference: T. Pfaff et al., "Learning Mesh-Based Simulation with Graph
Networks", ICLR 2021. arXiv:2010.03409
"""

import os
import json
import time
import struct
import hashlib
import argparse
import numpy as np
import tensorflow as tf
import boto3


if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()


# ────────── CLI arguments ──────────
parser = argparse.ArgumentParser()
parser.add_argument('--bucket', type=str, required=True,
                    help='S3 bucket name.')
parser.add_argument('--input-prefix', type=str, default='dataset_base',
                    help='S3 prefix of the input dataset (default: dataset_base).')
parser.add_argument('--out-prefix', type=str, default='dataset_v2',
                    help='S3 prefix of the output dataset (default: dataset_v2).')
parser.add_argument('--input-tfrecord', type=str, default='train.tfrecord',
                    help='Name of the input TFRecord file (default: train.tfrecord).')
parser.add_argument('--output-name', type=str, default='train_v2',
                    help='Base name of the output TFRecord files '
                         '(default: train_v2). When chunked, files are named '
                         '<output-name>_chunk_<i>.tfrecord; when a single '
                         'file is produced, it is named <output-name>.tfrecord.')
parser.add_argument('--chunk-size', type=int, default=250,
                    help='Trajectories per chunk (default: 250).')
parser.add_argument('--total-trajs', type=int, default=1000,
                    help='Total number of trajectories to process (default: 1000).')
parser.add_argument('--start-chunk', type=int, default=0,
                    help='Start from this chunk index (for resuming interrupted runs).')
parser.add_argument('--no-upload', action='store_true',
                    help='Do not upload results to S3 (keep local files).')
args = parser.parse_args()


# ────────── Constants ──────────
S3_BUCKET     = args.bucket
S3_PREFIX_IN  = args.input_prefix
S3_PREFIX_OUT = args.out_prefix
TFRECORD_IN   = args.input_tfrecord
OUTPUT_NAME   = args.output_name
META_IN       = 'meta.json'

# When the whole dataset fits in a single chunk, produce one unchunked file
SINGLE_FILE   = args.chunk_size >= args.total_trajs

LATENT_TOTAL = 128
SELF_DIMS    = 16
ANGULAR_DIMS = LATENT_TOTAL - SELF_DIMS

# Sinkhorn-Knopp hyperparameters
EPSILON_SK   = 0.5
MAX_ITER_SK  = 500
TOL_SK       = 1e-7


# ════════════════════════════════════════════════════════════════════════
# 1. ANGULAR ALLOCATION ALGORITHM
# ════════════════════════════════════════════════════════════════════════

def cells_to_edges_undirected(cells):
    """Triangles -> undirected unique edges (used to build the adjacency)."""
    e1 = cells[:, [0, 1]]
    e2 = cells[:, [1, 2]]
    e3 = cells[:, [2, 0]]
    e = np.concatenate([e1, e2, e3], axis=0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def triangles_to_edges_directed_TF_ORDER(cells):
    """Replicate EXACTLY the TF implementation of common.triangles_to_edges.

    Preserving the exact edge order is critical: the bin_start/bin_width
    tensors produced here must be indexed in the same order as the edges
    that the model reads at training time (via tf.unique). Otherwise, the
    scatter operation in the model would send messages to wrong bins,
    without raising any error.

    In TensorFlow, edges are deduplicated by packing (sender, receiver)
    into int64 and applying tf.unique, which preserves the order of first
    appearance. Reproducing this with NumPy requires `np.unique` with
    `return_index=True` and re-sorting by first-appearance index (a raw
    np.unique would sort lexicographically and shuffle the edge order).
    """
    edges = np.concatenate([
        cells[:, 0:2],
        cells[:, 1:3],
        np.stack([cells[:, 2], cells[:, 0]], axis=1)
    ], axis=0).astype(np.int32)

    # Canonical ordering: (min, max) so that (A, B) and (B, A) collide
    receivers = np.minimum(edges[:, 0], edges[:, 1])
    senders   = np.maximum(edges[:, 0], edges[:, 1])

    # Bitcast pack into int64, exactly as tf.bitcast does
    stacked = np.stack([senders, receivers], axis=1).astype(np.int32)
    packed = stacked.view(np.int64).reshape(-1)

    # Preserve order of first appearance (matches tf.unique behaviour)
    _, idx_first_occurrence = np.unique(packed, return_index=True)
    idx_first_occurrence.sort()
    packed_unique = packed[idx_first_occurrence]

    # Unpack back into (sender, receiver) pairs
    unpacked = packed_unique.view(np.int32).reshape(-1, 2)
    senders_u   = unpacked[:, 0]
    receivers_u = unpacked[:, 1]

    # Bidirectional
    s_bidir = np.concatenate([senders_u, receivers_u], axis=0)
    r_bidir = np.concatenate([receivers_u, senders_u], axis=0)
    return s_bidir, r_bidir


def angle_deg(p_from, p_to):
    """Angle in degrees from p_from to p_to, in [0, 360)."""
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    return np.degrees(np.arctan2(dy, dx)) % 360


def angulo_a_bin(angulo_deg, num_bins=ANGULAR_DIMS):
    """Map an angle in degrees to a (fractional) bin index."""
    return (angulo_deg % 360) / 360 * num_bins


def dist_circular(a, b, total=ANGULAR_DIMS):
    """Circular distance between two bin indices."""
    d = abs(a - b) % total
    return min(d, total - d)


def largest_remainder(values, target):
    """Round real values to integers such that they sum exactly to `target`.

    Uses the largest-remainder method: floor everything, then add one unit
    to the entries with the largest fractional part until the total matches.
    """
    floors = np.floor(values).astype(int)
    rem = values - floors
    deficit = int(target - floors.sum())
    order = np.argsort(-rem)
    out = floors.copy()
    for k in range(deficit):
        out[order[k]] += 1
    return out


def sinkhorn_knopp(angles, neighbors, n_nodes,
                   self_dims=SELF_DIMS, total=LATENT_TOTAL,
                   eps=EPSILON_SK, max_iter=MAX_ITER_SK, tol=TOL_SK):
    """Compute the doubly-stochastic bin allocation matrix.

    Returns a matrix B of shape [n_nodes, n_nodes] whose row and column
    sums are (approximately) equal to `total - self_dims`, and whose
    non-zero entries encode the angular affinity between each pair of
    neighbouring nodes.
    """
    target = total - self_dims
    B = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # Initial affinity: symmetric angular separation between neighbours
    for i in range(n_nodes):
        neigh_i = neighbors[i]
        if not neigh_i:
            continue
        ang_i = np.array([angles[i][j] for j in neigh_i])
        for k, j in enumerate(neigh_i):
            others = np.delete(ang_i, k)
            if len(others) == 0:
                u_ij = 180.0
            else:
                d = np.abs(others - ang_i[k])
                d = np.minimum(d, 360 - d)
                u_ij = d.min()
            neigh_j = neighbors[j]
            ang_j = np.array([angles[j][k2] for k2 in neigh_j])
            ang_ji = angles[j][i]
            idx_i = list(neigh_j).index(i)
            others_j = np.delete(ang_j, idx_i)
            if len(others_j) == 0:
                u_ji = 180.0
            else:
                d = np.abs(others_j - ang_ji)
                d = np.minimum(d, 360 - d)
                u_ji = d.min()
            u = (u_ij + u_ji) / 2
            B[i, j] = np.exp(-(1.0 / max(u, 1e-6)) / eps)

    # Sinkhorn-Knopp iterations: alternate row and column normalisation,
    # then symmetrise. Converges to a doubly-stochastic matrix.
    for _ in range(max_iter):
        rs = B.sum(axis=1, keepdims=True); rs[rs == 0] = 1
        B = B * (target / rs)
        cs = B.sum(axis=0, keepdims=True); cs[cs == 0] = 1
        B = B * (target / cs)
        B = (B + B.T) / 2
        err = max(abs(B.sum(axis=1) - target).max(),
                  abs(B.sum(axis=0) - target).max())
        if err < tol:
            break
    return B


def encajar_anillo(centros_ideales, anchos, total=ANGULAR_DIMS):
    """Fit blocks of size `anchos` into a ring of size `total`.

    Given the ideal angular position of each neighbour and the width of
    its block, sorts them by ideal position, places them consecutively
    (guaranteeing no overlap by construction), and rotates the whole ring
    by the offset that minimises the squared angular displacement of each
    block centre with respect to its ideal position.

    Returns a list of (start, width) ranges, one per input block, in the
    original order.
    """
    n = len(anchos)
    orden = np.argsort(centros_ideales)
    centros_ord = np.array(centros_ideales)[orden]
    anchos_ord  = np.array(anchos)[orden]
    offsets = np.concatenate([[0], np.cumsum(anchos_ord)[:-1]])
    centros_anillo = offsets + anchos_ord / 2.0

    # Try every possible rotation and keep the best one
    mejor_s = 0
    mejor_coste = np.inf
    for s in range(total):
        centros_rot = (centros_anillo + s) % total
        coste = sum(dist_circular(centros_rot[k], centros_ord[k], total) ** 2
                    for k in range(n))
        if coste < mejor_coste:
            mejor_coste = coste
            mejor_s = s

    rangos = [None] * n
    for k_ord, k_orig in enumerate(orden):
        ini = int((offsets[k_ord] + mejor_s) % total)
        rangos[k_orig] = (ini, int(anchos_ord[k_ord]))
    return rangos


def calcular_rangos_aristas(mesh_pos, cells):
    """Compute (bin_start, bin_width) for every directed edge of a mesh.

    Uses the TF edge ordering to ensure the resulting arrays match the
    order in which the model reads them at training time.
    """
    senders, receivers = triangles_to_edges_directed_TF_ORDER(cells)
    N = mesh_pos.shape[0]
    n_edges = len(senders)

    # Build the adjacency and geometric angles for every edge
    neighbors = [[] for _ in range(N)]
    angles    = [dict() for _ in range(N)]
    edges_und = cells_to_edges_undirected(cells)
    for a, b in edges_und:
        neighbors[a].append(b)
        neighbors[b].append(a)
        angles[a][b] = angle_deg(mesh_pos[a], mesh_pos[b])
        angles[b][a] = angle_deg(mesh_pos[b], mesh_pos[a])

    B = sinkhorn_knopp(angles, neighbors, N)

    # For every node, compute its per-neighbour angular ranges
    rangos_por_nodo = [dict() for _ in range(N)]
    for i in range(N):
        neigh = neighbors[i]
        if not neigh:
            continue
        diagonales = [(angles[i][j] + 180) % 360 for j in neigh]
        centros = [angulo_a_bin(d) for d in diagonales]
        valores = np.array([B[i, j] for j in neigh])
        anchos  = largest_remainder(valores, ANGULAR_DIMS)
        rangos = encajar_anillo(centros, anchos)
        for k, j in enumerate(neigh):
            rangos_por_nodo[i][j] = rangos[k]

    # Assemble the per-edge arrays in the same order as senders/receivers
    bin_start = np.zeros(n_edges, dtype=np.int32)
    bin_width = np.zeros(n_edges, dtype=np.int32)
    for k in range(n_edges):
        s = int(senders[k])
        r = int(receivers[k])
        ini, anc = rangos_por_nodo[r][s]
        bin_start[k] = ini
        bin_width[k] = anc

    return bin_start, bin_width


# ════════════════════════════════════════════════════════════════════════
# 2. TFRECORD UTILITIES
# ════════════════════════════════════════════════════════════════════════

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def make_example(traj_raw, bin_start, bin_width):
    """Build a tf.Example with the original fields plus the two new arrays."""
    features = {}
    for key, raw_bytes in traj_raw.items():
        features[key] = _bytes_feature(raw_bytes)
    features['edge_bin_start'] = _bytes_feature(bin_start.tobytes())
    features['edge_bin_width'] = _bytes_feature(bin_width.tobytes())
    return tf.train.Example(features=tf.train.Features(feature=features))


def parse_traj(meta, record_bytes):
    """Parse a serialised tf.Example into raw bytes and shaped arrays."""
    feat_dict = {k: tf.io.VarLenFeature(tf.string) for k in meta['features']}
    example = tf.io.parse_single_example(record_bytes, feat_dict)
    out_raw = {}
    out_shaped = {}
    for key, info in meta['features'].items():
        raw_field = tf.sparse.to_dense(example[key])
        joined = b''.join(raw_field.numpy())
        out_raw[key] = joined
        dtype = getattr(np, info['dtype'])
        arr = np.frombuffer(joined, dtype=dtype)
        if info['type'] == 'static':
            arr = arr.reshape(info['shape'])
        else:
            shape = list(info['shape'])
            shape[1] = -1
            arr = arr.reshape(shape)
        out_shaped[key] = arr
    return out_raw, out_shaped


# ════════════════════════════════════════════════════════════════════════
# 3. STREAMING FROM S3
# ════════════════════════════════════════════════════════════════════════

def stream_tfrecords_from_s3(bucket, key, skip=0):
    """Stream TFRecord entries from S3 one by one without downloading the file.

    This avoids materialising the whole (potentially very large) dataset on
    local disk. `skip` allows resuming from an intermediate trajectory.
    """
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    stream = obj['Body']
    skipped = 0
    while True:
        length_bytes = stream.read(8)
        if len(length_bytes) == 0:
            break
        if len(length_bytes) < 8:
            raise RuntimeError(
                f'Truncated stream: {len(length_bytes)} bytes for length')
        length = struct.unpack('<Q', length_bytes)[0]
        stream.read(4)  # CRC of the length
        data = stream.read(length)
        if len(data) < length:
            raise RuntimeError(f'Truncated stream: {len(data)}/{length}')
        stream.read(4)  # CRC of the payload
        if skipped < skip:
            skipped += 1
            continue
        yield data


# ════════════════════════════════════════════════════════════════════════
# 4. CHUNK PROCESSING
# ════════════════════════════════════════════════════════════════════════

def procesar_chunk(meta, chunk_idx, start_traj, num_trajs, mesh_cache):
    """Process one chunk of trajectories and write it to a local TFRecord.

    If the whole dataset fits in a single chunk, the file is named
    `<OUTPUT_NAME>.tfrecord`; otherwise, chunks are named
    `<OUTPUT_NAME>_chunk_<i>.tfrecord`.
    """
    if SINGLE_FILE:
        chunk_filename = f'{OUTPUT_NAME}.tfrecord'
    else:
        chunk_filename = f'{OUTPUT_NAME}_chunk_{chunk_idx}.tfrecord'

    print(f'\n{"─"*70}')
    print(f'CHUNK {chunk_idx} | trajectories '
          f'{start_traj}..{start_traj+num_trajs-1}')
    print(f'  Output: {chunk_filename}')
    print(f'{"─"*70}')

    reader = stream_tfrecords_from_s3(
        S3_BUCKET, f'{S3_PREFIX_IN}/{TFRECORD_IN}', skip=start_traj)
    writer = tf.io.TFRecordWriter(chunk_filename)

    t0 = time.time()
    n_processed = 0
    for record_bytes in reader:
        t_traj = time.time()
        raw, shaped = parse_traj(meta, record_bytes)
        mesh_pos = shaped['mesh_pos'][0]
        cells    = shaped['cells'][0]

        # Cache the angular allocation by mesh hash: in CylinderFlow all
        # trajectories share the same mesh, so this saves most of the work.
        mesh_hash = hashlib.md5(mesh_pos.tobytes() + cells.tobytes()).hexdigest()
        if mesh_hash in mesh_cache:
            bin_start, bin_width = mesh_cache[mesh_hash]
        else:
            bin_start, bin_width = calcular_rangos_aristas(mesh_pos, cells)
            mesh_cache[mesh_hash] = (bin_start, bin_width)

        example = make_example(raw, bin_start, bin_width)
        writer.write(example.SerializeToString())

        n_processed += 1
        dt = time.time() - t_traj
        if n_processed % 25 == 0 or n_processed == 1:
            elapsed = time.time() - t0
            eta = elapsed / n_processed * (num_trajs - n_processed) / 60
            print(f'  [{n_processed:3d}/{num_trajs}] '
                  f'traj {dt:5.1f}s | elapsed={elapsed/60:.1f}min | '
                  f'ETA={eta:.1f}min')
        if n_processed >= num_trajs:
            break

    writer.close()
    elapsed = time.time() - t0
    sz_gb = os.path.getsize(chunk_filename) / (1024**3)
    print(f'✓ Chunk {chunk_idx}: {n_processed} trajectories in '
          f'{elapsed/60:.1f} min, {sz_gb:.2f} GB')
    return chunk_filename, sz_gb


def subir_chunk_y_borrar(chunk_path):
    """Upload a chunk to S3 and delete the local file."""
    s3_path = f's3://{S3_BUCKET}/{S3_PREFIX_OUT}/{chunk_path}'
    print(f'  Uploading to {s3_path}...')
    rc = os.system(f'aws s3 cp {chunk_path} {s3_path}')
    if rc != 0:
        raise RuntimeError(f'Upload failed (rc={rc})')
    os.remove(chunk_path)
    print(f'  ✓ Uploaded and deleted local file.')


def main():
    print('═' * 70)
    print('PREPROCESSING - Sinkhorn-Knopp angular allocation')
    print(f'  Bucket             : {S3_BUCKET}')
    print(f'  Input prefix       : {S3_PREFIX_IN}')
    print(f'  Input file         : {TFRECORD_IN}')
    print(f'  Output prefix      : {S3_PREFIX_OUT}')
    print(f'  Output base name   : {OUTPUT_NAME}')
    print(f'  Total trajectories : {args.total_trajs}')
    print(f'  Chunk size         : {args.chunk_size}')
    n_chunks = (args.total_trajs + args.chunk_size - 1) // args.chunk_size
    print(f'  Number of chunks   : {n_chunks} '
          f'({"single file" if SINGLE_FILE else "chunked"})')
    if args.no_upload:
        print(f'  --no-upload: chunks kept locally, not uploaded to S3')
    print('═' * 70)

    # Fetch the original meta.json (needed to parse the TFRecord)
    if not os.path.exists(META_IN):
        os.system(f'aws s3 cp s3://{S3_BUCKET}/{S3_PREFIX_IN}/{META_IN} .')

    with open(META_IN) as f:
        meta = json.load(f)

    mesh_cache = {}
    t_global = time.time()
    for chunk_idx in range(args.start_chunk, n_chunks):
        start_traj = chunk_idx * args.chunk_size
        num_trajs = min(args.chunk_size, args.total_trajs - start_traj)
        if num_trajs <= 0:
            break
        chunk_path, _ = procesar_chunk(
            meta, chunk_idx, start_traj, num_trajs, mesh_cache)
        if not args.no_upload:
            subir_chunk_y_borrar(chunk_path)
        else:
            print(f'  Kept local: {chunk_path}')

    # Extend meta.json with the two new fields and upload it
    meta_new = json.loads(json.dumps(meta))
    meta_new['features']['edge_bin_start'] = {
        'type': 'static', 'shape': [1, -1, 1], 'dtype': 'int32'}
    meta_new['features']['edge_bin_width'] = {
        'type': 'static', 'shape': [1, -1, 1], 'dtype': 'int32'}
    if 'field_names' in meta_new:
        meta_new['field_names'] = list(meta_new['features'].keys())
    with open('meta.json.new', 'w') as f:
        json.dump(meta_new, f, indent=2)

    if not args.no_upload:
        os.system(
            f'aws s3 cp meta.json.new s3://{S3_BUCKET}/{S3_PREFIX_OUT}/meta.json')

    print(f'\n✓ COMPLETED in {(time.time()-t_global)/60:.1f} min')


if __name__ == '__main__':
    main()
