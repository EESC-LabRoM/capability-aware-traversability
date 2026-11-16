import sys
import json
import random
import hashlib
import argparse
from pathlib import Path

import numpy as np

from cat.configuration import load_config, resolve_config
from cat.embedding_generation.embedding_generator import sha1_of_paths

def _expected_shape(name, fingerprint, expected_arrays) -> tuple:
    """
    Compute a cache array's expected shape: (N, D) for vector arrays, (N, D, patch_h, patch_w) for patch grids.

    :param name: str - array file name, a key of expected_arrays
    :param fingerprint: dict - fingerprint.json contents
    :param expected_arrays: dict - array name to expected spec, None for vector arrays
    :return: tuple - expected shape
    """
    N = fingerprint['num_frames']
    D = fingerprint['embed_dim']
    if expected_arrays[name] is None:
        return (N, D)
    ph, pw = fingerprint['patch_grid']
    return (N, D, ph, pw)

def _npy_header_info(path) -> tuple:
    """
    Parse a .npy header without mapping the data payload.

    :param path: str | Path - .npy file
    :return: tuple - (shape, dtype, header_bytes), header_bytes = offset where the payload begins
    """
    # header only, payload may be truncated
    with open(path, 'rb') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _, dtype = np.lib.format.read_array_header_2_0(f)
        header_bytes = f.tell()
    return shape, dtype, header_bytes

def _sampled_rows(num_frames, verification) -> list[int]:
    """
    Pick a seeded random row sample that always includes the first and last rows.

    :param num_frames: int - total rows
    :param verification: dict - embedding.verification config section, uses 'sample_seed' and 'sampled_rows'
    :return: list[int] - sorted row indices
    """
    rng = random.Random(verification['sample_seed'])
    count = min(verification['sampled_rows'], num_frames)
    rows = set(rng.sample(range(num_frames), count))
    rows.add(0)
    rows.add(num_frames - 1)
    return sorted(rows)

def _check_sampled_rows(arrays, num_frames, ok, verification) -> bool:
    """
    Check sampled rows for non-finite values and all-zero RGB embeddings (unwritten rows).

    :param arrays: dict - {array_name: memmap} to check
    :param num_frames: int - rows in each array
    :param ok: bool - pass/fail state so far
    :param verification: dict - embedding.verification config section
    :return: bool - updated pass/fail state
    """
    rows = _sampled_rows(num_frames, verification)
    for name, arr in arrays.items():
        nonfinite_rows = []
        for i in rows:
            row = np.asarray(arr[i], dtype=np.float32)
            if not np.all(np.isfinite(row)):
                nonfinite_rows.append(i)
        if nonfinite_rows:
            print(f"FAIL: {name} has non-finite values at sampled rows {nonfinite_rows}")
            ok = False
        else:
            print(f"PASS: {name} finite at {len(rows)} sampled rows {rows[:3]}...{rows[-3:]}")

    # all-zero rows mean unwritten region
    if 'embeddings.npy' in arrays:
        rgb_emb = arrays['embeddings.npy']
        zero_rows = [i for i in rows if not np.any(rgb_emb[i])]
        if zero_rows:
            print(f"FAIL: found all-zero RGB embeddings at sampled rows {zero_rows} (interrupted write?)")
            ok = False
        else:
            print(f"PASS: sampled RGB embeddings ({len(rows)} rows), none all-zero")

    return ok

def _check_checksums(cache_dir, arrays, ok) -> bool:
    """
    Recompute per-block sha256 digests over full arrays and compare with checksums.json (slow).

    :param cache_dir: Path - embeddings cache directory
    :param arrays: dict - {array_name: memmap} to verify
    :param ok: bool - pass/fail state so far
    :return: bool - updated pass/fail state
    """
    checksums_path = cache_dir / 'checksums.json'
    if not checksums_path.exists():
        print("checksums: absent (legacy)")
        return ok

    with open(checksums_path, 'r') as f:
        checksums = json.load(f)
    block_rows = checksums['block_rows']

    for name, arr in arrays.items():
        if name not in checksums:
            print(f"FAIL: checksums.json has no entry for {name}")
            ok = False
            continue
        expected_digests = checksums[name]
        mismatched_blocks = []
        for block_idx, start in enumerate(range(0, arr.shape[0], block_rows)):
            block = arr[start:start + block_rows]
            digest = hashlib.sha256(np.ascontiguousarray(block).tobytes()).hexdigest()
            if digest != expected_digests[block_idx]:
                mismatched_blocks.append(block_idx)
        if mismatched_blocks:
            print(f"FAIL: {name} checksum mismatch at block(s) {mismatched_blocks}")
            ok = False
        else:
            print(f"PASS: {name} checksums match ({len(expected_digests)} blocks, full read)")
    return ok

def _check_depth_stats(cache_dir, fingerprint, ok) -> bool:
    """
    Compare depth_stats.json with the sha1 stored in the fingerprint.

    :param cache_dir: Path - embeddings cache directory
    :param fingerprint: dict - fingerprint.json contents
    :param ok: bool - pass/fail state so far
    :return: bool - updated pass/fail state
    """
    expected = fingerprint.get('depth_stats_sha1')
    if expected is None:
        return ok
    stats_path = cache_dir / 'depth_stats.json'
    if not stats_path.is_file():
        print("FAIL: depth_stats.json missing")
        return False
    digest = hashlib.sha1(stats_path.read_bytes()).hexdigest()
    if digest != expected:
        print(f"FAIL: depth_stats.json sha1 {digest} != fingerprint depth_stats_sha1 {expected}")
        return False
    print("PASS: depth_stats.json sha1 matches fingerprint")
    return ok

def check(cache_dir, verify_checksums=False, config=None) -> bool:
    """
    Run all cache checks, printing pass/fail per check.

    :param cache_dir: str | Path - embeddings cache directory
    :param verify_checksums: bool - also recompute full per-block checksums (slow)
    :param config: dict | None - CAT configuration, defaults loaded when None
    :return: bool - True if all checks pass
    """
    embedding = resolve_config(config)['embedding']
    cache_dir = Path(cache_dir)
    ok = True

    fingerprint_path = cache_dir / 'fingerprint.json'
    if not fingerprint_path.exists():
        print(f"FAIL: fingerprint.json missing in {cache_dir}")
        return False
    with open(fingerprint_path, 'r') as f:
        fingerprint = json.load(f)

    if fingerprint['complete']:
        print("PASS: fingerprint.json present and complete")
    else:
        print("FAIL: fingerprint.json present but complete=False")
        ok = False

    ok = _check_depth_stats(cache_dir, fingerprint, ok)

    arrays = {}
    for name in embedding['expected_arrays']:
        path = cache_dir / name
        if not path.exists():
            if name == 'depth_embeddings.npy' and not fingerprint['use_depth']:
                continue
            print(f"FAIL: {name} missing")
            ok = False
            continue

        header_shape, header_dtype, header_bytes = _npy_header_info(path)
        expected_shape = _expected_shape(name, fingerprint, embedding['expected_arrays'])
        expected_dtype = np.dtype(fingerprint['dtype'])
        if header_shape == expected_shape and header_dtype == expected_dtype:
            print(f"PASS: {name} shape={header_shape} dtype={header_dtype}")
        else:
            print(f"FAIL: {name} shape={header_shape} dtype={header_dtype}, "
                  f"expected shape={expected_shape} dtype={expected_dtype}")
            ok = False

        # file size = npy header + prod(shape) * itemsize
        expected_bytes = header_bytes + int(np.prod(header_shape)) * header_dtype.itemsize
        actual_bytes = path.stat().st_size
        if actual_bytes == expected_bytes:
            print(f"PASS: {name} file size {actual_bytes} bytes matches expected {expected_bytes} bytes")
        else:
            print(f"FAIL: {name} file size {actual_bytes} bytes, expected {expected_bytes} bytes "
                  f"(truncated or corrupted write)")
            ok = False
            continue  # truncated file is unsafe to mmap

        arrays[name] = np.load(str(path), mmap_mode='r')

    metadata_path = cache_dir / 'metadata.npy'
    if not metadata_path.exists():
        print("FAIL: metadata.npy missing")
        return False
    metadata = np.load(str(metadata_path), allow_pickle=True).item()

    num_frames = fingerprint['num_frames']
    for key in ('image_files', 'artificial_mask_files', 'traj_files', 'score_files'):
        if len(metadata[key]) == num_frames:
            print(f"PASS: metadata[{key}] length matches num_frames ({num_frames})")
        else:
            print(f"FAIL: metadata[{key}] length {len(metadata[key])} != num_frames {num_frames}")
            ok = False

    recomputed_id = sha1_of_paths(
        metadata['image_files'], metadata['artificial_mask_files'],
        metadata['traj_files'], metadata['score_files'],
    )
    if recomputed_id == fingerprint['dataset_id']:
        print("PASS: recomputed dataset_id matches fingerprint")
    else:
        print(f"FAIL: recomputed dataset_id {recomputed_id} != fingerprint dataset_id {fingerprint['dataset_id']}")
        ok = False

    ok = _check_sampled_rows(arrays, num_frames, ok, embedding['verification'])

    if verify_checksums:
        ok = _check_checksums(cache_dir, arrays, ok)
    else:
        checksums_path = cache_dir / 'checksums.json'
        if checksums_path.exists():
            print("checksums: present, not verified (pass --checksums for the full/slow check)")
        else:
            print("checksums: absent (legacy)")

    total_bytes = sum(f.stat().st_size for f in cache_dir.iterdir() if f.is_file())
    print(f"Total cache size on disk: {total_bytes / 1e9:.2f} GB")

    return ok

def cli(argv=None) -> None:
    """
    Verify an embeddings cache; exit 0 if all checks pass, else 1.

    :param argv: list[str] | None - command-line arguments (default: sys.argv)
    """
    parser = argparse.ArgumentParser(description="Verify an embeddings cache directory")
    parser.add_argument("cache_dir", type=str)
    parser.add_argument("--checksums", default=False, action='store_true',
                         help="Also recompute and compare full per-block checksums (slow, full read)")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML overriding the CAT configuration (embedding section)")
    args = parser.parse_args(argv)

    passed = check(args.cache_dir, verify_checksums=args.checksums, config=load_config(args.config))
    print("RESULT: PASS" if passed else "RESULT: FAIL")
    sys.exit(0 if passed else 1)

if __name__ == "__main__":
    cli()
