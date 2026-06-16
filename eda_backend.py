import os
import struct
import json
import concurrent.futures
import pandas as pd

CACHE_DIR = ".eda_cache"

def get_cache_paths(jsonl_path):
    """
    Returns the paths for the .idx file and the metadata .parquet file
    corresponding to the input jsonl_path.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    base_name = os.path.basename(jsonl_path)
    # Use MD5 or simple hashing if filenames might collide, but base_name is unique enough for this dataset.
    idx_path = os.path.join(CACHE_DIR, f"{base_name}.idx")
    parquet_path = os.path.join(CACHE_DIR, f"{base_name}.parquet")
    return idx_path, parquet_path

def get_or_build_index(jsonl_path, progress_callback=None):
    """
    Retrieves the byte offsets of lines in a JSONL file.
    If the binary index file (.idx) doesn't exist, it builds it.

    Returns a list of integer byte offsets.
    """
    idx_path, _ = get_cache_paths(jsonl_path)

    if os.path.exists(idx_path):
        try:
            with open(idx_path, 'rb') as f:
                data = f.read()
            count = len(data) // 8
            offsets = list(struct.unpack(f"<{count}q", data))
            return offsets
        except Exception as e:
            # If corruption or reading error, rebuild
            pass

    # Build index sequentially
    offsets = []
    file_size = os.path.getsize(jsonl_path)

    with open(jsonl_path, 'rb') as f:
        offset = 0
        while True:
            offsets.append(offset)
            line = f.readline()
            if not line:
                offsets.pop()  # Remove trailing EOF offset
                break

            offset = f.tell()
            if progress_callback and len(offsets) % 5000 == 0:
                progress_callback(offset, file_size)

    if progress_callback:
        progress_callback(file_size, file_size)

    # Save to binary index file (.idx)
    with open(idx_path, 'wb') as f_out:
        f_out.write(struct.pack(f"<{len(offsets)}q", *offsets))

    return offsets

DEFAULT_MAPPINGS = {
    "id": "id",
    "score": "score",
    "rating": "rating",
    "file_ext": "file_ext",
    "image_width": "image_width",
    "image_height": "image_height",
    "fav_count": "fav_count",
    "created_at": "created_at",
    "tags": ["tags", "tag_string"],
    "regular_summary": "regular_summary",
    "individual_parts": "individual_parts"
}

def resolve_config_key(record, key_mapping):
    """
    Safely resolves a configuration mapping value (string, dotted-string, or list of fallbacks)
    against a given dictionary record.
    """
    if not record or not key_mapping:
        return None

    # Standardize list of fallbacks
    keys = key_mapping if isinstance(key_mapping, list) else [key_mapping]

    for k in keys:
        if not isinstance(k, str):
            continue
        parts = k.split('.')
        current = record
        found = True
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found and current is not None:
            return current
    return None

def get_or_build_metadata(jsonl_path, offsets, mappings=None, progress_callback=None):
    """
    Retrieves the queryable metadata DataFrame.
    If the metadata parquet cache file doesn't exist, it builds it.

    Returns a pandas DataFrame.
    """
    _, parquet_path = get_cache_paths(jsonl_path)

    if mappings is None:
        try:
            if os.path.exists("schema_config.json"):
                with open("schema_config.json", "r") as f:
                    config = json.load(f)
                mappings = config.get("mappings", DEFAULT_MAPPINGS)
            else:
                mappings = DEFAULT_MAPPINGS
        except Exception:
            mappings = DEFAULT_MAPPINGS

    expected_cols = set(mappings.keys()) | {'byte_offset'}

    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            # Ensure proper columns
            if expected_cols.issubset(df.columns):
                return df
        except Exception as e:
            # If reading error, rebuild
            pass

    # Build metadata from JSONL lines sequentially
    metadata_list = []
    total_offsets = len(offsets)

    with open(jsonl_path, 'rb') as f:
        for idx, offset in enumerate(offsets):
            f.seek(offset)
            line_bytes = f.readline()
            if not line_bytes:
                continue

            try:
                line_str = line_bytes.decode('utf-8', errors='ignore')
                record = json.loads(line_str)

                record_data = {}
                for logical_key, physical_key in mappings.items():
                    val = resolve_config_key(record, physical_key)
                    # Safely parse values depending on logical key type
                    if logical_key in ['id', 'score', 'fav_count', 'image_width', 'image_height']:
                        if val is None or val == "":
                            val = 0
                        else:
                            try:
                                val = int(float(val))
                            except (ValueError, TypeError):
                                val = 0
                    elif logical_key == 'file_ext':
                        val = str(val or '').strip().lower()
                    elif logical_key in ['rating', 'created_at', 'tags', 'regular_summary', 'individual_parts']:
                        val = str(val or '').strip()
                    else:
                        if isinstance(val, (int, float, bool, str)) or val is None:
                            pass
                        else:
                            val = str(val)
                    record_data[logical_key] = val

                record_data['byte_offset'] = offset
                metadata_list.append(record_data)
            except Exception:
                # Malformed JSON lines are skipped gracefully
                pass

            if progress_callback and idx % 2000 == 0:
                progress_callback(idx, total_offsets)

    if progress_callback:
        progress_callback(total_offsets, total_offsets)

    df = pd.DataFrame(metadata_list)
    # Save cache file
    df.to_parquet(parquet_path, index=False)
    return df

def read_single_record_binary(jsonl_path, offset):
    """
    Seeks to a byte offset, reads a single line, and returns decoded JSON.
    """
    with open(jsonl_path, 'rb') as f:
        f.seek(offset)
        line_bytes = f.readline()
        if not line_bytes:
            return None
        line_str = line_bytes.decode('utf-8', errors='ignore')
        return json.loads(line_str)

def read_records_lazy(jsonl_path, offsets, max_workers=10):
    """
    Asynchronously reads lines at multiple byte offsets using ThreadPoolExecutor.

    Returns a list of dicts.
    """
    results = [None] * len(offsets)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(read_single_record_binary, jsonl_path, offset): idx
            for idx, offset in enumerate(offsets)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = {"error": str(e)}
    return results

def fetch_records_at_offsets(jsonl_path, offsets, max_workers=10):
    """
    Asynchronously reads lines at multiple byte offsets using ThreadPoolExecutor.
    Alias for read_records_lazy.
    """
    return read_records_lazy(jsonl_path, offsets, max_workers)

def stream_export_jsonl(src_path, dst_path, offsets, progress_callback=None):
    """
    Streams exact lines from source file matching standard byte offsets to destination file.
    Does not decode or parse JSON, maximizing streaming speed.
    """
    total = len(offsets)
    with open(src_path, 'rb') as f_in, open(dst_path, 'wb') as f_out:
        for i, offset in enumerate(offsets):
            f_in.seek(offset)
            line_bytes = f_in.readline()
            f_out.write(line_bytes)
            if progress_callback and i % 1000 == 0:
                progress_callback(i, total)
        if progress_callback:
            progress_callback(total, total)

def trim_jsonl(src_path, dst_path, all_offsets, exclude_offsets, progress_callback=None):
    """
    Writes a new JSONL excluding specified offsets.
    """
    exclude_set = set(exclude_offsets)
    keep_offsets = [offset for offset in all_offsets if offset not in exclude_set]
    stream_export_jsonl(src_path, dst_path, keep_offsets, progress_callback)
    return len(keep_offsets)
