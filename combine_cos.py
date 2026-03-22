"""
Combine two BlenderNeRF COS training datasets into one.

Non-destructive: always writes to a new output folder, never modifies inputs.

Usage:
    python combine_cos.py <dataset_a> <dataset_b> <output_dir>
    python combine_cos.py cos_close/ cos_far/ combined/ --force

Inputs can be folders or .zip files (will be extracted to a temp dir).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


INTRINSIC_KEYS = ["camera_angle_x", "camera_angle_y", "fl_x", "fl_y",
                  "k1", "k2", "p1", "p2", "cx", "cy", "w", "h", "aabb_scale"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def resolve_input(path):
    """Return a directory path — extract zip to a temp dir if needed."""
    p = Path(path)
    if p.is_dir():
        return p, None
    if p.suffix == ".zip" and p.is_file():
        tmp = tempfile.mkdtemp(prefix="cos_combine_")
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(tmp)
        # if the zip contains a single subfolder, use that
        entries = list(Path(tmp).iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            return entries[0], tmp
        return Path(tmp), tmp
    sys.exit(f"Error: '{path}' is not a folder or zip file.")


def detect_image_dir(dataset_dir):
    """Return the image subdirectory name ('train' or 'images')."""
    for name in ("train", "images"):
        if (dataset_dir / name).is_dir():
            return name
    return None


def check_intrinsics(data_a, data_b, force=False):
    """Warn if intrinsics don't match between the two datasets."""
    mismatches = []
    for key in INTRINSIC_KEYS:
        va = data_a.get(key)
        vb = data_b.get(key)
        if va is None or vb is None:
            continue
        if isinstance(va, float):
            if abs(va - vb) > 1e-6:
                mismatches.append(f"  {key}: {va} vs {vb}")
        elif va != vb:
            mismatches.append(f"  {key}: {va} vs {vb}")

    if mismatches:
        msg = "Intrinsics mismatch between datasets:\n" + "\n".join(mismatches)
        if force:
            print(f"WARNING: {msg}\nUsing dataset A's intrinsics (--force).")
        else:
            sys.exit(f"ERROR: {msg}\nUse --force to combine anyway (uses dataset A's intrinsics).")


def merge_frames(frames_a, frames_b, img_dir_a, img_dir_b, out_img_dir, output_dir):
    """
    Copy images and build a merged frame list.
    Dataset A keeps original numbering, dataset B is renumbered starting after A.
    """
    merged = []
    out_train = output_dir / out_img_dir
    out_train.mkdir(parents=True, exist_ok=True)

    # copy dataset A images as-is
    for frame in frames_a:
        src_rel = frame["file_path"]
        src_name = Path(src_rel).name
        # add extension if missing
        src_file = find_image(img_dir_a, src_name)
        if src_file:
            dst_name = src_file.name
            shutil.copy2(src_file, out_train / dst_name)
            new_frame = dict(frame)
            new_frame["file_path"] = f"{out_img_dir}/{Path(dst_name).stem}"
            merged.append(new_frame)
        else:
            print(f"  Warning: image not found for {src_rel}, skipping frame")

    # renumber and copy dataset B images
    offset = len(frames_a)
    for i, frame in enumerate(frames_b):
        src_rel = frame["file_path"]
        src_name = Path(src_rel).name
        src_file = find_image(img_dir_b, src_name)
        if src_file:
            ext = src_file.suffix
            new_index = offset + i
            dst_name = f"r_{new_index:04d}{ext}"
            shutil.copy2(src_file, out_train / dst_name)
            new_frame = dict(frame)
            new_frame["file_path"] = f"{out_img_dir}/r_{new_index:04d}"
            merged.append(new_frame)
        else:
            print(f"  Warning: image not found for {src_rel}, skipping frame")

    return merged


def find_image(img_dir, stem):
    """Find an image file by stem (with or without extension)."""
    stem = Path(stem).stem
    for ext in (".png", ".jpg", ".jpeg", ".exr"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def merge_ply(ply_a, ply_b, output_path):
    """Merge two ASCII PLY files by concatenating vertices."""
    if not ply_a.exists() or not ply_b.exists():
        # just copy whichever exists
        if ply_a.exists():
            shutil.copy2(ply_a, output_path)
        elif ply_b.exists():
            shutil.copy2(ply_b, output_path)
        return

    header_a, verts_a, count_a = parse_ply(ply_a)
    _, verts_b, count_b = parse_ply(ply_b)

    total = count_a + count_b

    with open(output_path, "w") as f:
        for line in header_a:
            if line.startswith("element vertex"):
                f.write(f"element vertex {total}\n")
            else:
                f.write(line + "\n")
        for v in verts_a:
            f.write(v + "\n")
        for v in verts_b:
            f.write(v + "\n")

    print(f"  Merged PLY: {count_a} + {count_b} = {total} vertices")


def parse_ply(path):
    """Parse an ASCII PLY, returning (header_lines, vertex_lines, vertex_count)."""
    header = []
    vertices = []
    count = 0
    in_header = True

    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if in_header:
                header.append(line)
                if line.startswith("element vertex"):
                    count = int(line.split()[-1])
                if line == "end_header":
                    in_header = False
            else:
                vertices.append(line)

    return header, vertices, count


def main():
    parser = argparse.ArgumentParser(
        description="Combine two BlenderNeRF COS datasets into one."
    )
    parser.add_argument("dataset_a", help="Path to first COS dataset (folder or .zip)")
    parser.add_argument("dataset_b", help="Path to second COS dataset (folder or .zip)")
    parser.add_argument("output_dir", help="Path to output combined dataset")
    parser.add_argument("--force", action="store_true",
                        help="Combine even if intrinsics don't match")
    args = parser.parse_args()

    # resolve inputs (extract zips if needed)
    dir_a, tmp_a = resolve_input(args.dataset_a)
    dir_b, tmp_b = resolve_input(args.dataset_b)

    output = Path(args.output_dir)
    if output.exists():
        sys.exit(f"Error: output directory '{output}' already exists. Choose a new path.")
    output.mkdir(parents=True)

    try:
        _combine(dir_a, dir_b, output, args.force)
    finally:
        # clean up temp dirs from zip extraction
        if tmp_a:
            shutil.rmtree(tmp_a, ignore_errors=True)
        if tmp_b:
            shutil.rmtree(tmp_b, ignore_errors=True)


def _combine(dir_a, dir_b, output, force):
    # load training transforms
    train_a_path = dir_a / "transforms_train.json"
    train_b_path = dir_b / "transforms_train.json"

    if not train_a_path.exists():
        sys.exit(f"Error: {train_a_path} not found")
    if not train_b_path.exists():
        sys.exit(f"Error: {train_b_path} not found")

    data_a = load_json(train_a_path)
    data_b = load_json(train_b_path)

    print(f"Dataset A: {len(data_a['frames'])} frames")
    print(f"Dataset B: {len(data_b['frames'])} frames")

    check_intrinsics(data_a, data_b, force)

    # detect image directories
    imgdir_a_name = detect_image_dir(dir_a)
    imgdir_b_name = detect_image_dir(dir_b)

    if imgdir_a_name is None:
        sys.exit(f"Error: no 'train/' or 'images/' directory found in {dir_a}")
    if imgdir_b_name is None:
        sys.exit(f"Error: no 'train/' or 'images/' directory found in {dir_b}")

    # use dataset A's convention for output
    out_img_dir = imgdir_a_name

    print(f"Merging images into '{out_img_dir}/'...")
    merged_frames = merge_frames(
        data_a["frames"], data_b["frames"],
        dir_a / imgdir_a_name, dir_b / imgdir_b_name,
        out_img_dir, output
    )

    # build combined transforms (intrinsics from A)
    combined = {k: v for k, v in data_a.items() if k != "frames"}
    combined["frames"] = merged_frames
    save_json(output / "transforms_train.json", combined)
    print(f"  Combined: {len(merged_frames)} total frames")

    # merge test transforms if both exist
    test_a = dir_a / "transforms_test.json"
    test_b = dir_b / "transforms_test.json"
    if test_a.exists() and test_b.exists():
        tdata_a = load_json(test_a)
        tdata_b = load_json(test_b)
        # test images go in 'test/' dir — copy without renumbering conflicts
        test_out = output / "test"
        test_out.mkdir(exist_ok=True)
        test_frames = []
        idx = 0
        for frame in tdata_a["frames"] + tdata_b["frames"]:
            src_name = Path(frame["file_path"]).name
            for d in (dir_a / "test", dir_b / "test"):
                src = find_image(d, src_name)
                if src:
                    ext = src.suffix
                    dst = f"r_{idx:04d}{ext}"
                    shutil.copy2(src, test_out / dst)
                    new_frame = dict(frame)
                    new_frame["file_path"] = f"test/r_{idx:04d}"
                    test_frames.append(new_frame)
                    idx += 1
                    break
        test_combined = {k: v for k, v in tdata_a.items() if k != "frames"}
        test_combined["frames"] = test_frames
        save_json(output / "transforms_test.json", test_combined)
        print(f"  Combined test: {len(test_frames)} frames")
    elif test_a.exists() or test_b.exists():
        src = test_a if test_a.exists() else test_b
        shutil.copy2(src, output / "transforms_test.json")
        print("  Copied test transforms from one dataset (only one had them)")

    # merge PLY files
    ply_a = dir_a / "points3d.ply"
    ply_b = dir_b / "points3d.ply"
    if ply_a.exists() or ply_b.exists():
        print("Merging PLY point clouds...")
        merge_ply(ply_a, ply_b, output / "points3d.ply")

    print(f"\nDone! Combined dataset written to: {output}")


if __name__ == "__main__":
    main()
