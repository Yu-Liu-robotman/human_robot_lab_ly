import os
import glob
import argparse

# ====== 你原来的函数直接保留 ======
from gmr_data_conversion import convert_pkl_to_custom   # 如果在同文件就删掉这行


def batch_convert(input_dir, output_dir, fps, recursive=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 选择是否递归
    if recursive:
        pattern = os.path.join(input_dir, "**", "*.pkl")
        pkl_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(input_dir, "*.pkl")
        pkl_files = glob.glob(pattern)

    if len(pkl_files) == 0:
        print("No PKL files found!")
        return

    print(f"Found {len(pkl_files)} files.")

    for pkl_path in pkl_files:
        # 生成输出路径
        filename = os.path.splitext(os.path.basename(pkl_path))[0]
        txt_path = os.path.join(output_dir, filename + ".txt")

        try:
            print(f"Processing: {pkl_path}")
            convert_pkl_to_custom(pkl_path, txt_path, fps)
        except Exception as e:
            print(f"❌ Failed: {pkl_path}")
            print(e)

    print("✅ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入PKL文件夹")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出TXT文件夹")
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--recursive", action="store_true",
                        help="是否递归子文件夹")

    args = parser.parse_args()

    batch_convert(
        args.input_dir,
        args.output_dir,
        args.fps,
        recursive=args.recursive
    )