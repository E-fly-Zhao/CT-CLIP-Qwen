import os
import subprocess
import argparse
import time
from pathlib import Path


def check_dcm2niix_installed():
    """检查系统是否已安装 dcm2niix"""
    try:
        subprocess.run(['dcm2niix', '-h'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False


def batch_convert(input_dir, output_dir):
    """
    自动化批量转换流程：按原文件夹名创建独立的输出子文件夹
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 确保输出主目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取输入目录下的所有一级子文件夹
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not subdirs:
        print(f"⚠️ 在 {input_dir} 下没有找到子文件夹。将尝试直接转换该目录...")
        subdirs = [input_path]

    total_folders = len(subdirs)
    print(f"🔍 扫描到 {total_folders} 个待处理文件夹。开始转换...\n" + "=" * 50)

    success_count = 0
    fail_count = 0

    start_time = time.time()

    # 遍历每个患者/检查的文件夹并执行转换
    for idx, folder in enumerate(subdirs, 1):
        folder_name = folder.name  # 例如：10050302800015
        print(f"⏳ [{idx}/{total_folders}] 正在处理: {folder_name} ...")

        patient_output_dir = output_path / folder_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        # dcm2niix 核心命令配置
        command = [
            'dcm2niix',
            '-z', 'y',  # 压缩为 .nii.gz
            '-f', '%i_%s_%p',  # 命名规则
            '-o', str(patient_output_dir),  # 输出到子文件夹
            str(folder)  # 输入的原始文件夹
        ]

        try:
            # 运行命令，捕获输出
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                print(f"   ✅ 成功! 文件已存入 -> {patient_output_dir.name}/")
                success_count += 1
            else:
                print(f"   ❌ 失败或警告! 详细信息:\n{result.stderr.strip()}")
                fail_count += 1

        except Exception as e:
            print(f"   ❌ 执行异常: {e}")
            fail_count += 1

    # 统计与耗时
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"🎉 批量转换任务完成！")
    print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
    print(f"📈 统计: 成功 {success_count} 个文件夹, 失败 {fail_count} 个文件夹。")
    print(f"📁 结果总目录: {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT-CLIP 自动化 DICOM 转 NIfTI 脚本 (独立子文件夹版)")
    parser.add_argument("-i", "--input_dir", required=True, help="原始 DICOM 数据所在的根目录")
    parser.add_argument("-o", "--output_dir", required=True, help="转换后 NIfTI 文件的输出主目录")

    args = parser.parse_args()

    if not check_dcm2niix_installed():
        print("❌ 严重错误: 系统中未找到 dcm2niix 工具。")
        print("请先通过 conda 安装: conda install -c conda-forge dcm2niix")
        exit(1)

    batch_convert(args.input_dir, args.output_dir)