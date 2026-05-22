import os
import json
from rich.console import Console
from rich.table import Table
from rich import box

# 配置你的路径
dicom_dir = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/10050302800015"
nifti_dir = "/home/huali/code/CT-CLIP-main/tools"

console = Console()


def get_dir_size(folder_path):
    total_size = 0
    for dirpath, _, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def get_file_size(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)


def analyze_nifti_outputs():
    console.print("\n[yellow]正在扫描与分析转换结果，请稍候...[/yellow]")
    dicom_size_mb = get_dir_size(dicom_dir)

    table = Table(title="NIfTI 转换结果分析与体素诊断 (V2 增强版)", box=box.ROUNDED, header_style="bold cyan")
    table.add_column("NIfTI 文件名", style="green")
    table.add_column("大小 (MB)", justify="right")
    table.add_column("层厚 (mm)", justify="center", style="magenta")
    table.add_column("图像类型 (Image Type)", style="yellow")
    table.add_column("AI 预训练建议", style="bold white")

    total_nifti_size = 0
    nifti_files = [f for f in os.listdir(nifti_dir) if f.endswith('.nii.gz')]

    for nii_file in sorted(nifti_files):
        nii_path = os.path.join(nifti_dir, nii_file)
        json_path = os.path.join(nifti_dir, nii_file.replace('.nii.gz', '.json'))

        size_mb = get_file_size(nii_path)
        total_nifti_size += size_mb

        thickness_val = None
        image_type = "未知"

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    # 尝试获取层厚，如果 SliceThickness 没有，尝试 SpacingBetweenSlices
                    thickness_val = meta.get("SliceThickness")
                    if thickness_val is None:
                        thickness_val = meta.get("SpacingBetweenSlices")

                    image_type = ", ".join(meta.get("ImageType", []))
            except Exception:
                image_type = "JSON读取出错"

        # 核心判断逻辑 (结合层厚与文件大小)
        advice = ""
        thickness_display = str(thickness_val) if thickness_val is not None else "未知/缺失"

        if "LOCALIZER" in image_type.upper():
            advice = "[red]❌ 丢弃 (2D定位图)[/red]"
        else:
            # 优先使用 Metadata 判断
            if thickness_val is not None:
                try:
                    t = float(thickness_val)
                    if t == 0:
                        advice = "[red]❌ 丢弃 (2D定位图)[/red]"
                    elif t > 3.0:
                        advice = "[yellow]⚠️ 丢弃 (Z轴分辨率极差的厚层)[/yellow]"
                    else:
                        advice = "[green]✅ 完美 (保留用于 3D 预训练)[/green]"
                except ValueError:
                    pass  # 如果转换 float 依然失败，交给下面的体积兜底逻辑

            # 如果 Metadata 缺失，使用物理体积进行兜底智能推断
            if advice == "":
                if size_mb > 30.0:
                    advice = "[green]✅ 完美 (大体积文件，推断为高精度薄层)[/green]"
                elif size_mb > 5.0:
                    advice = "[yellow]⚠️ 丢弃 (小体积文件，推断为厚层)[/yellow]"
                else:
                    advice = "[red]❌ 丢弃 (体积过小，推测为非标准图像)[/red]"

        base_name = nii_file.replace('.nii.gz', '')
        table.add_row(base_name, f"{size_mb:.2f}", thickness_display, image_type, advice)

    console.print(table)

    print("\n" + "=" * 50)
    console.print(f"📁 原始 DICOM 总大小: [bold red]{dicom_size_mb:.2f} MB[/bold red]")
    console.print(f"📦 转换后 nii.gz 总大小: [bold green]{total_nifti_size:.2f} MB[/bold green]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    analyze_nifti_outputs()