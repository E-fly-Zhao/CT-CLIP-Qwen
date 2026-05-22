import nibabel as nib
import os
from rich.console import Console

console = Console()

# 把这两个大文件的名字填进来
nii_files = [
    "/home/huali/code/CT-CLIP-main/tools/P0000520759_3102717156_Chest.nii.gz",
    "/home/huali/code/CT-CLIP-main/tools/P0000520759_3888079615_Chest.nii.gz"
]


def check_nifti_resolution():
    for file_path in nii_files:
        if not os.path.exists(file_path):
            console.print(f"[red]找不到文件: {file_path}[/red]")
            continue

        # 加载 NIfTI 文件 (只加载 Header，不加载庞大的图像数据到内存，瞬间完成)
        img = nib.load(file_path)

        # 提取图像的三维形状 (X, Y, Z)
        shape = img.shape
        # 提取体素的物理间距 (X_spacing, Y_spacing, Z_spacing)
        zooms = img.header.get_zooms()

        filename = os.path.basename(file_path)
        console.print(f"\n[cyan]▶ 正在分析: {filename}[/cyan]")
        console.print("-" * 50)

        # 打印空间维度
        console.print(f"📦 [bold]空间矩阵大小 (Shape)[/bold]: {shape}")
        if len(shape) == 3:
            console.print(f"   说明: 这是一个 3D 体积，由 {shape[2]} 张切片组成。")

        # 打印物理间距
        console.print(f"📏 [bold]物理体素间距 (Voxel Spacing/Zooms)[/bold]: {zooms}")
        if len(zooms) >= 3:
            z_spacing = zooms[2]
            console.print(f"   [yellow]核心结论: 该图像的真实层厚 (Z轴间距) 为 {z_spacing:.3f} mm[/yellow]")

            if z_spacing <= 2.0:
                console.print("   [green]✅ 鉴定完毕：这是真正的高精度薄层！[/green]")
            else:
                console.print(f"   [red]⚠️ 警告：层厚为 {z_spacing:.3f} mm，这是厚层！[/red]")
        console.print("-" * 50)


if __name__ == "__main__":
    check_nifti_resolution()