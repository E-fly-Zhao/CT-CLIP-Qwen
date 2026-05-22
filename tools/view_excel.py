import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

# 配置文件路径
excel_path = "/mnt/share_data/CT/ct_dataset_base_260316/0316_nodesk_ct_chest_1000.xlsx"

# 初始化 rich 控制台，它会自动获取当前终端的屏幕宽度
console = Console()


def display_dataframe(df_slice):
    """
    使用 rich 库渲染表格：
    1. 自动适应终端屏幕宽度
    2. 单元格内文本自动换行，绝不截断
    3. 严格对齐中英文字符
    """
    # 创建富文本表格，开启网格线以便于阅读多行文本
    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        show_lines=True  # 开启行与行之间的分割线，防止换行后上下行文字粘连
    )

    # 将原本隐藏的行号(Index)作为第一列加入，方便定位
    df_slice = df_slice.reset_index()
    df_slice.rename(columns={'index': 'Row_ID'}, inplace=True)

    # 动态添加表头
    for col in df_slice.columns:
        # overflow="fold" 是核心：允许长文本根据屏幕宽度自动折行，而不是省略
        table.add_column(str(col), overflow="fold")

    # 添加数据行
    for _, row in df_slice.iterrows():
        # 将所有数据转换为字符串，并将 NaN (空值) 替换为空白显示，保持整洁
        row_data = [str(item) if pd.notna(item) else "" for item in row]
        table.add_row(*row_data)

    # 在终端中渲染并打印完美表格
    console.print(table)


def main():
    console.print("[yellow]正在加载 Excel 文件，请稍候...[/yellow]")
    try:
        # 读取数据
        df = pd.read_excel(excel_path)
    except Exception as e:
        console.print(f"[red]❌ 加载失败: {e}[/red]")
        return

    console.print("\n[green]✅ 加载成功！[/green]")
    console.print(f"📊 数据集概览: 共 [bold cyan]{df.shape[0]}[/bold cyan] 行, [bold cyan]{df.shape[1]}[/bold cyan] 列")
    console.print(f"🏷️ 包含的列名: {list(df.columns)}")
    print("-" * 60)

    while True:
        print("\n选项操作：")
        print("1. 输入行范围查看 (例如: 0-5)")
        print("2. 输入单个列名查看前10行 (例如: PatientID)")
        print("3. 输入 'q' 退出")

        choice = input("👉 请输入您的指令: ").strip()

        if choice.lower() == 'q':
            print("退出查看。")
            break

        # 解析行范围 (如 0-5)
        if '-' in choice:
            try:
                start, end = map(int, choice.split('-'))
                # 截取用户请求的行数据，送入富文本渲染器
                display_dataframe(df.iloc[start:end])
            except ValueError:
                console.print("[red]⚠️ 输入格式有误，请输入纯数字范围，如 '0-5'。[/red]")

        # 解析单个列名
        elif choice in df.columns:
            display_dataframe(df[[choice]].head(10))

        else:
            console.print("[red]⚠️ 无法识别的指令或列名不存在，请检查大小写重试。[/red]")


if __name__ == "__main__":
    main()