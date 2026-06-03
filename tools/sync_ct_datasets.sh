#!/bin/bash

# ==========================================
# 路径配置区
# ==========================================
DEST_MAIN="/nas/share_data/CT/ct_dataset_base_260513"
DEST_PART0="/nas/share_data/CT/ct_dataset_base_260513/ct_dataset_base_260513_part0"

mkdir -p "$DEST_MAIN"
mkdir -p "$DEST_PART0"

# ==========================================
# 任务 1：合并传输 12 个基础文件夹 (带全局进度条)
# ==========================================
SOURCES_MAIN=(
    "/data2/ct_dataset_base_260513_part3_img"
    "/data2/0513_lz2nodesk_ct_chest_part3_doc"
    "/data2/ct_dataset_base_260513_part3_qc"
    "/data2/ct_dataset_base_260513_part4_img"
    "/data2/0513_lz2nodesk_ct_chest_part4_doc"
    "/data2/ct_dataset_base_260513_part4_qc"
    "/oss/share_data/CT/ct_dataset_base_260513/ct_dataset_base_260513_part5_img"
    "/oss/share_data/CT/ct_dataset_base_260513/0513_lz2nodesk_ct_chest_part5_doc"
    "/oss/share_data/CT/ct_dataset_base_260513/ct_dataset_base_260513_part5_qc"
    "/data4/ct_dataset_base_260513_part6_img"
    "/data4/0513_lz2nodesk_ct_chest_part6_doc"
    "/data4/ct_dataset_base_260513_part6_qc"
)

# 校验路径，剔除不存在的文件夹
VALID_SOURCES_MAIN=()
for src in "${SOURCES_MAIN[@]}"; do
    if [ -e "$src" ]; then
        VALID_SOURCES_MAIN+=("$src")
    else
        echo "❌ 警告: 找不到源路径 $src，已跳过"
    fi
done

if [ ${#VALID_SOURCES_MAIN[@]} -gt 0 ]; then
    echo -e "\n [任务 1/2] 开始同步 ${#VALID_SOURCES_MAIN[@]} 个基础文件夹..."
    echo " 正在进行全局扫描比对 (这将跳过已传文件，请稍候)..."
    
    # 核心修改：--info=progress2 提供唯一的全局进度条，--partial 支持断点续传
    rsync -a --info=progress2 --partial "${VALID_SOURCES_MAIN[@]}" "$DEST_MAIN/"
fi

# ==========================================
# 任务 2：传输新增的 part0 数据 (带全局进度条)
# ==========================================
SOURCES_PART0=(
    "/data1/sft/ct_workspace/processed/train_data"
    "/data1/sft/ct_workspace/train_reports.csv"
    "/data1/sft/ct_workspace/train_metadata.csv"
)

VALID_SOURCES_PART0=()
for src in "${SOURCES_PART0[@]}"; do
    if [ -e "$src" ]; then
        VALID_SOURCES_PART0+=("$src")
    else
        echo "❌ 警告: 找不到源路径 $src，已跳过"
    fi
done

if [ ${#VALID_SOURCES_PART0[@]} -gt 0 ]; then
    echo -e "\n [任务 2/2] 开始同步新增数据到 part0..."
    echo " 正在进行全局扫描比对..."
    
    rsync -a --info=progress2 --partial "${VALID_SOURCES_PART0[@]}" "$DEST_PART0/"
fi

echo -e "\n 所有数据传输及校验任务已全部完成！"
