import os

import pandas as pd
import pydicom
from tqdm import tqdm

# ================= Configuration =================
RAW_DICOM_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
DATASET_ROOT = "/data2/dataset"
REPORTS_CSV = f"{DATASET_ROOT}/reports.csv"
META_CSV_OUTPUT = f"{DATASET_ROOT}/metadata.csv"
START_IDX = 0
END_IDX = 1008

# ================= DICOM tag mapping =================
DICOM_KEYWORD_MAP = {
    "Manufacturer": "Manufacturer",
    "SeriesDescription": "SeriesDescription",
    "ManufacturerModelName": "ManufacturerModelName",
    "PatientSex": "PatientSex",
    "PatientAge": "PatientAge",
    "ReconstructionDiameter": "ReconstructionDiameter",
    "DistanceSourceToDetector": "DistanceSourceToDetector",
    "DistanceSourceToPatient": "DistanceSourceToPatient",
    "GantryDetectorTilt": "GantryDetectorTilt",
    "TableHeight": "TableHeight",
    "RotationDirection": "RotationDirection",
    "ExposureTime": "ExposureTime",
    "XRayTubeCurrent": "XRayTubeCurrent",
    "Exposure": "Exposure",
    "FilterType": "FilterType",
    "GeneratorPower": "GeneratorPower",
    "FocalSpots": "FocalSpot",
    "ConvolutionKernel": "ConvolutionKernel",
    "PatientPosition": "PatientPosition",
    "RevolutionTime": "RevolutionTime",
    "SingleCollimationWidth": "SingleCollimationWidth",
    "TotalCollimationWidth": "TotalCollimationWidth",
    "TableSpeed": "TableSpeed",
    "TableFeedPerRotation": "TableFeedPerRotation",
    "SpiralPitchFactor": "SpiralPitchFactor",
    "DataCollectionCenterPatient": "DataCollectionCenterPatient",
    "ReconstructionTargetCenterPatient": "ReconstructionTargetCenterPatient",
    "ExposureModulationType": "ExposureModulationType",
    "CTDIvol": "CTDIvol",
    "ImagePositionPatient": "ImagePositionPatient",
    "ImageOrientationPatient": "ImageOrientationPatient",
    "SliceLocation": "SliceLocation",
    "SamplesPerPixel": "SamplesPerPixel",
    "PhotometricInterpretation": "PhotometricInterpretation",
    "Rows": "Rows",
    "Columns": "Columns",
    "XYSpacing": "PixelSpacing",
    "RescaleIntercept": "RescaleIntercept",
    "RescaleSlope": "RescaleSlope",
    "RescaleType": "RescaleType",
    "ZSpacing": "SliceThickness",
    "StudyDate": "StudyDate",
}


def format_dicom_value(val):
    """Convert DICOM values to the expected CSV string representation."""
    if val is None:
        return "缺失"

    if isinstance(val, (pydicom.multival.MultiValue, list)):
        return "[" + ", ".join(str(x) for x in val) + "]"

    return str(val).strip()


def normalize_folder_name(val):
    """Normalize folder identifiers so path joins always receive clean strings."""
    if pd.isna(val):
        return None

    text = str(val).strip()
    if not text or text.lower() == "nan":
        return None

    # Defensive fix for CSVs that were inferred as floats before being serialized.
    if text.endswith(".0"):
        text = text[:-2]

    return text


def derive_folder_name(row):
    """Prefer SourceFolder, then fall back to the first segment of VolumeName."""
    folder_name = normalize_folder_name(row.get("SourceFolder"))
    if folder_name:
        return folder_name

    volume_name = normalize_folder_name(row.get("VolumeName"))
    if not volume_name:
        return None

    return volume_name.split("/")[0].split("\\")[0]


def find_real_3d_slice_and_count(base_folder):
    """
    Find a representative axial CT slice and estimate slice count from its series directory.
    This skips scout/localizer images and non-image/report files.
    """
    valid_ds = None
    target_dir = None

    for root, _, files in os.walk(base_folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                img_type = [str(x).upper() for x in getattr(ds, "ImageType", [])]
                if "LOCALIZER" in img_type or "SCOUT" in img_type:
                    continue

                if (
                    not hasattr(ds, "PixelSpacing")
                    or not hasattr(ds, "RescaleIntercept")
                    or not hasattr(ds, "RescaleSlope")
                ):
                    continue

                valid_ds = ds
                target_dir = root
                break
            except Exception:
                continue

        if valid_ds:
            break

    slice_count = 0
    if valid_ds and target_dir:
        slice_count = len(
            [
                f
                for f in os.listdir(target_dir)
                if os.path.isfile(os.path.join(target_dir, f))
            ]
        )

    return valid_ds, slice_count


def main():
    if not os.path.exists(REPORTS_CSV):
        print(f"找不到索引文件: {REPORTS_CSV}")
        return

    print("正在加载报告索引...")
    df_reports = pd.read_csv(REPORTS_CSV, dtype=str, keep_default_na=False)
    df_reports["folder_name"] = df_reports.apply(derive_folder_name, axis=1)
    df_reports = df_reports[df_reports["folder_name"].notna()].copy()
    unique_folders = df_reports["folder_name"].drop_duplicates().tolist()[START_IDX:END_IDX]

    metadata_records = []

    print(f"开始从原始 DICOM 中提取严谨 Metadata (共 {len(unique_folders)} 个患者)...")

    for folder_name in tqdm(unique_folders, desc="挖掘 DICOM"):
        folder_name = normalize_folder_name(folder_name)
        if not folder_name:
            continue

        dicom_folder_path = os.path.join(RAW_DICOM_DIR, folder_name)

        row_data = {col: "缺失" for col in DICOM_KEYWORD_MAP.keys()}
        row_data["folder_name"] = folder_name
        row_data["NumberofSlices"] = "缺失"

        if os.path.isdir(dicom_folder_path):
            dcm_dataset, slice_count = find_real_3d_slice_and_count(dicom_folder_path)

            if dcm_dataset is not None:
                row_data["NumberofSlices"] = slice_count

                for csv_col, dcm_keyword in DICOM_KEYWORD_MAP.items():
                    try:
                        val = getattr(dcm_dataset, dcm_keyword, None)

                        if csv_col == "ZSpacing" and val is None:
                            val = getattr(dcm_dataset, "SpacingBetweenSlices", None)

                        row_data[csv_col] = format_dicom_value(val)
                    except Exception:
                        pass

        metadata_records.append(row_data)

    df_meta = pd.DataFrame(metadata_records)

    print("\n正在与图文对齐索引合并...")
    df_final = df_reports[["VolumeName", "folder_name"]].merge(df_meta, on="folder_name", how="left")
    df_final = df_final.drop(columns=["folder_name"])

    df_final.to_csv(META_CSV_OUTPUT, index=False, encoding="utf-8-sig")
    print("=" * 50)
    print("DICOM Metadata 提取完毕，结果已保存。")
    print(f"输出路径: {META_CSV_OUTPUT}")


if __name__ == "__main__":
    main()
