import glob
import os
from typing import Any, Dict, Generator, List

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.conf import DataConfig


class MetadataExtractor:
    def __init__(self, config: DataConfig) -> None:
        self.config = config

    def get_data_frame(self) -> pd.DataFrame:
        metadata: List[Dict[str, Any]] = []
        for file_path in self._get_file_paths_generator():
            file_metadata = self._get_file_metadata(file_path)
            file_metadata.update(file_path)
            metadata.append(file_metadata)
        metadata_df = self._create_data_frame(metadata)
        return metadata_df

    def _get_file_paths_generator(self) -> Generator[Dict[str, str], None, None]:
        all_paths = os.path.join(str(self.config.dataset_path), "**", "*.tiff")
        for path in glob.iglob(all_paths, recursive=True):
            folder_name, file_name = path.split(os.sep)[-2:]
            yield {"folder_name": folder_name, "file_name": file_name}

    def _create_data_frame(self, metadata: List[Dict[str, Any]]) -> pd.DataFrame:
        metadata_df: pd.DataFrame = pd.DataFrame(metadata)
        metadata_df = self.group_channels(metadata_df)
        metadata_df = metadata_df[self.config.metadata_output_columns]
        metadata_df["well_id"] = LabelEncoder().fit_transform(metadata_df["well_id"])
        return metadata_df

    @staticmethod
    def group_channels(metadata_df: pd.DataFrame) -> pd.DataFrame:
        metadata_df["ids"] = metadata_df["folder_name"] + metadata_df["file_id"]
        metadata_df = metadata_df.pivot(index="ids", columns="channel").T.drop_duplicates().T
        new_colum_names = [i[0] + str(i[1]) if i[0] == "file_name" else i[0] for i in metadata_df.columns.tolist()]
        metadata_df.columns = new_colum_names
        metadata_df.reset_index(drop=True, inplace=True)
        return metadata_df

    @staticmethod
    def extract_file_metadata(file_name: str, folder_name: str) -> Dict[str, Any]:
        compound_id = int(file_name[1:3])
        concentration_id = int(file_name[4:6])
        channel = int(file_name[15:16])
        well_id = folder_name + file_name[0:6]
        file_id = (
            file_name[1:3]
            + file_name[4:6]
            + file_name[7:9]
            + file_name[10:12]
            + file_name[18:19]
            + file_name[21:22]
            + file_name[24:25]
        )
        return {
            "compound_id": compound_id,
            "concentration_id": concentration_id,
            "channel": channel,
            "file_id": file_id,
            "well_id": well_id,
        }

    @staticmethod
    def map_file_metadata(
        file_metadata: Dict[str, Any], compound_mapping: Dict[int, str], concentration_mapping: Dict[int, float]
    ) -> Dict[str, Any]:
        file_metadata["compound_name"] = compound_mapping[file_metadata["compound_id"]]
        file_metadata["concentration"] = concentration_mapping[file_metadata["concentration_id"]]
        return file_metadata

    def _get_file_metadata(self, file_path: Dict[str, str]) -> Dict[str, Any]:
        metadata = self.extract_file_metadata(file_name=file_path["file_name"], folder_name=file_path["folder_name"])
        metadata = self.map_file_metadata(
            file_metadata=metadata,
            compound_mapping=self.config.compound_mapping,
            concentration_mapping=self.config.concentration_mapping,
        )
        return metadata
