from torch.utils.data import DataLoader

from teacher3d.data.dummy import DummySceneDataset
from teacher3d.data.scene_txt import TxtMetadataSceneDataset


def build_dataset(config):
    name = config.data.name
    if name == "dummy_scene":
        return DummySceneDataset(
            length=config.data.length,
            image_size=config.data.image_size,
            hidden_proposals=config.data.hidden_proposals,
        )
    if name in {"re10k", "acid", "txt_scene"}:
        return TxtMetadataSceneDataset(config)
    raise ValueError(f"Unsupported dataset: {name}")


def build_dataloader(config, shuffle: bool = True) -> DataLoader:
    dataset = build_dataset(config)
    return DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=shuffle,
        num_workers=config.data.num_workers,
    )


__all__ = ["DummySceneDataset", "TxtMetadataSceneDataset", "build_dataset", "build_dataloader"]
