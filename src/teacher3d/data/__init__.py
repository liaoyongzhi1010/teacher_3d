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
    num_workers = int(getattr(config.data, "num_workers", 0))
    pin_memory = bool(getattr(config.data, "pin_memory", num_workers > 0))
    persistent_workers = bool(getattr(config.data, "persistent_workers", num_workers > 0)) and num_workers > 0
    kwargs = {
        "dataset": dataset,
        "batch_size": int(config.data.batch_size),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(getattr(config.data, "prefetch_factor", 2))
    return DataLoader(**kwargs)


__all__ = ["DummySceneDataset", "TxtMetadataSceneDataset", "build_dataset", "build_dataloader"]
