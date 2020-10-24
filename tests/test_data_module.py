from core.data.data_module import PCAMDataModule


def test_data_module_setup():
    data_module = PCAMDataModule('/Users/dilip.thiagarajan/data/pcam/', 128, 0, 32)
    data_module.setup()
    index, image, label = data_module.train_dataset[1]
    assert image.shape == (3, 32, 32)
