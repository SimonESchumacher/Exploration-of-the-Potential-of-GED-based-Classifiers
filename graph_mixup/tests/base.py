import pytest
from lightning import seed_everything


class BaseTest:
    @pytest.fixture(autouse=True)
    def seed(self):
        seed_everything(0)
