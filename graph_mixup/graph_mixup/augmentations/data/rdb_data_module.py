import logging
import random
from typing import Generic, assert_never

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing_extensions import override

from graph_mixup.augmentations.data.abstract_data_module import (
    AbstractDataModule,
)
from graph_mixup.augmentations.data.typing import (
    DataModuleConfigType,
)
from graph_mixup.mixup_generation.s_mixup.method.gmnet import (
    LitGraphMatchingNet,
)
from graph_mixup.mixup_generation.s_mixup.method.gmnet.training import (
    train_gmnet,
)
from graph_mixup.mixup_generation.s_mixup.method.mixup import (
    mixup as smixup,
)
from graph_mixup.mixup_generation.g_mixup.graphons import (
    compute_class_graphons_and_features,
    sample_graph,
)
from graph_mixup.augmentations.typing import AugDatasetConfig
from graph_mixup.config.typing import PreBatchMixupName

from graph_mixup.ged_database.handlers.base_handler import BaseHandler
from graph_mixup.ged_database.handlers.mixup_graph_fetcher import (
    MixupGraphFetcher,
)

logger = logging.getLogger(__name__)


class InsufficientMixupItemsException(Exception): ...


class RDBDataModule(AbstractDataModule, Generic[DataModuleConfigType]):
    def __init__(
        self,
        config: DataModuleConfigType,
        method_name: PreBatchMixupName | None,
        inner_fold: int | None,
    ) -> None:
        # Setup parameters.
        self.dataset_name = config.dataset_name
        self.data_dir = config.data_dir
        self.random_state = config.random_state
        self.num_workers = config.num_workers
        self.batch_size = config.batch_size
        # Cross validation parameters.
        self.num_outer_folds = config.num_outer_folds
        self.fold = config.fold
        self.num_inner_folds = config.num_inner_folds
        self.inner_fold = inner_fold

        # Mixup (prev: method + dataset) parameters.
        self.method_config = config.dataset_config.method_config
        self.method_name = method_name
        self.use_vanilla = (
            config.dataset_config.use_vanilla
            if isinstance(config.dataset_config, AugDatasetConfig)
            else True
        )
        self.augmented_ratio = (
            config.dataset_config.augmented_ratio
            if isinstance(config.dataset_config, AugDatasetConfig)
            else 0.0
        )
        logger.info(
            f"use_vanilla={self.use_vanilla}, augmented_ratio={self.augmented_ratio}"
        )

        # ===
        # Increases every time the train loader is initialized. Used to sample
        # different mixup graphs every time by adding the current count to the
        # random_state.
        # ===
        self.reload_train_loader_count = 0

        # Assigned in setup.
        self.train_set: list[Data] | None = None
        self.vanilla_train_set: list[Data] | None = None
        self.all_mixup_graphs: list[Data] | None = None
        self.val_set: list[Data] | None = None
        self.test_set: list[Data] | None = None

        # Method-specific: G-Mixup.
        self.graphons_features: dict[int, dict[str, Tensor]] | None = None
        self.class_labels: list[int] | None = None

        # Method-specific: S-Mixup.
        self.gmnet: LitGraphMatchingNet | None = None

        # Initialize parent class.
        db_manager = BaseHandler()
        self.vanilla_graphs = db_manager.get_vanilla_graphs(self.dataset_name)
        dataset = db_manager.get_dataset(self.dataset_name)
        super().__init__(
            config,
            dataset.num_features,
            dataset.num_classes,
            len(self.vanilla_graphs),
            config.eval_mode,
        )

    @override
    def setup(self, stage: str) -> None:
        # ===
        # Outer Cross-Validation. Split vanilla graphs into training/validation
        # and test data.
        # ===

        dataset_idx = np.arange(len(self.vanilla_graphs))
        dataset_labels = (
            np.array(
                [graph.label for graph in self.vanilla_graphs]
            )  # [num_graphs, 1, num_classes]
            .squeeze(1)  # [num_graphs, num_classes]
            .argmax(1)  # [num_graphs, ]
        )

        skf_outer = StratifiedKFold(
            self.num_outer_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        outer_cv_splits = list(skf_outer.split(dataset_idx, dataset_labels))
        outer_cv_train_idx, outer_cv_test_idx = outer_cv_splits[self.fold]

        if stage == "fit":
            if not self.eval_mode:
                # ===
                # Inner Cross-Validation: Split training data into training and
                # validation.
                # ===

                skf_inner = StratifiedKFold(
                    self.num_inner_folds,
                    shuffle=True,
                    random_state=self.random_state,
                )
                # Split.
                inner_cv_splits = list(
                    skf_inner.split(
                        outer_cv_train_idx, dataset_labels[outer_cv_train_idx]
                    )
                )
                # Get relative indices.
                inner_train_rel_idx, inner_val_rel_idx = inner_cv_splits[
                    self.inner_fold
                ]
                # Map back to original dataset indices.
                inner_cv_train_idx = outer_cv_train_idx[inner_train_rel_idx]
                inner_cv_val_idx = outer_cv_train_idx[inner_val_rel_idx]

                self.log_cv_indices(
                    inner_cv_train_idx, f"inner_fold_{self.inner_fold}-train"
                )
                self.log_cv_indices(
                    inner_cv_val_idx, f"inner_fold_{self.inner_fold}-val"
                )

            else:
                # ===
                # If in eval_mode, use complete training data for training (no
                # validation data for early stopping required).
                # ===

                inner_cv_train_idx = np.array(outer_cv_train_idx)
                inner_cv_val_idx = np.array([])

                self.log_cv_indices(inner_cv_train_idx, "eval-train")
                self.log_cv_indices(inner_cv_val_idx, "eval-val")

            # ===
            # Create PyG graphs from database graphs and keep track of original
            # IDs.
            # ===
            self.train_set = [
                self.vanilla_graphs[i].get_pyg_data()
                for i in inner_cv_train_idx
            ]
            self.val_set = [
                self.vanilla_graphs[i].get_pyg_data() for i in inner_cv_val_idx
            ]

            if self.method_name is None:
                return

            elif self.method_name in [
                PreBatchMixupName.FGW_MIXUP,
                PreBatchMixupName.IF_MIXUP,
                PreBatchMixupName.IF_MIXUP_SE,
                PreBatchMixupName.GED_MIXUP,
                PreBatchMixupName.GEOMIX,
                PreBatchMixupName.SUBMIX,
            ]:
                # ===
                # Mixup methods that store graphs in the database.
                #
                # Fetch all mixup graphs that belong to parents contained in
                # the train set.
                # ===

                vanilla_train_set_graph_ids = [
                    self.vanilla_graphs[i].graph_id for i in inner_cv_train_idx
                ]

                db_mixup_graphs = MixupGraphFetcher(
                    self.dataset_name,
                    self.method_name,
                    self.method_config,
                    vanilla_train_set_graph_ids,
                    self.config.dataset_config.ged_filter_flags,
                ).fetch_mixup_graphs()

                # ===
                # Check if enough mixup graphs are available.
                # ===

                num_results = len(db_mixup_graphs)
                num_required = round(len(self.train_set) * self.augmented_ratio)
                if num_results < num_required:
                    raise InsufficientMixupItemsException(
                        f"available: {num_results}, required: {num_required}"
                    )
                logger.info(
                    f"Obtained mixup graphs. Required={num_required}, available={num_results}"
                )

                # ===
                # Convert to PyG graphs and store in state.
                # ===

                self.all_mixup_graphs = [
                    mixup_graph.get_pyg_data()
                    for mixup_graph in db_mixup_graphs
                ]

            elif self.method_name is PreBatchMixupName.G_MIXUP:
                # ===
                # G-Mixup: Mixup graphs are generated "on-the-fly" at train
                # loader initialization.
                #
                # Graphons are computed here and are then cached.
                # ===

                logger.info("G-Mixup: Compute graphons ...")
                self.graphons_features = compute_class_graphons_and_features(
                    self.train_set
                )
                self.class_labels = list(self.graphons_features.keys())
                logger.info("G-Mixup: Graphons computed.")

            elif self.method_name is PreBatchMixupName.S_MIXUP:
                # ===
                # S-Mixup: Mixup graphs are generated "on-the-fly" at train
                # loader initialization.
                #
                # Train GMNet and store in state.
                # ===
                logger.info("S-Mixup: Train GMNet ...")
                self.gmnet = train_gmnet(
                    train_set=self.train_set,
                    node_feat_dim=self.num_features,
                    training_config=self.config.dataset_config.lit_gmnet_training_config,
                    lit_gmnet_config=self.config.dataset_config.lit_gmnet_config,
                    num_workers=self.config.num_workers,
                    device=self.config.device,
                )
                logger.info("S-Mixup: GMNet training completed.")

            else:
                assert_never(self.method_name)

        elif stage == "test":
            self.log_cv_indices(outer_cv_test_idx, "test")
            self.test_set = [
                self.vanilla_graphs[i].get_pyg_data() for i in outer_cv_test_idx
            ]

    @override
    def train_dataloader(self) -> DataLoader:
        if self.method_name is None:
            return super().train_dataloader()

        # ===
        # All mixup methods: Mixup graphs are added to the training data.
        #
        # Before sampling new mixup graphs, reset train set to vanilla
        # graphs.
        # ===

        if self.vanilla_train_set is None:
            self.vanilla_train_set = self.train_set
        self.train_set = self.vanilla_train_set

        # ===
        # Sample mixup graphs: Compute a new random sample every time
        # the dataloader is reloaded.
        # ===

        num_mixup_graphs = round(len(self.train_set) * self.augmented_ratio)
        random.seed(self.random_state + self.reload_train_loader_count)
        self.reload_train_loader_count += 1

        if self.method_name in [
            PreBatchMixupName.FGW_MIXUP,
            PreBatchMixupName.IF_MIXUP,
            PreBatchMixupName.IF_MIXUP_SE,
            PreBatchMixupName.GED_MIXUP,
            PreBatchMixupName.GEOMIX,
            PreBatchMixupName.SUBMIX,
        ]:
            # ===
            # Mixup methods that store graphs in the database.
            #
            # Sample mixup graphs from all available ones (fetched in `setup`).
            # ===

            sampled_mixup_graphs = random.sample(
                self.all_mixup_graphs,
                num_mixup_graphs,
            )
            logger.debug(
                f"Sampled {len(sampled_mixup_graphs)} mixup graphs, iteration={self.reload_train_loader_count}"
            )

        elif self.method_name is PreBatchMixupName.G_MIXUP:
            # ===
            # G-Mixup: Mixup graphs are generated "on-the-fly" below.
            # ===

            sampled_mixup_graphs: list[Data] = []
            for _ in range(num_mixup_graphs):
                class_indices = random.sample(self.class_labels, 2)
                mixup_graph = sample_graph(
                    (
                        self.graphons_features[class_indices[0]],
                        self.graphons_features[class_indices[1]],
                    ),
                    self.method_config,
                )
                sampled_mixup_graphs.append(mixup_graph)
            logger.debug(
                f"Created {len(sampled_mixup_graphs)} graphs with G-Mixup."
            )

        elif self.method_name is PreBatchMixupName.S_MIXUP:
            # ===
            # S-Mixup: Mixup graphs are generated "on-the-fly" below.
            # ===

            sampled_mixup_graphs: list[Data] = []
            while len(sampled_mixup_graphs) < num_mixup_graphs:
                g1, g2 = random.sample(self.train_set, 2)
                try:
                    mixup_graph = smixup(
                        g1,
                        g2,
                        graph_matching_network=self.gmnet,
                        config=self.method_config,
                        device=self.config.device,
                    )
                    sampled_mixup_graphs.append(mixup_graph)
                except RuntimeError as e:
                    logger.warning(f"S-Mixup Error: {e}")
            logger.debug(
                f"Created {len(sampled_mixup_graphs)} graphs with S-Mixup."
            )

        else:
            assert_never(self.method_name)

        # ===
        # All mixup methods: Combine vanilla and mixup graphs (if requested).
        # ===

        self.train_set = (
            self.train_set + sampled_mixup_graphs
            if self.use_vanilla
            else sampled_mixup_graphs
        )

        # ===
        # GeoMix, If-Mixup, & S-Mixup: Add edge weights to vanilla items.
        # ===
        if self.method_name in [
            PreBatchMixupName.IF_MIXUP,
            PreBatchMixupName.S_MIXUP,
            PreBatchMixupName.GEOMIX,
        ]:
            for graph in self.train_set:
                if graph.edge_weight is None:
                    graph.edge_weight = torch.ones(
                        graph.num_edges, dtype=torch.float
                    )

        return super().train_dataloader()
