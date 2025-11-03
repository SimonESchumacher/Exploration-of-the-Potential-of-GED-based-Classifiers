from graph_mixup.studies.objectives import sanity_check_params


def test_pre_loss_method_should_fail():
    assert not sanity_check_params(
        dataset_len=1113,
        num_folds=10,
        val_size=0.1,
        batch_size=16,
        use_vanilla=False,
        augmented_ratio=0.12,
        uses_batch_norm=True,
        pre_loss_method=True,
    )


def test_pre_loss_method_should_succeed():
    assert sanity_check_params(
        dataset_len=1113,
        num_folds=10,
        val_size=0.1,
        batch_size=32,
        use_vanilla=False,
        augmented_ratio=0.12,
        uses_batch_norm=True,
        pre_loss_method=True,
    )


def test_pre_collate_method_should_fail():
    assert not sanity_check_params(
        dataset_len=1113,
        num_folds=10,
        val_size=0.1,
        batch_size=128,
        use_vanilla=False,
        augmented_ratio=0.12,
        uses_batch_norm=True,
        pre_loss_method=False,
    )


def test_pre_collate_method_should_succeed():
    assert sanity_check_params(
        dataset_len=1113,
        num_folds=10,
        val_size=0.1,
        batch_size=128,
        use_vanilla=False,
        augmented_ratio=0.25,
        uses_batch_norm=True,
        pre_loss_method=False,
    )
