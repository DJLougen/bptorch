"""Unit tests verifying all training flow, data, optimization, scheduler, metric, and checkpoint primitives."""

from neural_blueprint.registry.registry import global_registry


def test_training_flow_primitives_registered():
    flow_node_types = [
        "builtin.training_sequence@1",
        "builtin.epoch_loop@1",
        "builtin.batch_loop@1",
        "builtin.validation_branch@1",
        "builtin.early_stopping_gate@1",
        "builtin.branch@1",
        "builtin.do_once@1",
        "builtin.while_loop@1",
        "builtin.for_loop@1",
        "builtin.event_on_train_begin@1",
        "builtin.event_on_epoch_start@1",
        "builtin.event_on_batch_end@1",
        "builtin.event_on_validation@1",
        "builtin.event_on_checkpoint@1",
        "builtin.event_on_anomaly@1",
        "builtin.get_variable@1",
        "builtin.set_variable@1",
    ]

    for type_id in flow_node_types:
        node_def = global_registry.get(type_id)
        assert node_def is not None, f"Node type {type_id} is not registered"
        in_ports = node_def.input_ports({})
        out_ports = node_def.output_ports({})
        assert isinstance(in_ports, list)
        assert isinstance(out_ports, list)


def test_data_pipeline_primitives_registered():
    data_node_types = [
        "builtin.dataset_source@1",
        "builtin.bpe_tokenizer@1",
        "builtin.batch_sampler@1",
        "builtin.dataloader@1",
        "builtin.data_augmentation@1",
    ]

    for type_id in data_node_types:
        node_def = global_registry.get(type_id)
        assert node_def is not None, f"Node type {type_id} is not registered"


def test_optimization_primitives_registered():
    opt_node_types = [
        "builtin.adamw_optimizer@1",
        "builtin.sgd_optimizer@1",
        "builtin.lion_optimizer@1",
        "builtin.clip_gradients@1",
        "builtin.zero_grad@1",
        "builtin.optimizer_step@1",
        "builtin.backward@1",
        "builtin.autocast_scope@1",
        "builtin.grad_scaler@1",
    ]

    for type_id in opt_node_types:
        node_def = global_registry.get(type_id)
        assert node_def is not None, f"Node type {type_id} is not registered"


def test_scheduler_and_metric_primitives_registered():
    sched_and_metric_types = [
        "builtin.cosine_annealing_lr@1",
        "builtin.linear_warmup_scheduler@1",
        "builtin.reduce_lr_on_plateau@1",
        "builtin.step_lr@1",
        "builtin.loss_aggregator@1",
        "builtin.accuracy_metric@1",
        "builtin.perplexity_metric@1",
        "builtin.metric_logger@1",
        "builtin.validation_loop@1",
        "builtin.save_checkpoint@1",
        "builtin.load_checkpoint@1",
        "builtin.export_model@1",
    ]

    for type_id in sched_and_metric_types:
        node_def = global_registry.get(type_id)
        assert node_def is not None, f"Node type {type_id} is not registered"


def test_exec_ports_identification():
    seq_def = global_registry.require("builtin.training_sequence@1")
    in_ports = seq_def.input_ports({"branch_count": 4})
    out_ports = seq_def.output_ports({"branch_count": 4})

    assert len(in_ports) == 1
    assert in_ports[0].kind == "exec"
    assert len(out_ports) == 4
    assert all(p.kind == "exec" for p in out_ports)
