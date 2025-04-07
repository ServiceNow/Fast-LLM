from unittest import mock

import pytest
import torch

from fast_llm.engine.base_model.base_model import LossDef
from fast_llm.engine.distributed.config import DistributedConfig, PhaseType
from fast_llm.engine.distributed.distributed import Distributed
from fast_llm.engine.multi_stage.multi_stage import MultiStageModel
from fast_llm.engine.schedule.config import ScheduleConfig
from fast_llm.engine.schedule.runner import BatchContext, ScheduleRunner
from fast_llm.engine.schedule.schedule import Schedule
from fast_llm.layers.transformer.config import TransformerRoutingMetrics


@pytest.fixture
def setup_runner():
    """
    Fixture to set up the test environment.
    TODO: Leave it here for now, but may be moved to common.py
    """
    # Mock objects needed for testing
    distributed_config = DistributedConfig()

    # Mock MultiStageModel with loss_defs
    multi_stage = mock.MagicMock(spec=MultiStageModel)
    multi_stage.base_model.loss_defs = [LossDef(name="test_loss", formatted_name="Test Loss", count=1)]
    multi_stage.base_model.metric_defs = [
        LossDef(
            name=TransformerRoutingMetrics.normalized_average_entropy, formatted_name="Normalized Entropy", count=1
        ),
        LossDef(name=TransformerRoutingMetrics.mutual_info, formatted_name="Mutual Information", count=1),
    ]

    # Create a schedule runner
    schedule_config = ScheduleConfig()
    runner = ScheduleRunner(config=schedule_config, multi_stage=multi_stage, distributed_config=distributed_config)

    # Mock distributed object
    distributed = mock.MagicMock(spec=Distributed)
    distributed.config = distributed_config
    distributed.device = torch.device("cpu")
    distributed.data_group = None
    distributed.pipeline_group = None

    # Setup the runner
    runner._distributed = distributed
    runner.is_initialized = True

    # Create a mock schedule
    schedule = mock.MagicMock(spec=Schedule)
    schedule.phase = PhaseType.training
    schedule.batch_config.num_inputs = 3
    schedule._schedule_config = schedule_config

    # Create a batch context with metrics and losses
    context = BatchContext(
        iteration=1,
        schedule=schedule,
    )

    return runner, context, schedule


def test_reduce_losses(setup_runner):
    """Test that _reduce_losses correctly reduces losses"""
    runner, context, _ = setup_runner

    # Add test metrics
    context.metrics = {
        # Metrics that should be reduced (in TransformerReducedMetrics)
        TransformerRoutingMetrics.normalized_average_entropy: [
            torch.tensor(0.5),
            torch.tensor(0.6),
            torch.tensor(0.7),
        ],
        TransformerRoutingMetrics.mutual_info: [torch.tensor(0.2), torch.tensor(0.3), torch.tensor(0.4)],
        # Metric that should not be reduced as its not registered in metric_defs of the base model
        "non_reduced_metric": [torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0)],
    }

    # Add test losses
    context.losses = {"test_loss": [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]}

    reduced_losses = runner._reduce_losses(context)
    reduced_metrics = runner._reduce_metrics(context)

    assert "test_loss" in reduced_losses
    assert pytest.approx(reduced_losses["test_loss"], 0.001) == 2.0
    assert "non_reduced_metric" in reduced_metrics
    assert pytest.approx(reduced_metrics["normalized_average_entropy"], 0.01) == 0.6
    assert pytest.approx(reduced_metrics["mutual_info"], 0.01) == 0.3
    assert isinstance(reduced_metrics["non_reduced_metric"], list)
