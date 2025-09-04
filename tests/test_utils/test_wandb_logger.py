"""Test wandb logger integration."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import hydra
from omegaconf import DictConfig

from sheeprl.utils.logger import get_logger


class TestWandbLogger:
    """Test wandb logger functionality."""

    def test_wandb_logger_config_loading(self):
        """Test that wandb logger configuration can be loaded."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create a temporary config file
            config_path = os.path.join(tmp_dir, "test_config.yaml")
            with open(config_path, "w") as f:
                f.write("""
defaults:
  - _self_
  - /logger@logger: wandb

log_every: 1000
disable_timer: False
log_level: 1
sync_on_compute: False

aggregator:
  _target_: sheeprl.utils.metric.MetricAggregator
  raise_on_missing: False
  metrics:
    Rewards/rew_avg: 
      _target_: torchmetrics.MeanMetric
      sync_on_compute: ${metric.sync_on_compute}
""")
            
            # Initialize Hydra
            with hydra.initialize(config_path=tmp_dir):
                cfg = hydra.compose(config_name="test_config")
                
                # Verify that the logger target is correct
                assert "wandb" in cfg.metric.logger._target_.lower()
                assert cfg.metric.logger._target_ == "lightning.pytorch.loggers.WandbLogger"

    @patch('sheeprl.utils.logger.hydra.utils.instantiate')
    def test_wandb_logger_instantiation(self, mock_instantiate):
        """Test that wandb logger is instantiated correctly."""
        # Mock fabric
        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_fabric.world_size = 1
        
        # Mock config
        mock_cfg = {
            'metric': {
                'log_level': 1,
                'logger': {
                    '_target_': 'lightning.pytorch.loggers.WandbLogger',
                    'project': 'test_project',
                    'name': 'test_run',
                    'save_dir': 'logs/runs/test'
                }
            },
            'exp_name': 'test_project',
            'run_name': 'test_run',
            'root_dir': 'test'
        }
        
        # Mock the instantiate function to return a mock logger
        mock_logger = MagicMock()
        mock_instantiate.return_value = mock_logger
        
        # Call get_logger
        result = get_logger(mock_fabric, mock_cfg)
        
        # Verify that instantiate was called
        mock_instantiate.assert_called_once()
        
        # Verify that the result is the mock logger
        assert result == mock_logger

    def test_wandb_logger_config_validation(self):
        """Test that wandb logger configuration is validated correctly."""
        # Mock fabric
        mock_fabric = MagicMock()
        mock_fabric.is_global_zero = True
        mock_fabric.world_size = 1
        
        # Mock config with wandb logger
        mock_cfg = {
            'metric': {
                'log_level': 1,
                'logger': {
                    '_target_': 'lightning.pytorch.loggers.WandbLogger',
                    'project': 'different_project',  # This should be overridden
                    'name': 'different_name',  # This should be overridden
                    'save_dir': 'different_dir'  # This should be overridden
                }
            },
            'exp_name': 'test_project',
            'run_name': 'test_run',
            'root_dir': 'test'
        }
        
        with patch('sheeprl.utils.logger.hydra.utils.instantiate') as mock_instantiate:
            mock_instantiate.return_value = MagicMock()
            
            # Call get_logger
            get_logger(mock_fabric, mock_cfg)
            
            # Verify that the config was updated correctly
            assert mock_cfg['metric']['logger']['project'] == 'test_project'
            assert mock_cfg['metric']['logger']['name'] == 'test_run'
            assert mock_cfg['metric']['logger']['save_dir'] == 'logs/runs/test'
