"""Smoke tests for CLI functionality."""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
import yaml

from src.valtapyV2.interface.cli import main


class TestCLISmoke:
    """Smoke tests for command-line interface."""
    
    def test_cli_help_command(self):
        """Test that CLI shows help without errors."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        
        # argparse exits with code 0 for help
        assert exc_info.value.code == 0
    
    def test_cli_list_metrics_command(self, capsys):
        """Test list-metrics command."""
        result = main(["list-metrics"])
        
        # Should complete successfully
        assert result == 0
        
        # Should output metric information
        captured = capsys.readouterr()
        assert "Available Metrics" in captured.out
        assert "fidelity" in captured.out.lower()
        assert "utility" in captured.out.lower()
        assert "privacy" in captured.out.lower()
    
    def test_cli_list_metrics_with_family_filter(self, capsys):
        """Test list-metrics command with family filter."""
        result = main(["list-metrics", "--family", "fidelity"])
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Fidelity Metrics" in captured.out
        assert "fidelity.ks" in captured.out
    
    def test_cli_init_config_command(self):
        """Test init-config command."""
        # Create temporary output file
        temp_config = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        temp_config.close()
        
        try:
            # Remove the file so init-config can create it
            os.unlink(temp_config.name)
            
            # Run init-config
            result = main(["init-config", "--output", temp_config.name])
            
            assert result == 0
            
            # Check that config file was created
            assert Path(temp_config.name).exists()
            
            # Check that it's valid YAML
            with open(temp_config.name) as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict)
            assert "data" in config
            assert "evaluation" in config
            
        finally:
            try:
                os.unlink(temp_config.name)
            except:
                pass
    
    def test_cli_init_config_no_overwrite(self):
        """Test init-config refuses to overwrite existing files."""
        temp_config = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        temp_config.write(b"existing content")
        temp_config.close()
        
        try:
            # Run init-config without --overwrite
            result = main(["init-config", "--output", temp_config.name])
            
            # Should fail
            assert result == 1
            
            # File content should be unchanged
            with open(temp_config.name, 'rb') as f:
                content = f.read()
            assert content == b"existing content"
            
        finally:
            try:
                os.unlink(temp_config.name)
            except:
                pass
    
    def test_cli_init_config_with_overwrite(self):
        """Test init-config with --overwrite flag."""
        temp_config = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        temp_config.write(b"existing content")
        temp_config.close()
        
        try:
            # Run init-config with --overwrite
            result = main(["init-config", "--output", temp_config.name, "--overwrite"])
            
            assert result == 0
            
            # File should be overwritten with YAML content
            with open(temp_config.name) as f:
                config = yaml.safe_load(f)
            
            assert isinstance(config, dict)
            assert "data" in config
            
        finally:
            try:
                os.unlink(temp_config.name)
            except:
                pass
    
    def test_cli_evaluate_missing_config(self):
        """Test evaluate command with missing config file."""
        result = main(["evaluate", "--config", "nonexistent.yaml"])
        
        # Should fail gracefully
        assert result == 1
    
    def test_cli_evaluate_with_valid_config(self, temp_config_file, capsys):
        """Test evaluate command with valid configuration."""
        result = main(["evaluate", "--config", temp_config_file])
        
        # Should complete (may have some failures but shouldn't crash)
        assert result == 0
        
        # Should output evaluation results
        captured = capsys.readouterr()
        assert "EVALUATION COMPLETED" in captured.out
        assert "Total metrics:" in captured.out
    
    def test_cli_evaluate_with_path_checking(self, sample_config, capsys):
        """Test evaluate command with path validation."""
        # Create config with non-existent files
        bad_config = sample_config.copy()
        bad_config["data"]["real"] = "nonexistent_real.csv"
        bad_config["data"]["synthetic"] = "nonexistent_synth.csv"
        
        # Save to temp file
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(bad_config, temp_config, default_flow_style=False)
        temp_config.close()
        
        try:
            # Mock user input to cancel
            with patch('builtins.input', return_value='n'):
                result = main(["evaluate", "--config", temp_config.name, "--check-paths"])
            
            # Should be cancelled by user
            assert result == 1
            
            captured = capsys.readouterr()
            assert "File path warnings:" in captured.out
            assert "Evaluation cancelled" in captured.out
            
        finally:
            try:
                os.unlink(temp_config.name)
            except:
                pass
    
    def test_cli_verbose_mode(self, temp_config_file):
        """Test CLI with verbose logging."""
        # This mainly tests that verbose mode doesn't crash
        result = main(["--verbose", "evaluate", "--config", temp_config_file])
        
        # Should complete without crashing
        assert result == 0
    
    def test_cli_with_log_file(self, temp_config_file):
        """Test CLI with log file output."""
        temp_log = tempfile.NamedTemporaryFile(suffix='.log', delete=False)
        temp_log.close()
        
        try:
            result = main([
                "--log-file", temp_log.name, 
                "evaluate", "--config", temp_config_file
            ])
            
            assert result == 0
            
            # Check that log file was created and has content
            assert Path(temp_log.name).exists()
            
            with open(temp_log.name) as f:
                log_content = f.read()
            
            # Should have some log entries
            assert len(log_content.strip()) > 0
            
        finally:
            try:
                os.unlink(temp_log.name)
            except:
                pass
    
    def test_cli_no_command_shows_help(self, capsys):
        """Test that running CLI without command shows help."""
        result = main([])
        
        assert result == 1
        
        captured = capsys.readouterr()
        assert "usage:" in captured.out.lower()
        assert "available commands" in captured.out.lower()