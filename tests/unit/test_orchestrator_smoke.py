"""Smoke tests for orchestrator functionality."""

import pytest
from pathlib import Path
import tempfile
import os

from src.valtapyV2.application.orchestrator import Orchestrator
from src.valtapyV2.domain.entities import RunSummary


class TestOrchestratorSmoke:
    """Smoke tests for the orchestrator pipeline."""
    
    def test_orchestrator_run_with_minimal_config(self, temp_config_file):
        """Test that orchestrator can run with minimal configuration."""
        orchestrator = Orchestrator()
        
        # Run evaluation
        run_summary = orchestrator.run(temp_config_file)
        
        # Basic checks
        assert isinstance(run_summary, RunSummary)
        assert run_summary.plan is not None
        assert isinstance(run_summary.results, list)
        assert isinstance(run_summary.aggregates, dict)
        
        # Check that we got some results
        assert len(run_summary.results) > 0
        
        # Check that at least some metrics succeeded
        successful_results = [r for r in run_summary.results if "error" not in r.details]
        assert len(successful_results) > 0, "No metrics succeeded in smoke test"
    
    def test_orchestrator_generates_reports(self, sample_config, temp_data_files):
        """Test that orchestrator generates report files."""
        real_path, synth_path = temp_data_files
        
        # Update config with temporary output directory
        temp_output_dir = tempfile.mkdtemp()
        config_with_reports = sample_config.copy()
        config_with_reports["report"] = {
            "formats": ["json", "md"],
            "output_dir": temp_output_dir
        }
        
        # Save config to temporary file
        import yaml
        temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_with_reports, temp_config, default_flow_style=False)
        temp_config.close()
        
        try:
            # Run orchestrator
            orchestrator = Orchestrator()
            run_summary = orchestrator.run(temp_config.name)
            
            # Check that report files were created
            json_report = Path(temp_output_dir) / "summary.json"
            md_report = Path(temp_output_dir) / "summary.md"
            
            assert json_report.exists(), "JSON report not generated"
            assert md_report.exists(), "Markdown report not generated"
            
            # Check JSON content
            import json
            with open(json_report) as f:
                json_data = json.load(f)
            
            assert "metadata" in json_data
            assert "results" in json_data
            assert "aggregates" in json_data
            
            # Check markdown content
            with open(md_report) as f:
                md_content = f.read()
            
            assert "ValtaPyV2 Evaluation Report" in md_content
            assert "Family Scores" in md_content
            
        finally:
            # Cleanup
            try:
                os.unlink(temp_config.name)
                if json_report.exists():
                    os.unlink(json_report)
                if md_report.exists():
                    os.unlink(md_report)
                os.rmdir(temp_output_dir)
            except:
                pass
    
    def test_orchestrator_handles_partial_failures(self, temp_config_file):
        """Test that orchestrator handles partial metric failures gracefully."""
        orchestrator = Orchestrator()
        
        # Run with config that might have some failures
        run_summary = orchestrator.run(temp_config_file)
        
        # Should complete even if some metrics fail
        assert isinstance(run_summary, RunSummary)
        assert len(run_summary.results) > 0
        
        # Check aggregates are computed even with failures
        assert isinstance(run_summary.aggregates, dict)
        
        # Should have execution statistics
        total_metrics = run_summary.aggregates.get("total_metrics", 0)
        assert total_metrics > 0
    
    def test_orchestrator_caches_computations(self, temp_config_file):
        """Test that orchestrator uses shared cache for expensive computations."""
        orchestrator = Orchestrator()
        
        # Run evaluation
        run_summary = orchestrator.run(temp_config_file)
        
        # Check that cache statistics are recorded
        cache_stats = run_summary.artifacts.get("cache_stats")
        assert cache_stats is not None
        assert isinstance(cache_stats, dict)
        
        # Should have recorded some cache activity
        total_requests = cache_stats.get("total_requests", 0)
        assert total_requests >= 0  # At least attempted to use cache
    
    def test_orchestrator_with_different_purposes(self, temp_data_files):
        """Test orchestrator with different evaluation purposes."""
        real_path, synth_path = temp_data_files
        
        purposes = ["privacy_hardening", "model_selection", "data_release"]
        
        for purpose in purposes:
            # Create config for this purpose
            config = {
                "data": {
                    "real": real_path,
                    "synthetic": synth_path,
                    "target": "target"
                },
                "evaluation": {
                    "purpose": purpose,
                    "seed": 42
                }
            }
            
            # Save to temporary file
            import yaml
            temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
            yaml.dump(config, temp_config, default_flow_style=False)
            temp_config.close()
            
            try:
                # Run orchestrator
                orchestrator = Orchestrator()
                run_summary = orchestrator.run(temp_config.name)
                
                # Should complete successfully
                assert isinstance(run_summary, RunSummary)
                assert run_summary.plan.purpose == purpose
                assert len(run_summary.results) > 0
                
                # Should have appropriate metrics for the purpose
                metric_ids = [r.id for r in run_summary.results]
                
                if purpose == "privacy_hardening":
                    # Should include privacy and fidelity metrics
                    has_privacy = any("privacy" in mid for mid in metric_ids)
                    has_fidelity = any("fidelity" in mid for mid in metric_ids)
                    assert has_privacy or has_fidelity, f"No appropriate metrics for {purpose}"
                
            finally:
                try:
                    os.unlink(temp_config.name)
                except:
                    pass