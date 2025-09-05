"""Command-line interface for ValtaPyV2."""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

from ..application.pipeline import run_from_config
from ..infrastructure.runtime.config import validate_file_paths, get_default_config, save_config
from ..infrastructure.runtime.logging import setup_logging
from ..domain.errors import ConfigError, SchemaError, MetricExecutionError


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="valtapyv2",
        description="ValtaPyV2: Modular framework for synthetic tabular data evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evaluate --config examples/config_minimal.yaml
  %(prog)s evaluate --config my_config.yaml --verbose
  %(prog)s init-config --output my_config.yaml
        """
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log output to file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Run synthetic data evaluation",
        description="Evaluate synthetic data against real data using configured metrics"
    )
    eval_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    eval_parser.add_argument(
        "--check-paths",
        action="store_true",
        help="Validate file paths before running evaluation"
    )
    
    # Init config command
    init_parser = subparsers.add_parser(
        "init-config",
        help="Generate default configuration file",
        description="Create a template configuration file with default settings"
    )
    init_parser.add_argument(
        "--output", "-o",
        type=str,
        default="valtapy_config.yaml",
        help="Output file path for configuration (default: valtapy_config.yaml)"
    )
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing configuration file"
    )
    
    # List metrics command
    list_parser = subparsers.add_parser(
        "list-metrics",
        help="List available metrics",
        description="Display all available evaluation metrics grouped by family"
    )
    list_parser.add_argument(
        "--family", "-f",
        choices=["fidelity", "utility", "privacy"],
        help="Filter metrics by family"
    )
    
    return parser


def handle_evaluate_command(args) -> int:
    """Handle the evaluate subcommand."""
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Configuration file not found: {args.config}")
            return 1
        
        # Validate file paths if requested
        if args.check_paths:
            from ..infrastructure.runtime.config import load_config
            try:
                config = load_config(args.config)
                warnings = validate_file_paths(config)
                if warnings:
                    print("File path warnings:")
                    for warning in warnings:
                        print(f"  ⚠️  {warning}")
                    
                    response = input("Continue anyway? [y/N]: ")
                    if response.lower() not in ['y', 'yes']:
                        print("Evaluation cancelled.")
                        return 1
            except ConfigError as e:
                print(f"Configuration error: {e}")
                return 1
        
        # Run evaluation
        print(f"Starting evaluation with config: {args.config}")
        run_summary = run_from_config(args.config)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION COMPLETED")
        print("="*60)
        print(f"Total metrics: {len(run_summary.results)}")
        print(f"Successful: {len([r for r in run_summary.results if 'error' not in r.details])}")
        print(f"Failed: {len([r for r in run_summary.results if 'error' in r.details])}")
        
        if "composite_score" in run_summary.aggregates:
            print(f"Composite score: {run_summary.aggregates['composite_score']:.3f}")
        
        for family in ["fidelity", "utility", "privacy"]:
            if f"{family}_score" in run_summary.aggregates:
                score = run_summary.aggregates[f"{family}_score"]
                print(f"{family.title()} score: {score:.3f}")
        
        return 0
        
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 1
    except SchemaError as e:
        print(f"Data schema error: {e}")
        return 1
    except MetricExecutionError as e:
        print(f"Metric execution error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def handle_init_config_command(args) -> int:
    """Handle the init-config subcommand."""
    try:
        output_path = Path(args.output)
        
        # Check if file exists
        if output_path.exists() and not args.overwrite:
            print(f"Error: Configuration file already exists: {args.output}")
            print("Use --overwrite to replace existing file")
            return 1
        
        # Generate default configuration
        default_config = get_default_config()
        save_config(default_config, args.output)
        
        print(f"✅ Default configuration saved to: {args.output}")
        print("\nNext steps:")
        print(f"1. Edit {args.output} to specify your data paths and preferences")
        print(f"2. Run: valtapyv2 evaluate --config {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error creating configuration file: {e}")
        return 1


def handle_list_metrics_command(args) -> int:
    """Handle the list-metrics subcommand."""
    try:
        from ..infrastructure.metrics.registry import get_metric_registry
        
        registry = get_metric_registry()
        
        if args.family:
            # List metrics for specific family
            metric_ids = registry.list_ids(family=args.family)
            print(f"\n{args.family.title()} Metrics:")
            print("-" * 40)
            for metric_id in sorted(metric_ids):
                print(f"  • {metric_id}")
        else:
            # List all metrics grouped by family
            families = registry.list_by_family()
            
            print("\nAvailable Metrics:")
            print("=" * 50)
            
            for family in sorted(families.keys()):
                print(f"\n{family.title()}:")
                print("-" * 20)
                for metric_id in sorted(families[family]):
                    print(f"  • {metric_id}")
        
        print(f"\nTotal metrics: {len(registry.list_ids())}")
        return 0
        
    except Exception as e:
        print(f"Error listing metrics: {e}")
        return 1


def main(argv: Optional[list] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)
    
    # Handle commands
    if args.command == "evaluate":
        return handle_evaluate_command(args)
    elif args.command == "init-config":
        return handle_init_config_command(args)
    elif args.command == "list-metrics":
        return handle_list_metrics_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())