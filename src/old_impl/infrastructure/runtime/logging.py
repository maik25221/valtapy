"""Logging configuration for the ValtaPyV2 framework."""

import logging
import sys
from typing import Optional
from pathlib import Path


def setup_logging(level: str = "INFO", 
                 log_file: Optional[str] = None,
                 format_string: Optional[str] = None) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        format_string: Custom format string for log messages
    """
    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        handlers=_create_handlers(log_file)
    )
    
    # Set specific logger levels to reduce noise
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("multiprocessing").setLevel(logging.WARNING)


def _create_handlers(log_file: Optional[str]) -> list:
    """Create logging handlers for console and optional file output."""
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    handlers.append(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    
    return handlers


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


class MetricLogger:
    """Specialized logger for metric execution tracking."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"metrics.{name}")
        self._start_times = {}
    
    def start_metric(self, metric_id: str) -> None:
        """Log start of metric computation."""
        import time
        self._start_times[metric_id] = time.time()
        self.logger.info(f"Starting computation: {metric_id}")
    
    def finish_metric(self, metric_id: str, success: bool = True, error: Optional[str] = None) -> None:
        """Log completion of metric computation."""
        import time
        
        duration = None
        if metric_id in self._start_times:
            duration = time.time() - self._start_times[metric_id]
            del self._start_times[metric_id]
        
        if success:
            msg = f"Completed computation: {metric_id}"
            if duration:
                msg += f" in {duration:.3f}s"
            self.logger.info(msg)
        else:
            msg = f"Failed computation: {metric_id}"
            if duration:
                msg += f" after {duration:.3f}s"
            if error:
                msg += f" - {error}"
            self.logger.error(msg)
    
    def log_metric_value(self, metric_id: str, value: float, details: dict = None) -> None:
        """Log metric computation result."""
        msg = f"Metric result: {metric_id} = {value}"
        if details:
            msg += f" (details: {details})"
        self.logger.debug(msg)


class ProgressLogger:
    """Logger for tracking progress of long-running operations."""
    
    def __init__(self, name: str, total_items: int):
        self.logger = logging.getLogger(f"progress.{name}")
        self.total_items = total_items
        self.completed_items = 0
        self._last_percent = -1
    
    def update(self, increment: int = 1) -> None:
        """Update progress counter."""
        self.completed_items += increment
        percent = int((self.completed_items / self.total_items) * 100)
        
        # Log every 10% or at completion
        if percent >= self._last_percent + 10 or self.completed_items == self.total_items:
            self.logger.info(f"Progress: {self.completed_items}/{self.total_items} ({percent}%)")
            self._last_percent = percent
    
    def finish(self) -> None:
        """Mark progress as completed."""
        self.logger.info(f"Completed: {self.completed_items}/{self.total_items}")


# Configure default logging on import
setup_logging()