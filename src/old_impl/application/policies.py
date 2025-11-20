"""Policy management for privacy budgets and resource constraints."""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PrivacyBudget:
    """Privacy budget management for differential privacy metrics."""
    
    total_epsilon: float
    total_delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    
    def can_spend(self, epsilon: float, delta: float) -> bool:
        """Check if privacy budget allows spending the requested amounts."""
        # TODO: Implement proper budget tracking
        return (self.consumed_epsilon + epsilon <= self.total_epsilon and
                self.consumed_delta + delta <= self.total_delta)
    
    def spend(self, epsilon: float, delta: float) -> bool:
        """
        Spend privacy budget if available.
        
        Returns:
            True if budget was successfully spent, False otherwise
        """
        if self.can_spend(epsilon, delta):
            self.consumed_epsilon += epsilon
            self.consumed_delta += delta
            return True
        return False
    
    def remaining_budget(self) -> tuple[float, float]:
        """Get remaining privacy budget."""
        return (self.total_epsilon - self.consumed_epsilon,
                self.total_delta - self.consumed_delta)


@dataclass 
class TimeBudget:
    """Time budget management for limiting evaluation duration."""
    
    max_total_seconds: float
    max_per_metric_seconds: float
    consumed_seconds: float = 0.0
    
    def can_run_metric(self, estimated_seconds: float) -> bool:
        """Check if time budget allows running a metric."""
        # TODO: Implement proper time tracking
        return (self.consumed_seconds + estimated_seconds <= self.max_total_seconds and
                estimated_seconds <= self.max_per_metric_seconds)
    
    def record_metric_time(self, actual_seconds: float) -> None:
        """Record actual time spent on a metric."""
        self.consumed_seconds += actual_seconds
    
    def remaining_time(self) -> float:
        """Get remaining time budget in seconds."""
        return max(0.0, self.max_total_seconds - self.consumed_seconds)


class PolicyManager:
    """
    Manages evaluation policies for privacy budgets and resource constraints.
    
    TODO: Implement the following features:
    - Privacy budget allocation across metrics
    - Time budget enforcement with early termination
    - Resource usage monitoring and limits
    - Policy violation handling and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.privacy_budget = self._init_privacy_budget()
        self.time_budget = self._init_time_budget()
    
    def _init_privacy_budget(self) -> Optional[PrivacyBudget]:
        """Initialize privacy budget from configuration."""
        privacy_config = self.config.get("privacy_budget")
        if not privacy_config:
            return None
        
        return PrivacyBudget(
            total_epsilon=privacy_config.get("epsilon", 1.0),
            total_delta=privacy_config.get("delta", 1e-5)
        )
    
    def _init_time_budget(self) -> Optional[TimeBudget]:
        """Initialize time budget from configuration."""
        time_config = self.config.get("time_budget")
        if not time_config:
            return None
        
        return TimeBudget(
            max_total_seconds=time_config.get("max_total_seconds", 3600),  # 1 hour default
            max_per_metric_seconds=time_config.get("max_per_metric_seconds", 300)  # 5 min default
        )
    
    def check_privacy_budget(self, metric_id: str, epsilon: float, delta: float) -> bool:
        """
        Check if privacy budget allows running the specified metric.
        
        Args:
            metric_id: Identifier of the metric requesting budget
            epsilon: Epsilon privacy parameter requested
            delta: Delta privacy parameter requested
            
        Returns:
            True if budget allows the request, False otherwise
        """
        if not self.privacy_budget:
            return True  # No budget constraints
        
        return self.privacy_budget.can_spend(epsilon, delta)
    
    def allocate_privacy_budget(self, metric_id: str, epsilon: float, delta: float) -> bool:
        """
        Allocate privacy budget to a metric.
        
        Args:
            metric_id: Identifier of the metric
            epsilon: Epsilon privacy parameter to allocate  
            delta: Delta privacy parameter to allocate
            
        Returns:
            True if budget was successfully allocated, False otherwise
        """
        if not self.privacy_budget:
            return True  # No budget constraints
        
        return self.privacy_budget.spend(epsilon, delta)
    
    def check_time_budget(self, metric_id: str, estimated_seconds: float) -> bool:
        """
        Check if time budget allows running the specified metric.
        
        Args:
            metric_id: Identifier of the metric
            estimated_seconds: Estimated execution time
            
        Returns:
            True if time budget allows the request, False otherwise
        """
        if not self.time_budget:
            return True  # No time constraints
        
        return self.time_budget.can_run_metric(estimated_seconds)
    
    def record_metric_execution_time(self, metric_id: str, actual_seconds: float) -> None:
        """
        Record actual execution time for a metric.
        
        Args:
            metric_id: Identifier of the metric
            actual_seconds: Actual execution time in seconds
        """
        if self.time_budget:
            self.time_budget.record_metric_time(actual_seconds)
    
    def get_policy_status(self) -> Dict[str, Any]:
        """
        Get current status of all managed policies.
        
        Returns:
            Dictionary containing policy status information
        """
        status = {}
        
        if self.privacy_budget:
            remaining_eps, remaining_delta = self.privacy_budget.remaining_budget()
            status["privacy_budget"] = {
                "remaining_epsilon": remaining_eps,
                "remaining_delta": remaining_delta,
                "consumed_epsilon": self.privacy_budget.consumed_epsilon,
                "consumed_delta": self.privacy_budget.consumed_delta
            }
        
        if self.time_budget:
            status["time_budget"] = {
                "remaining_seconds": self.time_budget.remaining_time(),
                "consumed_seconds": self.time_budget.consumed_seconds,
                "max_total_seconds": self.time_budget.max_total_seconds
            }
        
        return status