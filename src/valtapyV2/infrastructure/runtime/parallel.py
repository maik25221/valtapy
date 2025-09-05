"""Parallel execution utilities for CPU-bound metric computations."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Callable, Any
import logging
import time
from ...domain.entities import MetricResult
from ...domain.errors import MetricExecutionError


def run_in_parallel(tasks: List[Callable[[], MetricResult]], 
                   max_workers: int = 0) -> List[MetricResult]:
    """
    Execute metric computation tasks in parallel using multiprocessing.
    
    This function is optimized for CPU-bound tasks (metric computations).
    Each task should be self-contained and not rely on shared state.
    
    Args:
        tasks: List of callable tasks that return MetricResult
        max_workers: Maximum number of worker processes (0 = auto-detect CPU count)
        
    Returns:
        List of MetricResult objects in the same order as input tasks
        
    Note:
        Failed tasks return MetricResult with error details rather than raising exceptions
    """
    logger = logging.getLogger(__name__)
    
    if not tasks:
        logger.warning("No tasks provided for parallel execution")
        return []
    
    # Determine number of workers
    if max_workers <= 0:
        max_workers = mp.cpu_count()
    
    # Limit workers to avoid resource exhaustion
    max_workers = min(max_workers, len(tasks), mp.cpu_count())
    
    logger.info(f"Starting parallel execution of {len(tasks)} tasks with {max_workers} workers")
    
    results = [None] * len(tasks)
    start_time = time.time()
    
    # Single-threaded execution for small task lists or single worker
    if len(tasks) == 1 or max_workers == 1:
        logger.debug("Using single-threaded execution")
        for i, task in enumerate(tasks):
            results[i] = _execute_single_task(task, i)
    else:
        # Multi-process execution
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_index = {
                    executor.submit(_execute_single_task, task, i): i 
                    for i, task in enumerate(tasks)
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        result = future.result()
                        results[index] = result
                        completed_count += 1
                        
                        if completed_count % max(1, len(tasks) // 10) == 0:
                            logger.debug(f"Completed {completed_count}/{len(tasks)} tasks")
                            
                    except Exception as e:
                        logger.error(f"Task {index} failed with exception: {e}")
                        # Create error result
                        results[index] = MetricResult(
                            id=f"task_{index}",
                            value=0.0,
                            details={"error": str(e), "error_type": type(e).__name__},
                            family="fidelity",
                            purpose_tags=set()
                        )
        
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            # Fall back to sequential execution
            logger.info("Falling back to sequential execution")
            for i, task in enumerate(tasks):
                results[i] = _execute_single_task(task, i)
    
    execution_time = time.time() - start_time
    successful_results = sum(1 for r in results if r and "error" not in r.details)
    failed_results = len(results) - successful_results
    
    logger.info(f"Parallel execution completed in {execution_time:.2f}s: "
               f"{successful_results} successful, {failed_results} failed")
    
    return results


def _execute_single_task(task: Callable[[], MetricResult], task_index: int) -> MetricResult:
    """
    Execute a single task with error handling.
    
    Args:
        task: Task callable to execute
        task_index: Index of the task for error reporting
        
    Returns:
        MetricResult (with error details if task failed)
    """
    logger = logging.getLogger(__name__)
    
    try:
        start_time = time.time()
        result = task()
        execution_time = time.time() - start_time
        
        # Add execution time to result details
        if isinstance(result, MetricResult):
            updated_details = {**result.details, "execution_time_seconds": execution_time}
            return MetricResult(
                id=result.id,
                value=result.value,
                details=updated_details,
                family=result.family,
                purpose_tags=result.purpose_tags
            )
        else:
            logger.warning(f"Task {task_index} returned non-MetricResult: {type(result)}")
            return result
            
    except Exception as e:
        logger.error(f"Task {task_index} failed: {e}")
        
        # Create error result
        return MetricResult(
            id=f"task_{task_index}",
            value=0.0,
            details={
                "error": str(e),
                "error_type": type(e).__name__,
                "task_index": task_index
            },
            family="fidelity",  # Default family for error results
            purpose_tags=set()
        )


def get_optimal_worker_count(task_count: int, cpu_intensive: bool = True) -> int:
    """
    Get optimal number of workers for parallel execution.
    
    Args:
        task_count: Number of tasks to execute
        cpu_intensive: Whether tasks are CPU-intensive (vs I/O-intensive)
        
    Returns:
        Recommended number of workers
    """
    cpu_count = mp.cpu_count()
    
    if cpu_intensive:
        # For CPU-bound tasks, use up to CPU count
        optimal = min(cpu_count, task_count)
    else:
        # For I/O-bound tasks, can use more workers
        optimal = min(cpu_count * 2, task_count)
    
    # Always use at least 1, at most available CPUs
    return max(1, min(optimal, cpu_count))


def run_with_timeout(task: Callable[[], Any], timeout_seconds: float) -> Any:
    """
    Execute a single task with timeout.
    
    Args:
        task: Task to execute
        timeout_seconds: Maximum execution time
        
    Returns:
        Task result
        
    Raises:
        MetricExecutionError: If task times out or fails
    """
    logger = logging.getLogger(__name__)
    
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task)
            result = future.result(timeout=timeout_seconds)
            return result
            
    except mp.TimeoutError:
        raise MetricExecutionError(f"Task timed out after {timeout_seconds} seconds")
    except Exception as e:
        raise MetricExecutionError(f"Task failed: {e}", original_error=e)