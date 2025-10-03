#Utilities that standardize labels, span attributes, and metric naming across modules for consistent dashboards

from typing import Dict, Any, Optional
from contextlib import contextmanager
import time
import logging
from dataclasses import dataclass

# Configure logger
logger = logging.getLogger(__name__)

@dataclass
class TraceSpan:
    """Represents a trace span."""
    span_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ObservabilityManager:
    """Manages observability operations."""
    
    def __init__(self):
        self.active_spans: Dict[str, TraceSpan] = {}
        self.metrics: Dict[str, float] = {}
    
    def start_span(self, request_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None) -> TraceSpan:
        """Start a new trace span."""
        span = TraceSpan(
            span_id=f"{request_id}_{operation}",
            operation=operation,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.active_spans[span.span_id] = span
        logger.debug(f"Started span: {span.span_id}")
        return span
    
    def end_span(self, span_id: str) -> Optional[TraceSpan]:
        """End a trace span."""
        if span_id in self.active_spans:
            span = self.active_spans[span_id]
            span.end_time = time.time()
            duration = span.end_time - span.start_time
            logger.debug(f"Ended span: {span_id}, duration: {duration:.3f}s")
            return self.active_spans.pop(span_id)
        return None
    
    def log_metric(self, name: str, value: float) -> None:
        """Log a metric value."""
        self.metrics[name] = value
        logger.debug(f"Metric: {name} = {value}")
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()

# Global observability manager
observability = ObservabilityManager()

@contextmanager
def trace_request(request_id: str, operation: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for tracing operations."""
    span = observability.start_span(request_id, operation, metadata)
    try:
        yield span
    finally:
        observability.end_span(span.span_id)

def log_metrics(metrics: Dict[str, float]) -> None:
    """Log metrics using the global observability manager."""
    observability.log_metrics(metrics)

def log_metric(name: str, value: float) -> None:
    """Log a single metric."""
    observability.log_metric(name, value)