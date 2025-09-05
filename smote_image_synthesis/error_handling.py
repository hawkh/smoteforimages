"""
Comprehensive error handling and recovery mechanisms for SMOTE image synthesis pipeline.
"""

import logging
import traceback
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from pathlib import Path
import torch
import numpy as np
from datetime import datetime
import json
import warnings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class EncoderError(PipelineError):
    """Error in image encoding process."""
    pass


class DecoderError(PipelineError):
    """Error in image decoding process."""
    pass


class SMOTEError(PipelineError):
    """Error in SMOTE processing."""
    pass


class QualityAssessmentError(PipelineError):
    """Error in quality assessment."""
    pass


class ConfigurationError(PipelineError):
    """Error in configuration validation."""
    pass


class DataValidationError(PipelineError):
    """Error in data validation."""
    pass


class MemoryError(PipelineError):
    """Memory-related error."""
    pass


class ErrorRecoveryManager:
    """
    Manages error recovery strategies and fallback mechanisms for the pipeline.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_fallbacks: bool = True,
        log_errors: bool = True
    ):
        """
        Initialize error recovery manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retry attempts (seconds)
            enable_fallbacks: Whether to enable fallback mechanisms
            log_errors: Whether to log errors
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_fallbacks = enable_fallbacks
        self.log_errors = log_errors
        
        # Error statistics
        self.error_counts = {}
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'fallback_used': 0
        }
        
        # Fallback configurations
        self.fallback_configs = {
            'encoder': {
                'reduce_batch_size': True,
                'use_cpu_fallback': True,
                'simplify_architecture': True
            },
            'decoder': {
                'reduce_complexity': True,
                'disable_features': ['perceptual_loss', 'skip_connections'],
                'use_basic_architecture': True
            },
            'smote': {
                'reduce_k_neighbors': True,
                'disable_clustering': True,
                'simplify_validation': True
            },
            'quality': {
                'reduce_metrics': True,
                'use_simple_metrics': ['mse', 'mae'],
                'skip_diversity': True
            }
        }
    
    def handle_error(
        self,
        error: Exception,
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Handle an error with appropriate recovery strategy.
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation that failed
            context: Additional context information
            
        Returns:
            Tuple of (recovered, fallback_config)
        """
        self.recovery_stats['total_errors'] += 1
        error_key = f"{component}.{operation}.{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        if self.log_errors:
            logger.error(f"Error in {component}.{operation}: {error}")
            logger.debug(f"Error context: {context}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        # Determine recovery strategy
        recovery_strategy = self._get_recovery_strategy(error, component, operation)
        
        if recovery_strategy:
            try:
                fallback_config = self._apply_recovery_strategy(
                    recovery_strategy, component, operation, context
                )
                self.recovery_stats['recovered_errors'] += 1
                
                if recovery_strategy.get('use_fallback', False):
                    self.recovery_stats['fallback_used'] += 1
                
                logger.info(f"Applied recovery strategy for {component}.{operation}")
                return True, fallback_config
                
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
                self.recovery_stats['failed_recoveries'] += 1
        
        return False, None
    
    def _get_recovery_strategy(
        self, 
        error: Exception, 
        component: str, 
        operation: str
    ) -> Optional[Dict[str, Any]]:
        """Determine appropriate recovery strategy for the error."""
        
        # Memory-related errors
        if self._is_memory_error(error):
            return {
                'type': 'memory',
                'reduce_batch_size': True,
                'use_cpu_fallback': True,
                'use_fallback': True
            }
        
        # GPU/CUDA errors
        if self._is_gpu_error(error):
            return {
                'type': 'gpu',
                'use_cpu_fallback': True,
                'use_fallback': True
            }
        
        # Data validation errors
        if isinstance(error, (DataValidationError, ValueError)):
            return {
                'type': 'data_validation',
                'filter_invalid_data': True,
                'use_default_values': True
            }
        
        # Configuration errors
        if isinstance(error, ConfigurationError):
            return {
                'type': 'configuration',
                'use_default_config': True,
                'use_fallback': True
            }
        
        # Component-specific strategies
        if component == 'encoder':
            return self._get_encoder_recovery_strategy(error, operation)
        elif component == 'decoder':
            return self._get_decoder_recovery_strategy(error, operation)
        elif component == 'smote':
            return self._get_smote_recovery_strategy(error, operation)
        elif component == 'quality':
            return self._get_quality_recovery_strategy(error, operation)
        
        return None
    
    def _get_encoder_recovery_strategy(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Get recovery strategy for encoder errors."""
        base_strategy = {
            'type': 'encoder',
            'use_fallback': True
        }
        
        if operation == 'encode':
            base_strategy.update({
                'reduce_batch_size': True,
                'simplify_preprocessing': True
            })
        elif operation == 'fine_tune':
            base_strategy.update({
                'reduce_learning_rate': True,
                'reduce_epochs': True,
                'freeze_more_layers': True
            })
        
        return base_strategy
    
    def _get_decoder_recovery_strategy(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Get recovery strategy for decoder errors."""
        base_strategy = {
            'type': 'decoder',
            'use_fallback': True
        }
        
        if operation == 'decode':
            base_strategy.update({
                'reduce_complexity': True,
                'disable_advanced_features': True
            })
        elif operation == 'train':
            base_strategy.update({
                'reduce_learning_rate': True,
                'simplify_architecture': True,
                'disable_perceptual_loss': True
            })
        
        return base_strategy
    
    def _get_smote_recovery_strategy(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Get recovery strategy for SMOTE errors."""
        return {
            'type': 'smote',
            'reduce_k_neighbors': True,
            'disable_clustering': True,
            'use_basic_smote': True,
            'use_fallback': True
        }
    
    def _get_quality_recovery_strategy(self, error: Exception, operation: str) -> Dict[str, Any]:
        """Get recovery strategy for quality assessment errors."""
        return {
            'type': 'quality',
            'use_simple_metrics': True,
            'reduce_sample_size': True,
            'skip_advanced_analysis': True,
            'use_fallback': True
        }
    
    def _apply_recovery_strategy(
        self,
        strategy: Dict[str, Any],
        component: str,
        operation: str,
        context: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Apply the recovery strategy and return fallback configuration."""
        
        if not self.enable_fallbacks:
            return None
        
        fallback_config = {}
        
        # Apply general strategies
        if strategy.get('reduce_batch_size', False):
            fallback_config['batch_size'] = max(1, (context or {}).get('batch_size', 32) // 2)
        
        if strategy.get('use_cpu_fallback', False):
            fallback_config['device'] = 'cpu'
        
        # Apply component-specific fallbacks
        if component in self.fallback_configs:
            component_fallback = self.fallback_configs[component].copy()
            fallback_config.update(component_fallback)
        
        # Apply strategy-specific configurations
        if strategy['type'] == 'memory':
            fallback_config.update({
                'reduce_model_complexity': True,
                'use_gradient_checkpointing': True,
                'clear_cache': True
            })
        elif strategy['type'] == 'gpu':
            fallback_config.update({
                'force_cpu': True,
                'disable_mixed_precision': True
            })
        
        return fallback_config
    
    def _is_memory_error(self, error: Exception) -> bool:
        """Check if error is memory-related."""
        error_str = str(error).lower()
        memory_keywords = [
            'out of memory', 'cuda out of memory', 'memory error',
            'allocation failed', 'insufficient memory'
        ]
        return any(keyword in error_str for keyword in memory_keywords)
    
    def _is_gpu_error(self, error: Exception) -> bool:
        """Check if error is GPU/CUDA-related."""
        error_str = str(error).lower()
        gpu_keywords = [
            'cuda error', 'gpu error', 'device error',
            'cublas', 'cudnn', 'nvidia'
        ]
        return any(keyword in error_str for keyword in gpu_keywords)
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get error recovery statistics."""
        total_errors = self.recovery_stats['total_errors']
        recovery_rate = (
            self.recovery_stats['recovered_errors'] / total_errors * 100
            if total_errors > 0 else 0
        )
        
        return {
            'total_errors': total_errors,
            'recovered_errors': self.recovery_stats['recovered_errors'],
            'failed_recoveries': self.recovery_stats['failed_recoveries'],
            'fallback_used': self.recovery_stats['fallback_used'],
            'recovery_rate': recovery_rate,
            'error_counts': self.error_counts.copy()
        }
    
    def reset_stats(self):
        """Reset error recovery statistics."""
        self.error_counts.clear()
        self.recovery_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'failed_recoveries': 0,
            'fallback_used': 0
        }


def with_error_handling(
    component: str,
    operation: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    recovery_manager: Optional[ErrorRecoveryManager] = None
):
    """
    Decorator for adding error handling and recovery to pipeline methods.
    
    Args:
        component: Component name
        operation: Operation name
        max_retries: Maximum retry attempts
        retry_delay: Delay between retries
        recovery_manager: Error recovery manager instance
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            recovery_mgr = recovery_manager or getattr(args[0], '_recovery_manager', None)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as error:
                    last_error = error
                    
                    if attempt == max_retries:
                        # Final attempt failed
                        if recovery_mgr:
                            context = {
                                'attempt': attempt,
                                'args_info': _get_args_info(args, kwargs),
                                'function': func.__name__
                            }
                            recovery_mgr.handle_error(error, component, operation, context)
                        break
                    
                    # Try recovery if manager available
                    if recovery_mgr:
                        context = {
                            'attempt': attempt,
                            'args_info': _get_args_info(args, kwargs),
                            'function': func.__name__
                        }
                        
                        recovered, fallback_config = recovery_mgr.handle_error(
                            error, component, operation, context
                        )
                        
                        if recovered and fallback_config:
                            # Apply fallback configuration
                            kwargs = _apply_fallback_config(kwargs, fallback_config)
                            logger.info(f"Retrying {component}.{operation} with fallback config")
                    
                    # Wait before retry
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        logger.info(f"Retrying {component}.{operation} (attempt {attempt + 2}/{max_retries + 1})")
            
            # All retries exhausted
            logger.error(f"All retry attempts exhausted for {component}.{operation}")
            raise last_error
            
        return wrapper
    return decorator


def _get_args_info(args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract relevant information from function arguments."""
    info = {}
    
    # Extract common information
    if args:
        if hasattr(args[0], '__class__'):
            info['instance_type'] = args[0].__class__.__name__
    
    # Extract tensor/array shapes
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            info[f'{key}_shape'] = list(value.shape)
            info[f'{key}_device'] = str(value.device)
        elif isinstance(value, np.ndarray):
            info[f'{key}_shape'] = list(value.shape)
        elif isinstance(value, (int, float, str, bool)):
            info[key] = value
    
    return info


def _apply_fallback_config(kwargs: dict, fallback_config: Dict[str, Any]) -> dict:
    """Apply fallback configuration to function kwargs."""
    updated_kwargs = kwargs.copy()
    
    # Apply direct parameter updates
    for key, value in fallback_config.items():
        if key in updated_kwargs:
            updated_kwargs[key] = value
    
    # Apply special configurations
    if fallback_config.get('force_cpu', False):
        if 'device' in updated_kwargs:
            updated_kwargs['device'] = torch.device('cpu')
    
    if fallback_config.get('reduce_batch_size', False):
        if 'batch_size' in updated_kwargs:
            updated_kwargs['batch_size'] = max(1, updated_kwargs['batch_size'] // 2)
    
    return updated_kwargs


class PipelineHealthMonitor:
    """
    Monitors pipeline health and provides diagnostic information.
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks (seconds)
        """
        self.check_interval = check_interval
        self.health_history = []
        self.last_check = None
        
    def check_health(self, pipeline_components: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check health of pipeline components.
        
        Args:
            pipeline_components: Dictionary of pipeline components
            
        Returns:
            Health status report
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {},
            'warnings': [],
            'recommendations': []
        }
        
        # Check each component
        for name, component in pipeline_components.items():
            component_health = self._check_component_health(name, component)
            health_status['components'][name] = component_health
            
            if component_health['status'] != 'healthy':
                health_status['overall_status'] = 'degraded'
        
        # Check system resources
        resource_health = self._check_system_resources()
        health_status['system_resources'] = resource_health
        
        if resource_health['status'] != 'healthy':
            health_status['overall_status'] = 'degraded'
        
        # Store health history
        self.health_history.append(health_status)
        self.last_check = datetime.now()
        
        # Keep only recent history
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        return health_status
    
    def _check_component_health(self, name: str, component: Any) -> Dict[str, Any]:
        """Check health of individual component."""
        health = {
            'status': 'healthy',
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Check if component has required methods
            if hasattr(component, 'model') and component.model is not None:
                # Check model parameters
                if hasattr(component.model, 'parameters'):
                    total_params = sum(p.numel() for p in component.model.parameters())
                    health['metrics']['total_parameters'] = total_params
                    
                    # Check for NaN parameters
                    nan_params = sum(torch.isnan(p).sum().item() for p in component.model.parameters())
                    if nan_params > 0:
                        health['status'] = 'unhealthy'
                        health['issues'].append(f'Model has {nan_params} NaN parameters')
            
            # Check component-specific health
            if hasattr(component, '_is_trained'):
                health['metrics']['is_trained'] = component._is_trained
            
            if hasattr(component, 'device'):
                health['metrics']['device'] = str(component.device)
                
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f'Health check failed: {str(e)}')
        
        return health
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability."""
        resources = {
            'status': 'healthy',
            'memory': {},
            'gpu': {}
        }
        
        try:
            # Check GPU memory if available
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i)
                    memory_cached = torch.cuda.memory_reserved(i)
                    
                    resources['gpu'][f'device_{i}'] = {
                        'memory_allocated_mb': memory_allocated / 1024 / 1024,
                        'memory_cached_mb': memory_cached / 1024 / 1024
                    }
                    
                    # Check if memory usage is high
                    if memory_allocated / 1024 / 1024 > 8000:  # 8GB threshold
                        resources['status'] = 'warning'
        
        except Exception as e:
            resources['status'] = 'error'
            resources['error'] = str(e)
        
        return resources
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the specified time period."""
        if not self.health_history:
            return {'message': 'No health data available'}
        
        # Filter recent history
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_history = [
            h for h in self.health_history
            if datetime.fromisoformat(h['timestamp']).timestamp() > cutoff_time
        ]
        
        if not recent_history:
            return {'message': f'No health data available for last {hours} hours'}
        
        # Calculate statistics
        total_checks = len(recent_history)
        healthy_checks = sum(1 for h in recent_history if h['overall_status'] == 'healthy')
        degraded_checks = sum(1 for h in recent_history if h['overall_status'] == 'degraded')
        
        return {
            'period_hours': hours,
            'total_checks': total_checks,
            'healthy_checks': healthy_checks,
            'degraded_checks': degraded_checks,
            'health_rate': healthy_checks / total_checks * 100 if total_checks > 0 else 0,
            'last_check': self.last_check.isoformat() if self.last_check else None,
            'current_status': recent_history[-1]['overall_status'] if recent_history else 'unknown'
        }