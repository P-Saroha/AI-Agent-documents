"""
Comprehensive Error Handling Module
Provides custom exceptions, logging, and graceful error recovery
"""
import logging
import traceback
from typing import Optional, Callable, Any
from functools import wraps
import streamlit as st

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agentic_rag.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# Custom Exceptions
class AgenticRAGException(Exception):
    """Base exception for Agentic RAG system"""
    pass


class DocumentProcessingError(AgenticRAGException):
    """Raised when document processing fails"""
    pass


class VectorDBError(AgenticRAGException):
    """Raised when vector database operations fail"""
    pass


class LLMError(AgenticRAGException):
    """Raised when LLM API calls fail"""
    pass


class RetrievalError(AgenticRAGException):
    """Raised when retrieval operations fail"""
    pass


# Error Handler Decorator
def handle_errors(
    default_return: Any = None,
    user_message: str = "An error occurred",
    log_error: bool = True,
    raise_exception: bool = False
):
    """
    Decorator for comprehensive error handling
    
    Args:
        default_return: Value to return on error
        user_message: User-friendly message to display
        log_error: Whether to log the error
        raise_exception: Whether to re-raise the exception
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error with full traceback
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    logger.debug(traceback.format_exc())
                
                # Display user-friendly message in Streamlit
                error_details = f"{user_message}: {str(e)}"
                logger.error(error_details)
                
                # Re-raise if needed
                if raise_exception:
                    raise
                
                return default_return
        return wrapper
    return decorator


class ErrorMonitor:
    """Monitor and track errors across the application"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def log_error(self, context: str, error: Exception, severity: str = "ERROR"):
        """Log an error with context"""
        error_info = {
            "context": context,
            "error": str(error),
            "type": type(error).__name__,
            "severity": severity,
            "traceback": traceback.format_exc()
        }
        
        if severity == "ERROR":
            self.errors.append(error_info)
            logger.error(f"[{context}] {error}")
        else:
            self.warnings.append(error_info)
            logger.warning(f"[{context}] {error}")
    
    def get_error_summary(self) -> str:
        """Get summary of all errors"""
        summary = f"Errors: {len(self.errors)}, Warnings: {len(self.warnings)}"
        if self.errors:
            summary += f"\nRecent errors:\n"
            for err in self.errors[-3:]:
                summary += f"  - [{err['context']}] {err['error']}\n"
        return summary
    
    def clear(self):
        """Clear error history"""
        self.errors = []
        self.warnings = []


# Global error monitor instance
error_monitor = ErrorMonitor()


def safe_execute(func: Callable, *args, fallback=None, error_msg: str = "Operation failed", **kwargs):
    """
    Safely execute a function with error handling
    
    Args:
        func: Function to execute
        fallback: Value to return on error
        error_msg: Error message prefix
        *args, **kwargs: Arguments for the function
    
    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_msg}: {str(e)}")
        error_monitor.log_error(func.__name__, e)
        return fallback


def validate_input(value: Any, value_name: str, expected_type: type = None, required: bool = True):
    """
    Validate input parameters
    
    Args:
        value: Value to validate
        value_name: Name of the parameter
        expected_type: Expected type
        required: Whether the value is required
    
    Raises:
        ValueError: If validation fails
    """
    if required and value is None:
        raise ValueError(f"{value_name} is required but not provided")
    
    if expected_type and value is not None and not isinstance(value, expected_type):
        raise TypeError(f"{value_name} must be of type {expected_type.__name__}, got {type(value).__name__}")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry function on failure
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} failed after {max_retries} attempts")
                        raise
                    
                    logger.warning(f"{func.__name__} attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
            
        return wrapper
    return decorator


def handle_streamlit_errors(error: Exception, context: str = ""):
    """
    Display user-friendly error messages in Streamlit
    
    Args:
        error: The exception that occurred
        context: Context where the error occurred
    """
    error_type = type(error).__name__
    
    # Map technical errors to user-friendly messages
    user_messages = {
        "DocumentProcessingError": "📄 Failed to process document. Please check the file format.",
        "VectorDBError": "🗄️ Database operation failed. Try clearing the database and re-uploading.",
        "LLMError": "🤖 AI model error. Please check your API key or try again.",
        "RetrievalError": "🔍 Search failed. Please try rephrasing your question.",
        "ConnectionError": "🌐 Connection error. Please check your internet connection.",
        "timeout": "⏱️ Operation timed out. Please try again.",
    }
    
    # Find matching message
    user_msg = user_messages.get(error_type, "❌ An unexpected error occurred")
    
    # Display in Streamlit
    st.error(f"{user_msg}")
    
    # Show technical details in expander
    with st.expander("🔧 Technical Details"):
        st.code(f"Error Type: {error_type}\nContext: {context}\nDetails: {str(error)}")
    
    # Log the error
    logger.error(f"[{context}] {error_type}: {str(error)}")
    error_monitor.log_error(context, error)


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key: API key to validate
    
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
    
    if api_key == "your_google_api_key_here":
        return False
    
    if len(api_key) < 20:
        logger.warning("API key seems too short")
        return False
    
    return True


def get_error_report() -> str:
    """Generate a comprehensive error report"""
    report = "=== AGENTIC RAG ERROR REPORT ===\n\n"
    report += error_monitor.get_error_summary()
    report += "\n\nFor support, please check the logs at: agentic_rag.log"
    return report
