"""Coral NPU hardware detection and availability checking."""

import os
import logging
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default Coral NPU device paths
DEFAULT_DEVICE_PATH = "/dev/coral-npu"
ALTERNATIVE_DEVICE_PATHS = [
    "/dev/apex_0",
    "/dev/apex",
    "/dev/edgetpu",
]

class CoralNPUDetector:
    """Detects and validates Coral NPU hardware availability."""
    
    def __init__(self, device_path: Optional[str] = None):
        """
        Initialize Coral NPU detector.
        
        Args:
            device_path: Optional path to Coral NPU device (default: from env or /dev/coral-npu)
        """
        self.device_path = device_path or os.getenv("CORALNPU_DEVICE_PATH", DEFAULT_DEVICE_PATH)
        self._available = None
        self._capabilities = None
    
    def is_available(self) -> bool:
        """
        Check if Coral NPU hardware is available.
        
        Returns:
            True if NPU is available, False otherwise
        """
        if self._available is not None:
            return self._available
        
        # Check device file exists
        if os.path.exists(self.device_path):
            self._available = True
            logger.info(f"Coral NPU detected at {self.device_path}")
            return True
        
        # Try alternative paths
        for alt_path in ALTERNATIVE_DEVICE_PATHS:
            if os.path.exists(alt_path):
                self.device_path = alt_path
                self._available = True
                logger.info(f"Coral NPU detected at {alt_path}")
                return True
        
        # Check if runtime libraries are available
        try:
            import pycoral
            self._available = True
            logger.info("Coral NPU runtime libraries available (software mode)")
            return True
        except ImportError:
            pass
        
        self._available = False
        logger.warning("Coral NPU not detected")
        return False
    
    def check_permissions(self) -> bool:
        """
        Check if we have permissions to access the NPU device.
        
        Returns:
            True if we have permissions, False otherwise
        """
        if not self.is_available():
            return False
        
        try:
            # Try to open device file
            with open(self.device_path, "rb") as f:
                pass
            return True
        except PermissionError:
            logger.warning(f"No permissions to access Coral NPU at {self.device_path}")
            return False
        except Exception as e:
            logger.warning(f"Error checking NPU permissions: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get NPU hardware capabilities.
        
        Returns:
            Dictionary with capability information
        """
        if self._capabilities is not None:
            return self._capabilities
        
        capabilities = {
            "available": self.is_available(),
            "device_path": self.device_path if self.is_available() else None,
            "has_permissions": self.check_permissions() if self.is_available() else False,
            "runtime_available": False,
        }
        
        # Check for runtime libraries
        try:
            import pycoral
            capabilities["runtime_available"] = True
            capabilities["runtime_version"] = getattr(pycoral, "__version__", "unknown")
        except ImportError:
            pass
        
        # Try to get device info if available
        if capabilities["has_permissions"]:
            try:
                # Use pycoral to get device info if available
                import pycoral.utils.edgetpu as edgetpu
                devices = edgetpu.list_edge_tpus()
                if devices:
                    capabilities["device_count"] = len(devices)
                    capabilities["device_info"] = devices[0] if devices else None
            except Exception as e:
                logger.debug(f"Could not get device info: {e}")
        
        self._capabilities = capabilities
        return capabilities
    
    def validate_runtime(self) -> bool:
        """
        Validate that Coral NPU runtime is available and working.
        
        Returns:
            True if runtime is valid, False otherwise
        """
        try:
            import pycoral
            import pycoral.utils.edgetpu as edgetpu
            
            # Try to list devices
            devices = edgetpu.list_edge_tpus()
            return len(devices) > 0 or self.is_available()
        except ImportError:
            logger.warning("Coral NPU runtime (pycoral) not installed")
            return False
        except Exception as e:
            logger.warning(f"Coral NPU runtime validation failed: {e}")
            return False
    
    def log_hardware_status(self):
        """Log current hardware status."""
        caps = self.get_capabilities()
        logger.info(f"Coral NPU Status: available={caps['available']}, "
                   f"device_path={caps['device_path']}, "
                   f"permissions={caps['has_permissions']}, "
                   f"runtime={caps['runtime_available']}")


def detect_coral_npu(device_path: Optional[str] = None) -> CoralNPUDetector:
    """
    Detect and return Coral NPU detector instance.
    
    Args:
        device_path: Optional device path
        
    Returns:
        CoralNPUDetector instance
    """
    detector = CoralNPUDetector(device_path)
    detector.log_hardware_status()
    return detector

