import libvmi
import logging
import json
import os
import asyncio
import threading
from typing import Any, Dict
from configparser import ConfigParser
from sentry_sdk import capture_exception, init as sentry_init
from tenacity import retry, wait_exponential, stop_after_attempt

# Initialize Sentry for error tracking
sentry_init(dsn=os.getenv('SENTRY_DSN'))

# Set up advanced logging
class AdvancedFormatter(logging.Formatter):
    def format(self, record):
        log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

handler = logging.FileHandler('vmi_integration.log')
handler.setFormatter(AdvancedFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

class VMIIntegrationError(Exception):
    """Custom exception for VMI Integration errors."""
    pass

class VMIIntegration:
    def __init__(self, config_path: str = 'vmi_config.ini'):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.vmi = None
        self.vmi_instance = None
        self.lock = threading.Lock()
        self.load_config()
    
    def load_config(self):
        """Load configuration from a file or environment variables."""
        config = ConfigParser()
        try:
            config.read(self.config_path)
            self.config = {section: dict(config.items(section)) for section in config.sections()}
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            capture_exception(e)
            raise VMIIntegrationError("Error loading configuration.")
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    async def initialize_vmi(self):
        """Asynchronously initialize and configure the VMI framework."""
        try:
            async with asyncio.Lock():
                if not self.vmi:
                    self.vmi = libvmi.init()
                    if not self.vmi:
                        raise RuntimeError("Failed to initialize VMI framework.")
                    await self.configure_vmi()
                    logger.info("VMI initialized and configured successfully.")
        except Exception as e:
            logger.error(f"Error initializing VMI: {e}")
            capture_exception(e)
            raise VMIIntegrationError("Error initializing VMI framework.")
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    async def configure_vmi(self):
        """Apply configuration settings to the VMI instance."""
        try:
            if self.vmi:
                for key, value in self.config.get('VMI', {}).items():
                    libvmi.set_option(self.vmi, key, value)
            logger.info("VMI configured successfully.")
        except Exception as e:
            logger.error(f"Error configuring VMI: {e}")
            capture_exception(e)
            raise VMIIntegrationError("Error configuring VMI.")
    
    def get_vmi_instance(self):
        """Obtain an instance of the VMI framework."""
        try:
            with self.lock:
                if self.vmi:
                    self.vmi_instance = libvmi.get_instance(self.vmi)
                    if not self.vmi_instance:
                        raise RuntimeError("Failed to obtain VMI instance.")
                    logger.info("VMI instance obtained successfully.")
                else:
                    raise RuntimeError("VMI is not initialized.")
            return self.vmi_instance
        except Exception as e:
            logger.error(f"Error obtaining VMI instance: {e}")
            capture_exception(e)
            raise VMIIntegrationError("Error obtaining VMI instance.")
    
    async def manage_vmi(self):
        """Manage the VMI lifecycle using async/await."""
        try:
            await self.initialize_vmi()
            vmi_instance = self.get_vmi_instance()
            # Use vmi_instance for further operations
        except Exception as e:
            logger.error(f"Unhandled exception in VMI management: {e}")
            capture_exception(e)
            raise
    
    def __enter__(self):
        """Enter the runtime context related to this object."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.manage_vmi())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context related to this object."""
        if self.vmi:
            libvmi.cleanup(self.vmi)
            logger.info("VMI resources cleaned up.")
