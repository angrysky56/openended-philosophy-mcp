"""
nars_manager.py - ONA Subprocess Lifecycle Manager

Manages the entire lifecycle of an OpenNARS for Applications (ONA) subprocess,
ensuring robust startup, graceful shutdown, and structured communication.
This is the definitive communication layer for the NARS engine.
"""

import asyncio
import atexit
import logging
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class NARSManager:
    """
    Manages the ONA subprocess lifecycle with robust cleanup patterns.

    This class implements a comprehensive lifecycle management system for ONA,
    supporting multiple installation methods:
    1. pip-installed ONA package
    2. Local ONA executable
    3. NARS-GPT installation

    Uses environment variables for configuration, ensuring portability
    across different deployment environments.
    """

    def __init__(self):
        """
        Initializes the NARS manager, relying on the pip-installed `ona` package.
        """
        # Load environment variables
        load_dotenv()

        self.ona_path = self._get_ona_executable_path()
        self.process: subprocess.Popen | None = None
        self.process_lock = asyncio.Lock()
        self._shutdown_initiated = False

        # Configuration from environment
        self.memory_size = int(os.getenv("NARS_MEMORY_SIZE", "1000"))
        self.inference_steps = int(os.getenv("NARS_INFERENCE_STEPS", "50"))
        self.silent_mode = os.getenv("NARS_SILENT_MODE", "true").lower() == "true"
        self.decision_threshold = float(os.getenv("NARS_DECISION_THRESHOLD", "0.6"))

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"NARS Manager initialized with ONA at: {self.ona_path}")
        logger.debug(f"Configuration: memory_size={self.memory_size}, "
                    f"inference_steps={self.inference_steps}, "
                    f"silent_mode={self.silent_mode}")

    def _get_ona_executable_path(self) -> Path:
        """
        Finds the ONA executable from the installed `ona` package.
        """
        try:
            import ona
            # The executable is typically in a 'NAR' subdirectory of the package
            package_path = Path(ona.__file__).parent
            ona_executable = package_path / "NAR" / "NAR"
            if ona_executable.exists() and ona_executable.is_file():
                logger.info(f"Found ONA executable at: {ona_executable}")
                return ona_executable
            else:
                 # For some installations, it might be named NARexe
                ona_executable = package_path / "NAR" / "NARexe"
                if ona_executable.exists() and ona_executable.is_file():
                    logger.info(f"Found ONA executable at: {ona_executable}")
                    return ona_executable
        except ImportError:
            logger.error("The 'ona' package is not installed. Please install it using 'pip install ona'")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while finding the ONA executable: {e}")
            raise

        raise FileNotFoundError("Could not find the ONA executable within the 'ona' package. "
                                "Please ensure the package is installed correctly.")

    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handles OS signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating NARS shutdown")
        self.cleanup()
        sys.exit(0)

    async def start(self) -> bool:
        """
        Starts the ONA subprocess with proper configuration.

        Implements idempotent startup with configuration based on
        environment variables for maximum flexibility.
        
        Returns:
            bool: True if process started successfully, False otherwise
        """
        async with self.process_lock:
            if self.process and self.process.poll() is None:
                logger.debug("ONA process already running")
                return True

            logger.info("Starting ONA subprocess...")
            try:
                self.process = subprocess.Popen(
                    [str(self.ona_path), "shell"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,  # Line buffered
                    preexec_fn=os.setsid if sys.platform != "win32" else None
                )

                # Configure ONA based on environment settings
                if self.silent_mode:
                    await self._send_command("*volume=0")

                await self._send_command(f"*memorysize={self.memory_size}")
                await self._send_command(f"*decisionthreshold={self.decision_threshold}")
                await self._send_command("*reset")

                logger.info(f"ONA process started with PID: {self.process.pid}")
                return True

            except Exception as e:
                logger.error(f"Failed to start ONA process: {e}")
                self.process = None
                return False

    async def stop(self) -> None:
        """
        Stops the ONA subprocess gracefully with timeout fallback.

        Implements a two-phase shutdown:
        1. Graceful shutdown via quit command
        2. Forced termination if graceful fails
        """
        async with self.process_lock:
            if not self.process:
                return

            if self.process.poll() is None:
                logger.info("Initiating graceful ONA shutdown...")
                try:
                    if self.process.stdin:
                        self.process.stdin.write("quit\n")
                        await asyncio.to_thread(self.process.stdin.flush)

                    # Wait for graceful shutdown
                    await asyncio.wait_for(
                        asyncio.to_thread(self.process.wait),
                        timeout=5.0
                    )
                    logger.info("ONA shutdown gracefully")

                except (BrokenPipeError, asyncio.TimeoutError) as e:
                    logger.warning(f"Graceful shutdown failed ({e}), forcing termination")
                    self._terminate_process()

            self.process = None

    def _terminate_process(self) -> None:
        """Force-terminates the ONA process group."""
        if self.process and self.process.poll() is None:
            try:
                if sys.platform != "win32":
                    # Unix: Kill entire process group
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    # Windows: Kill process tree
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                        check=False
                    )
                self.process.wait(timeout=2)
                logger.info("ONA process terminated forcefully")
            except Exception as e:
                logger.error(f"Error during process termination: {e}")

    async def _send_command(self, command: str) -> None:
        """
        Sends a command to ONA with automatic process management.

        Ensures process is running before sending commands and handles
        communication errors gracefully.
        """
        if self.process is None or self.process.poll() is not None:
            await self.start()

        if not self.process or not self.process.stdin:
            raise RuntimeError("NARS process not available for commands")

        try:
            self.process.stdin.write(f"{command}\n")
            await asyncio.to_thread(self.process.stdin.flush)
            logger.debug(f"Sent command to ONA: {command}")
        except BrokenPipeError:
            logger.error("Lost connection to ONA process")
            self.process = None
            raise

    async def query(self, narsese: str, timeout: float = 5.0) -> dict[str, Any]:
        """
        Sends a query to ONA and returns parsed results.

        Args:
            narsese: The Narsese query statement
            timeout: Maximum time to wait for response

        Returns:
            Dictionary containing parsed results with structure:
            {
                "answers": List of answer beliefs,
                "derivations": List of derived beliefs,
                "raw": Raw output string,
                "error": Error message if any
            }
        """
        async with self.process_lock:
            if not self.process or self.process.poll() is not None:
                success = await self.start()
                if not success:
                    return {"error": "Failed to start NARS process", "answers": [], "derivations": []}

            try:
                logger.debug(f"Sending query to ONA: {narsese}")
                await self._send_command(narsese)
                
                # Give ONA a moment to process
                await asyncio.sleep(0.1)
                
                # Trigger inference cycle
                await self._send_command("1")
                
                output_lines = await asyncio.wait_for(
                    self._get_output(),
                    timeout=timeout
                )
                
                logger.debug(f"Received {len(output_lines)} lines of output")
                result = self._parse_output(output_lines)
                logger.debug(f"Parsed result: {result}")
                return result

            except asyncio.TimeoutError:
                logger.warning(f"ONA query timeout for: {narsese}")
                return {"error": "timeout", "answers": [], "derivations": [], "raw": ""}

            except Exception as e:
                logger.error(f"Error during ONA query: {e}", exc_info=True)
                return {"error": str(e), "answers": [], "derivations": [], "raw": ""}

    async def _get_output(self) -> list[str]:
        """
        Reads output from ONA with improved timeout and completion detection.

        Implements line-by-line reading with proper EOF handling,
        multiple completion markers, and timeout protection.
        """
        if not self.process or not self.process.stdout:
            raise RuntimeError("NARS process not available for output")

        lines = []
        completion_markers = [
            "done with 0 additional inference steps.",
            "done with",
            "NAR>",
            ""  # Empty line can also indicate completion
        ]
        
        # Read lines with timeout protection
        start_time = asyncio.get_event_loop().time()
        max_wait_time = 2.0  # Maximum wait time for output
        
        try:
            while True:
                # Check timeout
                if asyncio.get_event_loop().time() - start_time > max_wait_time:
                    logger.debug("Output reading timeout reached")
                    break
                
                # Try to read a line with timeout
                try:
                    line = await asyncio.wait_for(
                        asyncio.to_thread(self.process.stdout.readline),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # No more output available
                    break
                
                if not line:  # EOF
                    break

                line = line.strip()
                
                # Check for completion markers
                if any(marker in line for marker in completion_markers if marker):
                    if line:  # Add the completion line if it's not empty
                        lines.append(line)
                    break

                if line:  # Skip empty lines but add non-empty ones
                    lines.append(line)
                    
                # If we have some output and see an empty line, might be done
                if not line and lines:
                    break

        except Exception as e:
            logger.warning(f"Error reading ONA output: {e}")
            
        return lines

    def _parse_output(self, lines: list[str]) -> dict[str, Any]:
        """
        Parses ONA output into structured format.

        Extracts answers, derivations, and truth values from
        raw ONA output lines.
        """
        result = {
            "answers": [],
            "derivations": [],
            "raw": "\n".join(lines)
        }

        for line in lines:
            if line.startswith("Answer:"):
                answer = self._parse_task(line[7:].strip())
                if answer:
                    result["answers"].append(answer)

            elif line.startswith("Derived:"):
                derivation = self._parse_task(line[8:].strip())
                if derivation:
                    result["derivations"].append(derivation)

        return result

    def _parse_task(self, task_str: str) -> dict[str, Any] | None:
        """
        Parses a single ONA task/belief string.

        Extracts term and truth value (frequency, confidence) from
        task representation.
        """
        if not task_str:
            return None

        # Pattern for truth values
        truth_pattern = r'Truth:\s*frequency=([\d.]+)\s*confidence=([\d.]+)'
        truth_match = re.search(truth_pattern, task_str)

        if truth_match:
            term = task_str[:truth_match.start()].strip()
            truth = {
                "frequency": float(truth_match.group(1)),
                "confidence": float(truth_match.group(2))
            }
            return {"term": term, "truth": truth}
        else:
            # No truth value, just term
            return {"term": task_str.strip(), "truth": None}

    def cleanup(self) -> None:
        """
        Master cleanup handler for graceful shutdown.

        Called by atexit and signal handlers to ensure ONA process
        is properly terminated.
        """
        if not self._shutdown_initiated:
            self._shutdown_initiated = True
            logger.info("Executing NARS Manager cleanup")
            self._terminate_process()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        await self.stop()
