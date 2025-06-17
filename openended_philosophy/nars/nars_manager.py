"""
NARS Manager - ONA Process Management with Proper Cleanup
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Manages OpenNARS for Applications (ONA) subprocess with comprehensive
process lifecycle management to prevent resource leaks.
"""

import asyncio
import atexit
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class NARSManager:
    """
    Manages ONA subprocess lifecycle with proper cleanup patterns.
    
    This implementation follows MCP best practices for subprocess management:
    - Signal handling for graceful shutdown
    - Process tracking and termination
    - Resource cleanup guarantees
    - Timeout management
    """
    
    def __init__(self, ona_path: Optional[Path] = None):
        """
        Initialize NARS manager.
        
        Args:
            ona_path: Path to ONA executable. If None, searches standard locations.
        """
        self.ona_path = ona_path or self._find_ona_executable()
        self.process: Optional[subprocess.Popen] = None
        self.process_lock = asyncio.Lock()
        self._shutdown_initiated = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        logger.info(f"NARS Manager initialized with ONA path: {self.ona_path}")
        
    def _find_ona_executable(self) -> Path:
        """Find ONA executable in standard locations."""
        # Check relative to NARS-GPT location
        base_path = Path(__file__).parent.parent.parent.parent
        possible_paths = [
            base_path / "NARS-GPT" / "OpenNARS-for-Applications" / "NAR",
            base_path.parent / "NARS-GPT" / "OpenNARS-for-Applications" / "NAR",
            Path("/home/ty/Repositories/ai_workspace/NARS-GPT/OpenNARS-for-Applications/NAR"),
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                logger.debug(f"Found ONA executable at: {path}")
                return path
                
        # If not found, return expected location
        expected = Path("/home/ty/Repositories/ai_workspace/NARS-GPT/OpenNARS-for-Applications/NAR")
        logger.warning(f"ONA executable not found. Expected at: {expected}")
        return expected
        
    def _signal_handler(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating NARS shutdown")
        self.cleanup()
        
    async def start(self) -> None:
        """Start ONA subprocess with proper error handling."""
        async with self.process_lock:
            if self.process is not None and self.process.poll() is None:
                logger.debug("ONA process already running")
                return
                
            try:
                # Ensure ONA executable exists
                if not self.ona_path.exists():
                    raise FileNotFoundError(f"ONA executable not found at: {self.ona_path}")
                    
                # Start ONA process
                self.process = subprocess.Popen(
                    [str(self.ona_path), "shell"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    preexec_fn=os.setsid  # Create new process group for proper cleanup
                )
                
                # Initial configuration
                await self._send_command("*volume=0")  # Reduce verbosity
                await self._send_command("*reset")     # Clean state
                
                logger.info(f"ONA process started with PID: {self.process.pid}")
                
            except Exception as e:
                logger.error(f"Failed to start ONA process: {e}")
                raise
                
    async def stop(self) -> None:
        """Stop ONA subprocess gracefully."""
        async with self.process_lock:
            if self.process is None:
                return
                
            try:
                # Send quit command
                if self.process.poll() is None:
                    self.process.stdin.write("quit\n")
                    self.process.stdin.flush()
                    
                    # Wait for graceful shutdown (max 5 seconds)
                    try:
                        await asyncio.wait_for(
                            asyncio.create_task(asyncio.to_thread(self.process.wait)),
                            timeout=5.0
                        )
                    except asyncio.TimeoutError:
                        logger.warning("ONA process did not shut down gracefully, terminating")
                        self._terminate_process()
                        
            except Exception as e:
                logger.error(f"Error stopping ONA process: {e}")
                self._terminate_process()
            finally:
                self.process = None
                
    def _terminate_process(self) -> None:
        """Force terminate ONA process."""
        if self.process is None:
            return
            
        try:
            # Terminate process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            
            # Wait briefly
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
                
            logger.info("ONA process terminated")
        except Exception as e:
            logger.error(f"Error terminating ONA process: {e}")
            
    async def _send_command(self, command: str) -> None:
        """Send command to ONA process."""
        if self.process is None or self.process.poll() is not None:
            await self.start()
            
        self.process.stdin.write(f"{command}\n")
        self.process.stdin.flush()
        
    async def query(self, narsese: str, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Query ONA with Narsese statement.
        
        Args:
            narsese: Narsese query statement
            timeout: Maximum time to wait for response
            
        Returns:
            Query results including answers, derivations, and truth values
        """
        async with self.process_lock:
            if self.process is None or self.process.poll() is not None:
                await self.start()
                
            try:
                # Send query
                await self._send_command(narsese)
                
                # Get output with timeout
                output = await asyncio.wait_for(
                    self._get_output(),
                    timeout=timeout
                )
                
                return self._parse_output(output)
                
            except asyncio.TimeoutError:
                logger.warning(f"ONA query timeout for: {narsese}")
                return {"error": "timeout", "answers": []}
            except Exception as e:
                logger.error(f"ONA query error: {e}")
                return {"error": str(e), "answers": []}
                
    async def _get_output(self) -> list[str]:
        """Get output from ONA process."""
        await self._send_command("0")  # Trigger output
        
        lines = []
        while True:
            line = await asyncio.to_thread(self.process.stdout.readline)
            if not line or "done with 0 additional inference steps." in line:
                break
            lines.append(line.strip())
            
        return lines
        
    def _parse_output(self, lines: list[str]) -> Dict[str, Any]:
        """Parse ONA output into structured format."""
        result = {
            "answers": [],
            "derivations": [],
            "raw": "\n".join(lines)
        }
        
        for line in lines:
            if line.startswith("Answer:"):
                answer = self._parse_task(line.split("Answer: ")[1])
                result["answers"].append(answer)
            elif line.startswith("Derived:"):
                derivation = self._parse_task(line.split("Derived: ")[1])
                result["derivations"].append(derivation)
                
        return result
        
    def _parse_task(self, task_str: str) -> Dict[str, Any]:
        """Parse ONA task output."""
        task = {"term": "None", "truth": None}
        
        # Extract term
        if " Truth:" in task_str:
            task["term"] = task_str.split(" Truth:")[0].strip()
            
            # Extract truth values
            truth_part = task_str.split("Truth: ")[1]
            if "frequency=" in truth_part and "confidence=" in truth_part:
                freq = float(truth_part.split("frequency=")[1].split(" ")[0])
                conf = float(truth_part.split("confidence=")[1].split(" ")[0])
                task["truth"] = {"frequency": freq, "confidence": conf}
                
        else:
            task["term"] = task_str.strip()
            
        return task
        
    def cleanup(self) -> None:
        """Clean up ONA process and resources."""
        if self._shutdown_initiated:
            return
            
        self._shutdown_initiated = True
        logger.info("Cleaning up NARS Manager")
        
        if self.process is not None and self.process.poll() is None:
            self._terminate_process()
            
        self.process = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
