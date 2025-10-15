import logging
import subprocess
import sys
from pathlib import Path

from ignite.handlers import ProgressBar

default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger_formatter = logging.Formatter("%(asctime)s   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
if not len(default_logger.handlers):
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(default_logger_formatter)
    default_logger.addHandler(sh)


class Logger:
    """
    Base class for logging functionality.

    This class provides basic logging capabilities, including writing to a file
    and optionally displaying progress through an ignite progress bar.

    Parameters
    ----------
    log_path : Path
        Path to the log file.
    progress_bar : ProgressBar | None
        An optional Ignite ProgressBar object for displaying progress, defaults to None.
    logger_name : str
        The name of the logger.
    """

    def __init__(
        self,
        log_path: Path,
        logger_name: str,
        progress_bar: ProgressBar | None = None,
    ) -> None:
        self._log_path = log_path
        self._log = []
        self._progress_bar = progress_bar

        formatter = logging.Formatter("%(asctime)s   %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        fh = logging.FileHandler(self._log_path, mode="w")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # If progress bar is not set, add stdout logging
        if self._progress_bar is None:
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        logger.setLevel(logging.DEBUG)
        self.logger = logger

    def log(self, msg: str, level: int = logging.INFO) -> None:
        """
        Log a message to the progress bar, internal log, and file.

        Parameters
        ----------
        msg : str
            The message to be logged.
        level : int, optional
            The logging level (default is logging.INFO).
        """

        if self._progress_bar is not None:
            self._progress_bar.log_message(msg)
        self._log.append(msg)
        logs = msg.split("\n")
        for log in logs:
            self.logger.log(level, log)

    def get_logs(self) -> str:
        """
        Get all logged messages as a single string.

        Returns
        -------
        str
            A string containing all logged messages, separated by newlines.
        """

        return "\n".join(self._log)

    def log_dict(self, log_dict: dict) -> None:
        """
        Log dictionary values.

        Parameters
        ----------
        log_dict : dict
            A dictionary to be logged.
        """

        for k, v in log_dict.items():
            self.log(f"{k} = {v}")

    def log_nvidia_smi(self) -> None:
        """
        Log the output of nvidia-smi command.

        This method attempts to run the nvidia-smi command and log its output.
        If the command fails, it logs a warning message.
        """

        self.log("--- nvidia-smi output ---")
        try:
            out = subprocess.check_output("nvidia-smi", text=True, encoding="UTF-8")  # noqa: S603, S607
            self.log(out)
        except Exception as err:
            self.log(f"nvidia-smi cannot be run: {err}", logging.WARN)
