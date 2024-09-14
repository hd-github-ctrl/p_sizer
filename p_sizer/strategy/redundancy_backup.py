import logging
import os
import json
import shutil
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RedundancyBackup:
    """
    Provides a redundancy system for creating, storing, and managing backups of trading bot data,
    with support for periodic and critical backups.
    """

    def __init__(self, backup_directory: str, backup_frequency: int, max_backups: int = 5):
        """
        Initialize the RedundancyBackup system.

        :param backup_directory: Directory where backups will be stored.
        :param backup_frequency: Frequency of backups in minutes.
        :param max_backups: Maximum number of backup files to retain. Older backups will be deleted.
        """
        self.backup_directory = backup_directory
        self.backup_frequency = timedelta(minutes=backup_frequency)
        self.max_backups = max_backups

        if not os.path.exists(self.backup_directory):
            os.makedirs(self.backup_directory)
            logger.info(f"Created backup directory at {self.backup_directory}")

        logger.info(f"RedundancyBackup initialized with backup frequency: {backup_frequency} minutes, "
                    f"and max backups: {max_backups}")

    def create_backup(self, data: dict, backup_type: str = "regular") -> None:
        """
        Creates a backup of the current state.

        :param data: The data to be backed up.
        :param backup_type: Type of backup (e.g., "regular", "critical").
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{backup_type}_backup_{timestamp}.json"
        backup_path = os.path.join(self.backup_directory, backup_filename)

        try:
            with open(backup_path, 'w') as backup_file:
                json.dump(data, backup_file)
                logger.info(f"Created {backup_type} backup at {backup_path}")
            self._manage_old_backups()
        except Exception as e:
            logger.error(f"Error creating {backup_type} backup: {e}")

    def restore_backup(self, backup_filename: str) -> dict:
        """
        Restores the bot's state from a specified backup file.

        :param backup_filename: Name of the backup file to restore from.
        :return: The restored data.
        """
        backup_path = os.path.join(self.backup_directory, backup_filename)

        if not os.path.exists(backup_path):
            logger.error(f"Backup file {backup_filename} does not exist.")
            return {}

        try:
            with open(backup_path, 'r') as backup_file:
                data = json.load(backup_file)
                logger.info(f"Restored data from backup file {backup_filename}")
                return data
        except Exception as e:
            logger.error(f"Error restoring backup {backup_filename}: {e}")
            return {}

    def _manage_old_backups(self) -> None:
        """
        Manages old backups by deleting the oldest backups if the number exceeds max_backups.
        """
        try:
            backups = sorted([f for f in os.listdir(self.backup_directory) if f.endswith('.json')],
                             reverse=True)

            if len(backups) > self.max_backups:
                for backup_to_delete in backups[self.max_backups:]:
                    backup_path = os.path.join(self.backup_directory, backup_to_delete)
                    os.remove(backup_path)
                    logger.info(f"Deleted old backup: {backup_path}")
        except Exception as e:
            logger.error(f"Error managing old backups: {e}")

    def periodic_backup(self, data: dict) -> None:
        """
        Performs a periodic backup based on the configured backup frequency.

        :param data: The data to be backed up.
        """
        logger.info("Performing periodic backup...")
        self.create_backup(data, backup_type="periodic")

    def critical_backup(self, data: dict) -> None:
        """
        Performs an immediate critical backup, typically triggered by critical system events.

        :param data: The data to be backed up.
        """
        logger.warning("Performing critical backup due to a critical event...")
        self.create_backup(data, backup_type="critical")

    def copy_to_backup_location(self, source_directory: str, target_directory: str) -> None:
        """
        Copies the trading bot's entire directory to a secondary backup location.

        :param source_directory: Directory of the trading bot's current state.
        :param target_directory: Directory where the full backup will be copied to.
        """
        try:
            shutil.copytree(source_directory, target_directory, dirs_exist_ok=True)
            logger.info(f"Copied trading bot directory from {source_directory} to {target_directory}")
        except Exception as e:
            logger.error(f"Failed to copy trading bot directory: {e}")

    def monitor_system_health(self, data: dict) -> None:
        """
        Monitors the health of the trading bot system and triggers a critical backup if issues are detected.
        """
        logger.info("Monitoring system health...")
        # Placeholder for actual system health check logic
        # If an issue is detected, trigger critical backup
        system_health = self._check_system_health()

        if not system_health:
            logger.warning("System health deteriorated. Initiating critical backup...")
            self.critical_backup(data)

    def _check_system_health(self) -> bool:
        """
        Placeholder for checking system health.
        Implement actual checks here (e.g., memory, CPU, network usage).
        :return: True if system is healthy, False otherwise.
        """
        # Placeholder logic. Replace with actual checks.
        return True

    def schedule_periodic_backups(self, data: dict) -> None:
        """
        Schedules periodic backups at the configured frequency.
        """
        logger.info(f"Scheduling periodic backups every {self.backup_frequency} minutes.")
        next_backup_time = datetime.now() + self.backup_frequency

        while True:
            current_time = datetime.now()
            if current_time >= next_backup_time:
                self.periodic_backup(data)
                next_backup_time = current_time + self.backup_frequency
            time.sleep(60)  # Sleep for 1 minute before rechecking
