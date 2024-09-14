import json
import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StrategySettingsManager:
    def __init__(self, settings_directory: str = "strategy_settings"):
        """
        Initializes the StrategySettingsManager with a directory to store the settings.

        :param settings_directory: Directory where the settings files will be saved.
        """
        self.settings_directory = settings_directory
        if not os.path.exists(settings_directory):
            os.makedirs(settings_directory)
            logger.info(f"Created settings directory: {settings_directory}")
        else:
            logger.info(f"Using existing settings directory: {settings_directory}")

    def save_settings(self, template_name: str, settings: Dict[str, Any]) -> None:
        """
        Saves the given settings to a JSON file.

        :param template_name: Name of the template (filename) under which to save the settings.
        :param settings: Dictionary containing the strategy settings to be saved.
        """
        file_path = os.path.join(self.settings_directory, f"{template_name}.json")
        try:
            with open(file_path, 'w') as file:
                json.dump(settings, file, indent=4)
            logger.info(f"Settings saved successfully under the template: {template_name}")
        except Exception as e:
            logger.error(f"Failed to save settings under the template {template_name}: {e}")
            raise

    def load_settings(self, template_name: str) -> Dict[str, Any]:
        """
        Loads settings from a JSON file.

        :param template_name: Name of the template (filename) from which to load the settings.
        :return: Dictionary containing the loaded strategy settings.
        """
        file_path = os.path.join(self.settings_directory, f"{template_name}.json")
        if not os.path.exists(file_path):
            error_message = f"Template {template_name} does not exist in the settings directory."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            with open(file_path, 'r') as file:
                settings = json.load(file)
            logger.info(f"Settings loaded successfully from the template: {template_name}")
            return settings
        except Exception as e:
            logger.error(f"Failed to load settings from the template {template_name}: {e}")
            raise

    def list_available_templates(self) -> Dict[str, str]:
        """
        Lists all available templates in the settings directory.

        :return: A dictionary with template names as keys and file paths as values.
        """
        templates = {}
        try:
            for file_name in os.listdir(self.settings_directory):
                if file_name.endswith(".json"):
                    template_name = file_name.replace(".json", "")
                    templates[template_name] = os.path.join(self.settings_directory, file_name)
            logger.info(f"Available templates: {list(templates.keys())}")
        except Exception as e:
            logger.error(f"Failed to list templates in the settings directory: {e}")
            raise

        return templates

    def delete_template(self, template_name: str) -> None:
        """
        Deletes a specific settings template.

        :param template_name: Name of the template (filename) to delete.
        """
        file_path = os.path.join(self.settings_directory, f"{template_name}.json")
        if not os.path.exists(file_path):
            error_message = f"Template {template_name} does not exist and cannot be deleted."
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        try:
            os.remove(file_path)
            logger.info(f"Template {template_name} has been deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete the template {template_name}: {e}")
            raise

    def update_template(self, template_name: str, updated_settings: Dict[str, Any]) -> None:
        """
        Updates an existing settings template.

        :param template_name: Name of the template (filename) to update.
        :param updated_settings: Dictionary containing the updated settings.
        """
        existing_settings = self.load_settings(template_name)
        existing_settings.update(updated_settings)
        self.save_settings(template_name, existing_settings)
        logger.info(f"Template {template_name} has been updated successfully.")

    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """
        Validates the given settings to ensure they conform to expected structure.

        :param settings: Dictionary containing the strategy settings to validate.
        :return: Boolean indicating if the settings are valid.
        """
        required_keys = ['risk_percentage', 'leverage', 'spread_thresholds', 'atr_multiplier', 'liquidity_filters', 'scaling_options', 'hedging_settings']
        for key in required_keys:
            if key not in settings:
                logger.error(f"Validation failed: Missing required setting {key}.")
                return False
        logger.info("Settings validation passed.")
        return True

    def load_or_default(self, template_name: str, default_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Loads a settings template, falling back to default settings if the template does not exist.

        :param template_name: The name of the template to load.
        :param default_settings: Optional dictionary of default settings.
        :return: The loaded or default settings.
        """
        try:
            settings = self.load_settings(template_name)
        except FileNotFoundError:
            logger.warning(f"Template {template_name} not found. Using default settings.")
            settings = default_settings or {}
        return settings
