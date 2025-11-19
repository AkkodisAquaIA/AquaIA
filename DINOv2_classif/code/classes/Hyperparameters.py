import configparser

class Hyperparameters:
    def __init__(self, file_path):
        """
        Initialize the Hyperparameters class and load parameters from an INI file.
        
        Args:
            file_path (str): Path to the INI file containing hyperparameters.
        """
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve key case
        config.read(file_path)
        self.parameters = {section: dict(config[section]) for section in config.sections()}

    def get_parameters(self):
        """
        Get all hyperparameters as a dictionary.
        
        Returns:
            dict: Dictionary of all hyperparameters grouped by section.
        """
        return self.parameters

    def __repr__(self):
        """Returns a string representation of the Hyperparameters instance."""
        return f"Hyperparameters({self.parameters})"

    def save_to_ini(self, file_path):
        """
        Saves hyperparameters to an INI file.
        
        Args:
            file_path (str): The path to the INI file where hyperparameters will be saved.
        """
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve key case on saving as well
        for section, params in self.parameters.items():
            config[section] = {k: str(v) for k, v in params.items()}
        with open(file_path, 'w') as configfile:
            config.write(configfile)
