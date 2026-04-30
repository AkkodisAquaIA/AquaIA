import configparser
import platform

class Config:
    def __init__(self, path="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(path)

        self.system = platform.system()

    # --- helpers ---
    def get_bool(self, section, key):
        value = self.get_str(section, key).lower()
        if value in ("true", "1", "yes", "on", "oui"):
            return True
        elif value in ("false", "0", "no", "off", "non"):
            return False
        else:
            raise ValueError(f"[{section}] {key} doit être un booléen")

    def get_int(self, section, key):
        value = self.get_str(section, key)
        try:
            return int(value)
        except ValueError:
            raise ValueError(f"[{section}] {key} doit être un entier valide")

    def get_float(self, section, key):
        value = self.get_str(section, key)
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"[{section}] {key} doit être un réel valide")

    def get_str(self, section, key):
        return self.config.get(section, key)

    # --- computed values ---
    @property
    def PATH_USER(self):
        if self.system == "Windows":
            return self.get_str("paths", "WINDOWS")
        else:
            return self.get_str("paths", "LINUX")
        