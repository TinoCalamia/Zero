"""Load Config Details."""
import ast
import os

import yaml

if not os.path.isdir("data"):
    os.mkdir("data")


class Setup:
    """Setup class."""

    def __init__(self, config_file):
        """Initialize class."""
        self.config_file = config_file
        self.config_dict = dict()
        self.yaml_config_raw = yaml.safe_load(open(self.config_file))

    def parse_yaml_config(self):
        """Parse a yml config file and create a config_dict class attribute."""
        ENVIRONMENTS = ["local", "dev", "staging", "live"]

        assert os.path.exists(
            self.config_file
        ), f"Config ini file {self.config_file} does not exist"
        yaml_config = self.yaml_config_raw
        env_config = yaml_config["environment_variables"]
        del yaml_config["environment_variables"]

        def set_attributes_from_environment_variables():
            """Read the environment variables and update the config_dict with them."""

            for env_variable in env_config:
                if env_variable.get("type") == "list":
                    self.config_dict[env_variable["config_name"]] = ast.literal_eval(
                        os.getenv(env_variable["env_name"])
                    )
                else:
                    self.config_dict[env_variable["config_name"]] = os.getenv(
                        env_variable["env_name"]
                    )

        def parse_yaml_file_section():
            """Parse the yaml_file with the appropriate section and region."""

            for key, value in yaml_config.items():
                if isinstance(value, dict):
                    assert set(value.keys()).issubset(
                        ENVIRONMENTS
                    ), f"{set(value.keys())-set(ENVIRONMENTS)} not in {ENVIRONMENTS} for the key '{key}'"
                    assert set(value.keys()) == set(
                        ENVIRONMENTS
                    ), f"{set(ENVIRONMENTS)-set(value.keys())} environment is missing for the key '{key}'"
                    for env_key, env_value in value.items():
                        self.config_dict[key] = value[self.config_dict["section"]]
                else:
                    self.config_dict[key] = value

        set_attributes_from_environment_variables()
        parse_yaml_file_section()

    def set_attributes(self):
        """Set attributes of setup class using config_dict key-value pairs."""
        for key in self.config_dict:
            setattr(self, key, self.config_dict[key])

    def load_and_process_config(self):
        """Load and parse YAML config file, preprocess some values."""

        # Set up model descriptors attributes
        self.parse_yaml_config()
        self.set_attributes()


object_detection_setup_config = Setup("utils/CONFIG.yaml")
object_detection_setup_config.load_and_process_config()
