import yaml

# Load the YAML configuration file
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_config(config_path):
    config = load_config(config_path)

    # Resolve environment variables or placeholders like ${base_dir}
    for key, value in config["data_path"].items():
        #for var_name in config["variables"]:
        config["data_path"][key] = \
            value.replace("${base_dir}", config["base_dir"]).\
            replace("${GCM}", config["GCM"]).\
            replace("${FIRST_YEAR_3km}", config["FIRST_YEAR_3km"]).\
            replace("${LAST_YEAR_3km}", config["LAST_YEAR_3km"]).\
            replace("${FIRST_YEAR_12km}", config["FIRST_YEAR_12km"]).\
            replace("${LAST_YEAR_12km}", config["LAST_YEAR_12km"]).\
            replace("${MLEXP}", config["MLEXP"]) #.\
            #replace("${variables}", config["variables"])
    config["output_dir"] = config["output_dir"].\
            replace("${base_dir}", config["base_dir"])

    return config
