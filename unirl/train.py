import os
import numpy as np
import torch
from absl import app, flags, logging

from unirl_env.libs.utils.config import YamlConfig
from unirl_agent.rl_learn_single_gpu import learn

FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", False, "Debug mode")
flags.DEFINE_boolean("save_log", False, "Log to logfile or stdout")
flags.DEFINE_string("config_path", "./unirl/unirl_env/libs/configs", "configuration file path")

torch.autograd.set_detect_anomaly(True)


def main(argv):

    if FLAGS.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    
    if not FLAGS.log_dir:
        FLAGS.log_dir = "./"

    yaml_dict = {}
    with os.scandir(FLAGS.config_path) as entries:
        for entry in entries:
            if entry.is_file():
                splitted_entry_name = entry.name.split("_")
                if splitted_entry_name[-1] == "config.yml":
                    yaml_dict["_".join(splitted_entry_name[:-1])] = entry.path

    yaml_config = YamlConfig(yaml_dict)

    if FLAGS.save_log:
        os.makedirs(os.path.join(FLAGS.log_dir, 'log'), exist_ok=True)
        logging.get_absl_handler().use_absl_log_file(log_dir=os.path.join(FLAGS.log_dir, 'log'))

    for k, v in flags.FLAGS.__flags.items():
        logging.log(logging.INFO, '[{}:{}]'.format(k, v.value))
    for k, v in yaml_config.items():
        logging.log(logging.INFO, '[{}:{}]'.format(k, v))

    learn(yaml_config)


if __name__ == "__main__":
    
    app.run(main)