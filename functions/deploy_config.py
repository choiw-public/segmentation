import importlib as imp
from bunch import Bunch
import os


def deploy(args):
    phase = args.phase
    config_path = ".".join(['models', args.model_name, "config", "config_%s" % phase])
    config = imp.import_module(config_path).config
    config["model_dir"] = os.path.join('./models', args.model_name)
    config['phase'] = phase
    model_fn_path = ".".join(['models', args.model_name, 'model_fn'])
    config['model_fn'] = imp.import_module(model_fn_path).ModelFunction
    if phase == "train":
        config["training"] = True
    else:
        config["training"] = False
        config["train_tfrecord3"] = None
        config["train_tfrecord2"] = None
        config["eval_log_dir"] = "/".join(["./model", "eval_metric"])
        config["batch_size"] = 1
        if phase == "eval":
            config["eval_log_dir"] = "/".join(["./model", "eval_metric"])
            os.makedirs(config["eval_log_dir"], exist_ok=True)
        elif phase == "vis":
            config["img_dir"] = config["data_dir"]
            if config["data_type"] == "image":
                config["vis_result_dir"] = "/".join(["./model", "vis_results", "image"])
            elif config["data_type"] == "video":
                config["vis_result_dir"] = "/".join(["./model", "vis_results", "video"])
            os.makedirs(config["vis_result_dir"], exist_ok=True)
        else:
            raise ValueError('Unexpected phase')
    config = Bunch(config)
    return config
