import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import time
import os
import io
import pathlib
import hydra
import torch
import dill
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf
from omegaconf import open_dict
import pickle

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    elif isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    elif isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x



torch.backends.cudnn.enabled = True

'''final'''
# @hydra.main(
#     version_base=None,
#     config_path=str(pathlib.Path(__file__).parent.joinpath(
#         'diffusion_policy','Eval_Final_Clean_Eval_Table_1_config','lift')
#     )
# )
@hydra.main(
    version_base=None,)
def main(cfg):
    checkpoint = cfg.checkpoint
    task = cfg.task
    # print(f'Task: {task}')
    algo = cfg.algo
    n_envs = cfg.n_envs
    device = cfg.device
    attack = cfg.attack
    epsilon = cfg.epsilon
    view = cfg.view
    print(f"Running attack {attack} on {view} view")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if attack : 
        save_dir='/results/'
        output_dir = os.path.join(save_dir,f'{task}/{algo}/checkpoint_num{cfg.checkpoint_num}/epsilon_{cfg.epsilon}_view_{view}_targeted_{cfg.targeted}_Wall_RunTime_{timestamp}/')
    else:
        save_dir='/results/Unattacked/'
        output_dir = os.path.join(save_dir,f'{task}/{algo}/checkpoint_num{cfg.checkpoint_num}/Unattacked_Wall_RunTime_{timestamp}/')
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f'Path of outout dir: {output_dir}')
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg_loaded = payload['cfg']
    with open_dict(cfg):
        cfg.action_space = cfg_loaded.shape_meta.action.shape

    cls = hydra.utils.get_class(cfg_loaded._target_)
    workspace = cls(cfg_loaded, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    try:
        policy = workspace.model
    except AttributeError:
        policy = workspace.policy

    try:
        if cfg_loaded.training.use_ema:
            policy = workspace.ema_model
    except:
        pass

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    print(f'POLICY LOADED AND SET TO EVAL')

    with open_dict(cfg_loaded.task.env_runner):
        # if attack:
        print('---' * 50)
       
        cfg_loaded.task.env_runner['_target_'] = cfg.env_runner
        print(f'CHECK IF ENV RUNNER MATCHES CONFIG {cfg_loaded.task.env_runner._target_}')
        cfg_loaded.task.env_runner['n_envs'] = n_envs
        cfg_loaded.task.env_runner['n_test'] = cfg.n_test
        cfg_loaded.task.env_runner['n_test_vis'] = cfg.n_test_vis
        cfg_loaded.task.env_runner['n_train'] = cfg.n_train
        cfg_loaded.task.env_runner['n_train_vis'] = cfg.n_train_vis
        cfg_loaded.task.env_runner['fps'] = cfg.fps
        cfg_loaded.task.env_runner['crf'] = cfg.crf
        if cfg.max_steps is not None:
            cfg_loaded.task.env_runner['max_steps'] = cfg.max_steps
        try:
            cfg_loaded.task.env_runner['dataset_path'] = cfg.dataset_path
        except:
            print("No dataset path provided")
            pass
        print('---' * 20)

        print(OmegaConf.to_yaml(cfg))
        env_runner = hydra.utils.instantiate(cfg_loaded.task.env_runner,
                                             output_dir=output_dir)
        print(f'ENSURE CORRECT ENV RUNNER')
        final_full_cfg_path = os.path.join(output_dir, "final_eval_cfg.yaml")
        with open(final_full_cfg_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg_loaded))
        print(f"Saved full final eval config to {final_full_cfg_path}")
        print('---!!---' * 50)
        if attack and cfg.attack_type == 'patch':
            with open(cfg.patch_path, 'rb') as f:
                patch = CPU_Unpickler(f).load()
            print("Running adversarial Attack")
            patch = move_to_device(patch, torch.device(cfg.device))
            print(f"Evaluating UAP saved patch")
            # patch = pickle.load(open(cfg.patch_path, 'rb'),)
            runner_log = env_runner.run(policy, adversarial_patch=patch, cfg=cfg)
        else:
            runner_log = env_runner.run(policy, cfg=cfg)

        json_log = dict()
        for k, v in runner_log.items():
            if hasattr(v, "_path"):  # e.g. videos
                json_log[k] = v._path
            else:
                try:
                    json.dumps(v)  # test serializable
                    json_log[k] = v
                except TypeError:
                    json_log[k] = str(v)

        if "test/mean_score" in json_log:
            print("Test/mean_score:", json_log["test/mean_score"])
        if "train/mean_score" in json_log:
            print("Train/mean_score:", json_log["train/mean_score"])

        out_path = os.path.join(output_dir, "eval_log.json")
        with open(out_path, "w") as f:
            json.dump(json_log, f, indent=2, sort_keys=True)

        out_path_txt = os.path.join(output_dir, "eval_log.txt")
        with open(out_path_txt, "w") as f:
            for k, v in json_log.items():
                f.write(f"{k}: {v}\n")

        print(f"Saved evaluation log to {out_path}")

if __name__ == '__main__':
    main()