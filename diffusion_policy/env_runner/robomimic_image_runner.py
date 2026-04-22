import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.utils.attack_utils import optimize_linear, clip_perturb

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from pathlib import Path
import random

# for deterministic results
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = False


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self,
                 output_dir,
                 dataset_path,
                 shape_meta: dict,
                 n_train=10,
                 n_train_vis=3,
                 train_start_idx=0,
                 n_test=22,
                 n_test_vis=6,
                 test_start_seed=10000,
                 max_steps=400,
                 n_obs_steps=2,
                 n_action_steps=8,
                 render_obs_key='agentview_image',
                 fps=10,
                 crf=22,
                 past_action=False,
                 abs_action=False,
                 tqdm_interval_sec=5.0,
                 n_envs=None,
                 ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        try:
            env_meta = FileUtils.get_env_metadata_from_dataset(
                dataset_path)
        except FileNotFoundError:
            # append the relative path to the dataset path
            directory_path = Path(__file__).parent.parent.parent
            dataset_path = os.path.join(directory_path, dataset_path)
            print(f"New dataset path: {dataset_path}")
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta,
                enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        print(f'num envs {n_envs}')
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                            enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                        enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy,adversarial_patch=None, cfg=None,save_pkl=False):
        # print(f"CHECK n_vis, inside run of robomimic env :{cfg.n_vis}")
        device = policy.device
        dtype = policy.dtype
        env = self.env
        # torch.use_deterministic_algorithms(True)

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        observations = []
        # randomly select envs to visualize from the n_envs
        vis_envs = np.random.choice(n_envs, cfg.n_vis, replace=False)
        # try:
        #     obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        # except:
        #     obs_encoder = policy.obs_encoder
        # image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
        policy.eval()
        # set_seed(1024)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                          args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx + 1}/{n_chunks}",
                             leave=False, mininterval=self.tqdm_interval_sec)
            if cfg.view == 'both':
                views = ['agentview_image', 'robot0_eye_in_hand_image']
            elif cfg.view == 'None':
                views = []
            else:
                views = [cfg.view]
            # if save_pkl:
            #     obs_ls=[]
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                                                 :, -(self.n_obs_steps - 1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))
                # print(f"Min and Max of obs_dict: {torch.min(obs_dict['robot0_eye_in_hand_image'])}, {torch.max(obs_dict['robot0_eye_in_hand_image'])}")
                clean_obs_dict = obs_dict.copy()
                random_obs_dict = obs_dict.copy()
                if adversarial_patch is not None:
                    for view in views:
                        # print(f'Attacking view{view}')
                        obs_dict[view] = obs_dict[view] + adversarial_patch[view].to(device)
                        obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
                    # maximum_perturbation = torch.max(torch.abs(adversarial_patch))
                    # random_perturbation = torch.rand_like(obs_dict[cfg.view]) * 2 * maximum_perturbation - maximum_perturbation
                    # random_obs_dict[cfg.view] = random_obs_dict[cfg.view] + random_perturbation
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                # run policy
                with torch.no_grad():
                    # calculate the l2 distance clean and obs_dict
                    clean_action_dict = policy.predict_action(clean_obs_dict)
                    action_dict = policy.predict_action(obs_dict)
                    # clean_output = policy.predict_action(clean_obs_dict, return_latent=True)['output']
                    # output = policy.predict_action(obs_dict, return_latent=True)['output']
                    # latents = policy.predict_action(obs_dict, return_latent=True)
                    # clean_latents = policy.predict_action(clean_obs_dict, return_latent=True)
                    # # print("L2 distance between clean and adversarial output: ", torch.norm(clean_output - output, p=2) / output.shape[0])
                    # print(latents['latent'])
                    # print(clean_latents['latent'])
                    # print("L2 distance between clean and adversarial latents: ", torch.norm(clean_latents['latent'][0].float() - latents['latent'][0].float(), p=2) / latents['latent'][0].shape[0])
                    # random_action_dict = policy.predict_action(random_obs_dict)
                '''uncommenting for TH'''
                observations.append(obs_dict['robot0_eye_in_hand_image'][vis_envs].detach().cpu().numpy())
                # print(f"observations appending has shape {observations[0].shape}")
                # device_transfer
                try:
                    np_action_dict = dict_apply(action_dict,
                                                lambda x: x.detach().to('cpu').numpy())
                    # np_clean_action_dict = dict_apply(clean_action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    # np_random_action_dict = dict_apply(random_action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action']
                    # clean_actions = np_clean_action_dict['action']
                    # random_actions = np_random_action_dict['action']
                except AttributeError:
                    action = action_dict['action'].detach().to('cpu').numpy()
                # l1 distance between actions and clean actions before perturbation
                # for i in range(action.shape[-1]):
                #     print(f"l1_distance_{i}: {torch.norm(action_dict['action'][:,:,i] - clean_action_dict['action'][:,:,i], p=1) / action.shape[0]}")
                if cfg.targeted and cfg.log:
                    pass
                    #COMMENTED FOR IBC
                    # try:
                    #     # log the l1 distance between the predicted action and the target action per environment
                    #     target_action = clean_action_dict['action'].to('cpu') + adversarial_patch['perturbations'].to(
                    #         'cpu')
                    #     for i in range(target_action.shape[-1]):
                    #         wandb.log({f"l1_distance_{i}": torch.norm(
                    #             action_dict['action'][:, :, i].to('cpu') - target_action[:, :, i], p=1) /
                    #                                        target_action.shape[0]})
                    # except KeyError:
                    #     pass
                # features = np_action_dict['features']
                # clean_features = np_clean_action_dict['features']
                # random_features = np_random_action_dict['features']
                # features = np.mean(features, axis=0)
                # actions = np.mean(action, axis=0)
                # clean_actions = np.mean(clean_actions, axis=0)
                # clean_features = np.mean(clean_features, axis=0)
                # random_features = np.mean(random_features, axis=0)
                # random_actions = np.mean(random_actions, axis=0)
                # l2 distance between features
                # diff = np.linalg.norm(features - clean_features)
                # diff_actions = np.linalg.norm(actions - clean_actions)
                # diff_random = np.linalg.norm(random_features - clean_features)
                # diff_random_actions = np.linalg.norm(random_actions - clean_actions)
                # print(f"Diff: {diff}, Random Diff: {diff_random}")
                # print(f"Diff Actions: {diff_actions}, Random Diff Actions: {diff_random_actions}")
                # if cfg.log:
                #     wandb.log({"Diff": diff, "Random Diff": diff_random,
                #                 "Diff Actions": diff_actions, "Random Diff Actions": diff_random_actions})
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)


                obs, reward, done, info = env.step(env_action)
                # obs_ls.append(obs)
                done = np.all(done)
                past_action = action
                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # visualize the video and save to wandb
        if cfg.save_video and cfg.log:
            for i in range(cfg.n_vis):
                ims = []
                fig, ax = plt.subplots()
                for j in range(len(observations)):
                    try:
                        # print(observations[j][i][1].shape)
                        im = observations[j][i][1].transpose(1, 2, 0)
                    except IndexError:
                        im = observations[j][i][0].transpose(1, 2, 0)
                    ims.append([ax.imshow(im)])
                save_path = Path(cfg.patch_path).parent
                save_path = save_path.joinpath(f'{cfg.exp_name}_vis_{i}.gif')
                print(f"Saving video to {save_path}")

                if os.path.exists(save_path):
                    os.remove(save_path)

                ani = ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
                ani.save(str(save_path), writer='pillow')
                plt.close(fig)
                '''UNCOMMENTING TO CHECK SCORE AT EVAL WITH VIS'''
                wandb.log({f'{cfg.exp_name}_vis_{i}': wandb.Image(str(save_path))})

        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f'sim_max_reward_{seed}'] = max_reward

        #     # visualize sim
        #     video_path = all_video_paths[i]
        #     if video_path is not None:
        #         sim_video = wandb.Video(video_path)
        #         log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + 'mean_score'
            value = np.mean(value)
            log_data[name] = value
        # save_pkl_dir='/home/ak/Documents/Adversarial_diffusion_policy/pre_trained_checkpoints/square/'
        # pickle.dump(obs_ls, save_pkl_dir.joinpath('square_obs.pkl').open('wb'))

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3:3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction


