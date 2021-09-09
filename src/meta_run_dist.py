"""
Code framework adapted from https://github.com/TonghanWang/ROMA
"""

# decentralized, self-interested agents
# in training, sharing scheme is represented as a gaussian distribution

import datetime
import os
import pprint
import threading
from os.path import dirname, abspath
from types import SimpleNamespace as SN

import torch
import torch as th
from torch.distributions.uniform import Uniform

from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from controller.ac_z_separate_controller import ZACSeparateMAC
from controller.d_separate_controller import SeparateMAC
from learner import REGISTRY as le_REGISTRY
from runner import REGISTRY as r_REGISTRY
from utils.logging import Logger


# torch.autograd.set_detect_anomaly(True)
def run(_run, _config, _log):
    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    if args.use_cuda:
        th.cuda.set_device(args.device_num)

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        args.latent_role_direc = os.path.join(tb_exp_direc, "{}").format('latent_role')
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_distance_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


# meta-training runner
def run_distance_sequential(args, logger):
    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]

    # Default/Base scheme
    scheme = {
        # "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "z_q": {"vshape": (args.latent_relation_space_dim,), "group": "agents", "dtype": th.float},
        "z_p": {"vshape": (args.latent_relation_space_dim,), "group": "agents", "dtype": th.float},
        "z_idx": {"vshape": (1,), "dtype": th.int64},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "rewards": {"vshape": env_info["reward_shape"], },
        "redistributed_rewards": {"vshape": env_info["reward_shape"], },
        "adjacent_agents": {"vshape": env_info["adjacent_agents_shape"], "group": "agents", },
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    train_phase = "pretrain"

    # Setup multiagent controllers
    mac = SeparateMAC(buffer.scheme, groups, args, train_phase)
    z_mac = ZACSeparateMAC(buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args, z_mac=z_mac)

    if args.use_cuda:
        learner.cuda()

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    ############################### meta-training agent models on proposed sharing scheme #########################

    if args.load_pretrained_model:  # loading pretrained model
        logger.console_logger.info("Loading pretrained model")

        learner.load_models(args.pretrained_model_load_path)
    else:
        logger.console_logger.info("Beginning pretrained for {} timesteps".format(args.total_pretrain_steps))

        # sharing schemes proposed
        if args.deterministic_pretrained_tasks:
            tasks = gen_uniform_tasks(args)
        elif args.hardcoded_pretrained_tasks:
            tasks = get_hardcoded_tasks(args)
        else:
            tasks = sample_dist_norm(args)

        # meta-training
        while runner.t_env <= args.total_pretrain_steps:
            for idx in range(len(tasks)):
                # Run for a whole episode at a time
                z_q_cp = tasks[idx][0].clone()
                z_p_cp = tasks[idx][1].clone()
                episode_batch = runner.run(z_q=z_q_cp, z_p=z_p_cp, z_idx=[[idx]], test_mode=False,
                                           train_phase=train_phase)
                buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

            # Execute test runs once in a while
            n_test_runs = max(1, args.test_nepisode // runner.batch_size)
            if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
                last_test_T = runner.t_env
                for z_q, z_p in tasks:
                    for _ in range(n_test_runs):
                        runner.run(z_q=z_q, z_p=z_p, z_idx=[[idx]], test_mode=True, train_phase=train_phase)

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

            if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
                model_save_time = runner.t_env
                save_path = os.path.join(args.local_results_path, "pretrained_models", args.unique_token,
                                         str(runner.t_env))
                os.makedirs(save_path, exist_ok=True)
                logger.console_logger.info("Saving models to {}".format(save_path))

                # learner should handle saving/loading -- delegate actor save/load to mac,
                # use appropriate filenames to do critics, optimizer states
                learner.save_models(save_path)
            episode += args.batch_size_run


    ############################### optimizing sharing structure #########################

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.total_z_training_steps * args.env_steps_every_z))

    # reinitialize training parameters
    episode = 0
    runner.t_env = 0
    last_log_T = 0
    z_train_steps = 0
    env_steps_threshold = 0

    # initialize sharing scheme actor and its optimizer
    z_p, z_q = sample_dist_norm(args, train=True)  # initial sharing scheme
    device = "cpu" if args.buffer_cpu_only else args.device
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device=device)

    train_phase = "train"
    while z_train_steps <= args.total_z_training_steps:

        env_steps_threshold += args.env_steps_every_z
        mac.init_epsilon_schedule(train_phase)
        train_data = {"z_q": z_q, "z_p": z_p}
        actor_train_batch = {}
        for k, v in train_data.items():
            if not isinstance(v, th.Tensor):
                v = th.tensor(v, dtype=th.float, device=device)
            else:
                v.to(device)
            actor_train_batch.update({k: v})
        z_p, _, z_q, _ = z_mac.forward(actor_train_batch)

        while runner.t_env <= env_steps_threshold:
            episode_batch = runner.run(z_q, z_p, test_mode=False, train_phase=train_phase)
            buffer.insert_episode_batch(episode_batch)

            if buffer.can_sample(args.batch_size):
                episode_sample = buffer.sample(args.batch_size)
                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]
                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

            if (runner.t_env - last_log_T) >= args.log_interval:
                logger.log_stat("episode", episode, runner.t_env)
                logger.print_recent_stats()
                last_log_T = runner.t_env

        # sample for optimizing z critic
        episode_returns = []
        for _ in range(args.z_sample_runs):
            episode_returns.append(
                runner.run(z_q, z_p, test_mode=True, sample_return_mode=True, train_phase=train_phase))

        # generate data for training z critic
        data = {"z_p": z_p, "z_q": z_q, "evals": episode_returns}
        critic_train_data = {}
        for k, v in data.items():
            if not isinstance(v, th.Tensor):
                v = th.tensor(v, dtype=th.float, device=device)
            else:
                v.to(device)
            critic_train_data.update({k: v})

        # train z critic
        critic_train_data["evals"] = torch.sum(critic_train_data["evals"], dim=0) / args.z_sample_runs
        learner.z_critic_train(critic_train_data, z_train_steps)

        # update z actor
        learner.z_actor_train()


        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:
            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(z_q, z_p, test_mode=True, train_phase=train_phase)

         # in the desperation to understand what is going on

        def softmax(vector):
            import math
            e = [math.exp(x) for x in vector]
            return [x / sum(e) for x in e]

        def distance(a, b):
            import numpy
            ret = numpy.linalg.norm(a - b, ord=2)
            return ret

        z_q_cp = z_q.clone().detach().cpu().numpy()
        z_p_cp = z_p.clone().detach().cpu().numpy()
        dist = []
        for giver in range(args.n_agents):
            dist.append(softmax([- distance(z_q_cp[giver], z_p_cp[receiver]) for receiver in range(args.n_agents)]))

        for receiver in range(args.n_agents):
            for giver in range(args.n_agents):
                logger.log_stat("giver agent {} to receiver agent {}".format(giver, receiver), dist[giver][receiver],
                                runner.t_env)

        t_max = args.env_steps_every_z * args.total_z_training_steps
        logger.log_vec(tag="z_p", mat=z_p, global_step=runner.t_env)
        logger.log_vec(tag="z_q", mat=z_q, global_step=runner.t_env)
        logger.console_logger.info("t_env: {} / {}".format(runner.t_env, t_max))

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            learner.save_models(save_path, train=True)  # also save z_critic and etc

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def sample_dist_norm(args, train=False, autograd=False):
    # [(z_q, z_p)]: z_q: [n_agents][space_dim]
    l = torch.tensor([args.latent_relation_space_lower_bound], dtype=torch.float)
    u = torch.tensor([args.latent_relation_space_upper_bound], dtype=torch.float)
    if autograd:  # weird fix
        l.to(args.device)
        u.to(args.device)
    d = Uniform(low=l, high=u)

    dim_num = args.latent_relation_space_dim
    size = torch.Size([args.n_agents, dim_num])

    # sample for meta-training
    if not train:
        return [(d.sample(size).view(1, args.n_agents, dim_num),
                 d.sample(size).view(1, args.n_agents, dim_num)) for _ in range(args.pretrained_task_num)]

    # sample for training
    z = (d.sample(size).view(args.n_agents, dim_num),
         d.sample(size).view(args.n_agents, dim_num))
    z[0].requires_grad = autograd
    z[1].requires_grad = autograd

    return z

# test on simple hardcoded example
def get_hardcoded_tasks(args):
    tasks = []
    for z_q, z_p in zip(args.hardcoded_tasks_zq, args.hardcoded_tasks_zp):
        tasks.append((torch.tensor(z_q, dtype=torch.float).view(1, args.n_agents, args.latent_relation_space_dim),
                      torch.tensor(z_p, dtype=torch.float).view(1, args.n_agents, args.latent_relation_space_dim)))
    return tasks


# test to see if guaranteed uniform task distribution improve performances
def gen_uniform_tasks(args):
    lower = args.latent_relation_space_lower_bound
    upper = args.latent_relation_space_upper_bound
    divides = args.div_num
    dim_num = args.latent_relation_space_dim
    tasks = gen_uniform_tasks_dim(dim_num, 0, divides, lower, upper)
    tasks = [torch.tensor(task, dtype=torch.float) for task in tasks]

    from itertools import combinations_with_replacement
    tasks = list(combinations_with_replacement(tasks, args.n_agents))
    return list(combinations_with_replacement([torch.stack(task, dim=0).unsqueeze(dim=0) for task in tasks], 2))


def gen_uniform_tasks_dim(dim_num, cur_dim, div_num, lower, upper):
    if cur_dim == dim_num:
        return [[]]
    old_ret = gen_uniform_tasks_dim(dim_num, cur_dim + 1, div_num, lower, upper)
    ret = []
    for div in range(0, div_num + 1):
        new_val = lower + div * (upper - lower) / div_num
        for l in old_ret:
            t = l.copy()
            t.append(new_val)
            ret.append(t)
    return ret


def args_sanity_check(config, _log):
    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
