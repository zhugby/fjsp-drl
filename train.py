import copy
import json
import os
import random
import time
from collections import deque

import pandas as pd
import torch
import numpy as np

import PPO_model
from env import make_fjsp_env
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_tensorboard_writer(enabled, save_path, train_paras):
    if not enabled:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("TensorBoard logging is enabled, but tensorboard is not installed.")
        print("Install it with: pip install tensorboard")
        print("Training will continue without TensorBoard logging.")
        return None

    log_dir = os.path.join(save_path, "tensorboard")
    writer = SummaryWriter(log_dir=log_dir, comment=train_paras.get("viz_name", ""))
    print("TensorBoard logging enabled: ", log_dir)
    return writer


def is_all_done(dones):
    """
    将环境返回的 done 标志统一转换为 Python bool。

    dones 可能是：
    1. torch.Tensor
    2. numpy.ndarray
    3. Python bool
    """
    if torch.is_tensor(dones):
        return bool(dones.all().item())

    if isinstance(dones, np.ndarray):
        return bool(dones.all())

    return bool(dones)


def to_float(value):
    """
    将标量 tensor / numpy 标量 / Python 数值统一转成 float。
    """
    if torch.is_tensor(value):
        return float(value.item())

    if isinstance(value, np.ndarray):
        return float(value.item())

    return float(value)


def write_training_results(save_path, str_time, train_paras, valid_results, valid_results_100):
    """
    训练结束后统一写入验证结果。

    修复点：
    原代码一开始创建 ExcelWriter 后 close 了，
    后面又继续使用同一个 writer，会导致 writer 已关闭的问题。
    """
    iterations = np.arange(
        train_paras["save_timestep"],
        train_paras["max_iterations"] + 1,
        train_paras["save_timestep"]
    )

    # 写平均验证结果
    ave_path = "{0}/training_ave_{1}.xlsx".format(save_path, str_time)

    data_ave = pd.DataFrame({
        "iterations": iterations[:len(valid_results)],
        "res": valid_results
    })

    data_ave.to_excel(ave_path, sheet_name="Sheet1", index=False)

    # 写每个验证实例的结果
    detail_path = "{0}/training_100_{1}.xlsx".format(save_path, str_time)

    if len(valid_results_100) > 0:
        data_100_array = np.array(
            torch.stack(valid_results_100, dim=0).to("cpu")
        )

        column = [i_col for i_col in range(data_100_array.shape[1])]
        data_detail = pd.DataFrame(data_100_array, columns=column)
        data_detail.insert(0, "iterations", iterations[:len(valid_results_100)])
    else:
        data_detail = pd.DataFrame(columns=["iterations"])

    data_detail.to_excel(detail_path, sheet_name="Sheet1", index=False)

    print("Training average results saved to: ", ave_path)
    print("Training detail results saved to: ", detail_path)


def main():
    # PyTorch initialization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    print("PyTorch device: ", device.type)

    torch.set_printoptions(
        precision=None,
        threshold=np.inf,
        edgeitems=None,
        linewidth=None,
        profile=None,
        sci_mode=False
    )

    # Load config and init objects
    with open("./config.json", "r") as load_f:
        load_dict = json.load(load_f)

    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]

    env_paras["device"] = device
    model_paras["device"] = device

    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]

    model_paras["actor_in_dim"] = (
        model_paras["out_size_ma"] * 2
        + model_paras["out_size_ope"] * 2
    )
    model_paras["critic_in_dim"] = (
        model_paras["out_size_ma"]
        + model_paras["out_size_ope"]
    )

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]

    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    memories = PPO_model.Memory()
    model = PPO_model.PPO(
        model_paras,
        train_paras,
        num_envs=env_paras["batch_size"]
    )

    # Create an environment for validation
    env_valid = get_validate_env(env_valid_paras)

    maxlen = 1
    best_models = deque()
    makespan_best = float("inf")

    # Generate save path
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = "./save/train_{0}".format(str_time)
    os.makedirs(save_path, exist_ok=True)

    # Use TensorBoard to visualize the training process
    tb_writer = create_tensorboard_writer(
        train_paras["viz"],
        save_path,
        train_paras
    )

    valid_results = []
    valid_results_100 = []

    # Start training iteration
    start_time = time.time()
    env = None

    for i in range(1, train_paras["max_iterations"] + 1):
        # Replace training instances every x iteration
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # mathcal{B} instances use consistent operations to speed up training
            nums_ope = [
                random.randint(opes_per_job_min, opes_per_job_max)
                for _ in range(num_jobs)
            ]

            case = CaseGenerator(
                num_jobs,
                num_mas,
                opes_per_job_min,
                opes_per_job_max,
                nums_ope=nums_ope
            )

            env = make_fjsp_env(case=case, env_paras=env_paras)

            # 关键修复：
            # Gym 环境必须先 reset，再 step。
            env.reset()

            print(
                "num_job: ",
                num_jobs,
                "\tnum_mas: ",
                num_mas,
                "\tnum_opes: ",
                sum(nums_ope)
            )

        # Get state and completion signal
        state = env.state
        dones = env.done_batch
        done = is_all_done(dones)

        last_time = time.time()

        # Schedule in parallel
        # 关键修复：
        # 原来是 while ~done，这是按位取反，不是逻辑取反。
        # 应该使用 while not done。
        while not done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)

            state, rewards, dones = env.step(actions)

            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)

            done = is_all_done(dones)

        print("spend_time: ", time.time() - last_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")

        # 当前 batch 跑完后 reset，供下一轮复用。
        env.reset()

        # Update policy
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)

            loss_value = to_float(loss)
            reward_value = to_float(reward)

            print(
                "reward: ",
                "%.3f" % reward_value,
                "; loss: ",
                "%.3f" % loss_value
            )

            memories.clear_memory()

            if tb_writer is not None:
                tb_writer.add_scalar("train/reward", reward_value, i)
                tb_writer.add_scalar("train/loss", loss_value, i)

        # Validate policy
        if i % train_paras["save_timestep"] == 0:
            print("\nStart validating")

            vali_result, vali_result_100 = validate(
                env_valid_paras,
                env_valid,
                model.policy_old
            )

            vali_value = to_float(vali_result)

            valid_results.append(vali_value)
            valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_value < makespan_best:
                makespan_best = vali_value

                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    if os.path.exists(delete_file):
                        os.remove(delete_file)

                save_file = "{0}/save_best_{1}_{2}_{3}.pt".format(
                    save_path,
                    num_jobs,
                    num_mas,
                    i
                )

                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

                print("Best model saved to: ", save_file)
                print("Best makespan: ", makespan_best)

            if tb_writer is not None:
                tb_writer.add_scalar("valid/makespan", vali_value, i)

    # Save the data of training curve to files
    write_training_results(
        save_path=save_path,
        str_time=str_time,
        train_paras=train_paras,
        valid_results=valid_results,
        valid_results_100=valid_results_100
    )

    if tb_writer is not None:
        tb_writer.close()

    print("total_time: ", time.time() - start_time)


if __name__ == "__main__":
    main()