"""PPO on the batched MuJoCo Warp env (wam/warp_env.py): milestone 4 of the port (log.md 2026-11-28).

CleanRL's ppo_continuous_action (as in train_ppo.py: same Agent, GAE, clipped objective, observation normalisation,
return-based reward scaling with clipping at 10) rewritten for tensors that already live on the GPU: every rollout
step covers all `--num-envs` worlds at once. Episodes end on success, spill or 250 steps (truncation treated as
termination, as train_ppo.py does).

Saves runs/WamFamilyContactFK-v0__<exp>__<seed>__w/<exp>.cleanrl_model and obs_rms.npz in train_ppo.py's
formats, so `scripts/eval_policy.py WamFamilyContactFK-v0 <seed> <task> ... --exp <exp>` scores it in C MuJoCo.
Progress goes to progress.csv in the same folder. Every --save-every iterations the full training state goes to
ckpt.pt; rerunning the same command resumes from it (a preempted, requeued job continues instead of restarting), and
exits at once if the final model already exists. On resume every world starts a fresh episode.

    python scripts/train_ppo_warp.py --seed 1 --exp warp1 [--num-envs 4096 --num-steps 32 --total-timesteps 100000000 ...]
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

sys.path.insert(0, ".")
from wam.warp_env import WarpFamilyEnv, start_pool  # noqa: E402

ENV_ID = "WamFamilyContactFK-v0"  # the C env with the same observation, reward and task, used to evaluate


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--exp", default="warp")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=32)
    p.add_argument("--total-timesteps", type=int, default=100_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=5)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--pool-size", type=int, default=20_000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-every", type=int, default=25, help="iterations between resumable checkpoints (ckpt.pt)")
    return p.parse_args()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """train_ppo.py's Agent, with the hidden width as a parameter (64 there)."""

    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.critic = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, hidden)),
                                    nn.Tanh(), layer_init(nn.Linear(hidden, 1), std=1.0))
        self.actor_mean = nn.Sequential(layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(), layer_init(nn.Linear(hidden, hidden)),
                                        nn.Tanh(), layer_init(nn.Linear(hidden, act_dim), std=0.01))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        probs = Normal(mean, torch.exp(self.actor_logstd.expand_as(mean)))
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class RunningMeanStd:
    """gymnasium's RunningMeanStd on the device (parallel-variance update over a batch)."""

    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = 1e-4

    def update(self, x):
        x = x.double()
        batch_mean, batch_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta, total = batch_mean - self.mean, self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total


def main():
    args = parse()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run = Path(f"runs/{ENV_ID}__{args.exp}__{args.seed}__w")
    if (run / f"{args.exp}.cleanrl_model").exists():
        print(f"{run} already finished", flush=True)
        return
    run.mkdir(parents=True, exist_ok=True)
    pool_path = Path(f"runs/warp_pool_F_{args.pool_size}_{args.seed}.npz")
    t0 = time.time()
    pool = start_pool(args.pool_size, seed=1_000_000 + 100_000 * args.seed, path=pool_path)
    print(f"start pool: {len(pool['qpos'])} starts ({time.time() - t0:.0f} s)", flush=True)

    env = WarpFamilyEnv(args.num_envs, pool, device=args.device, seed=args.seed)
    obs_dim, act_dim = 46, 4
    agent = Agent(obs_dim, act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    obs_rms, ret_rms = RunningMeanStd(obs_dim, device), RunningMeanStd((), device)
    returns_acc = torch.zeros(args.num_envs, device=device, dtype=torch.float64)

    def normalize(o):
        obs_rms.update(o)
        return ((o - obs_rms.mean.float()) / torch.sqrt(obs_rms.var.float() + 1e-8)).clamp(-10, 10)

    def scale_reward(r, done):
        nonlocal returns_acc
        returns_acc = returns_acc * args.gamma + r.double()
        ret_rms.update(returns_acc)
        returns_acc = returns_acc * (~done)
        return (r / torch.sqrt(ret_rms.var.float() + 1e-8)).clamp(-10, 10)

    N, T = args.num_envs, args.num_steps
    batch = N * T
    minibatch = batch // args.num_minibatches
    iterations = args.total_timesteps // batch
    obs_buf = torch.zeros(T, N, obs_dim, device=device)
    act_buf = torch.zeros(T, N, act_dim, device=device)
    logp_buf, rew_buf, done_buf, val_buf = (torch.zeros(T, N, device=device) for _ in range(4))

    global_step, first = 0, 1
    ckpt = run / "ckpt.pt"
    if ckpt.exists():
        c = torch.load(ckpt, map_location=device, weights_only=False)
        agent.load_state_dict(c["agent"]), optimizer.load_state_dict(c["optimizer"])
        obs_rms.mean, obs_rms.var, obs_rms.count = c["obs_rms"]
        ret_rms.mean, ret_rms.var, ret_rms.count = c["ret_rms"]
        global_step, first = c["global_step"], c["iteration"] + 1
        torch.set_rng_state(c["rng"]), env.gen.set_state(c["env_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(c["cuda_rng"])
        print(f"resumed from iteration {c['iteration']} ({global_step:,} steps)", flush=True)
    next_obs = normalize(env.reset())
    next_done = torch.zeros(N, device=device)
    new_log = not (run / "progress.csv").exists() or first == 1
    progress = open(run / "progress.csv", "w" if new_log else "a", newline="")
    log = csv.writer(progress)
    if new_log:
        log.writerow(["iteration", "global_step", "sps", "episodes", "success", "spill", "timeout", "valid_fraction",
                      "action_std_min", "action_std_mean", "entropy", "approx_kl", "value_loss", "policy_loss"])
    start, start_step = time.time(), global_step

    def save():
        torch.save(agent.state_dict(), run / f"{args.exp}.cleanrl_model")
        np.savez(run / "obs_rms.npz", mean=obs_rms.mean.cpu().numpy()[None], var=obs_rms.var.cpu().numpy()[None],
                 count=np.array([obs_rms.count]))

    def checkpoint(it):
        state = dict(agent=agent.state_dict(), optimizer=optimizer.state_dict(), iteration=it, global_step=global_step,
                     obs_rms=(obs_rms.mean, obs_rms.var, obs_rms.count), ret_rms=(ret_rms.mean, ret_rms.var, ret_rms.count),
                     rng=torch.get_rng_state(), env_rng=env.gen.get_state(),
                     cuda_rng=torch.cuda.get_rng_state() if device.type == "cuda" else None)
        torch.save(state, run / "ckpt.tmp")
        os.replace(run / "ckpt.tmp", ckpt)  # atomic: a preemption mid-write leaves the previous checkpoint intact

    for it in range(first, iterations + 1):
        for g in optimizer.param_groups:
            g["lr"] = (1.0 - (it - 1.0) / iterations) * args.learning_rate
        ended = {"n": 0, "success": 0.0, "spill": 0.0, "timeout": 0.0, "valid": 0.0}
        for step in range(T):
            global_step += N
            obs_buf[step], done_buf[step] = next_obs, next_done
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            val_buf[step], act_buf[step], logp_buf[step] = value.flatten(), action, logp
            raw_obs, reward, term, trunc, info = env.step(action)
            done = term | trunc
            rew_buf[step] = scale_reward(reward, done)
            next_obs, next_done = normalize(raw_obs), done.float()
            if done.any():
                ended["n"] += int(done.sum())
                for k, key in (("success", "success"), ("spill", "spill"), ("timeout", "timeout")):
                    ended[k] += float(info[key][done].float().sum())
                ended["valid"] += float(info["valid_fraction"][done & info["success"]].sum())

        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            adv = torch.zeros_like(rew_buf)
            last = torch.zeros(N, device=device)
            for t in reversed(range(T)):
                nonterminal = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
                nv = next_value if t == T - 1 else val_buf[t + 1]
                delta = rew_buf[t] + args.gamma * nv * nonterminal - val_buf[t]
                last = delta + args.gamma * args.gae_lambda * nonterminal * last
                adv[t] = last
            ret = adv + val_buf

        b_obs, b_act, b_logp = obs_buf.reshape(batch, obs_dim), act_buf.reshape(batch, act_dim), logp_buf.reshape(batch)
        b_adv, b_ret, b_val = adv.reshape(batch), ret.reshape(batch), val_buf.reshape(batch)
        for _ in range(args.update_epochs):
            perm = torch.randperm(batch, device=device)
            for s in range(0, batch, minibatch):
                mb = perm[s:s + minibatch]
                _, newlogp, entropy, newvalue = agent.get_action_and_value(b_obs[mb], b_act[mb])
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip_coef, 1 + args.clip_coef)).mean()
                newvalue = newvalue.view(-1)
                v_clipped = b_val[mb] + (newvalue - b_val[mb]).clamp(-args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((newvalue - b_ret[mb]) ** 2, (v_clipped - b_ret[mb]) ** 2).mean()
                loss = pg_loss - args.ent_coef * entropy.mean() + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        sps = int((global_step - start_step) / (time.time() - start))
        std = agent.actor_logstd.exp().detach()
        n = max(ended["n"], 1)
        row = [it, global_step, sps, ended["n"], ended["success"] / n, ended["spill"] / n, ended["timeout"] / n,
               ended["valid"] / max(ended["success"], 1), float(std.min()), float(std.mean()), float(entropy.mean().detach()),
               float(approx_kl), float(v_loss), float(pg_loss)]
        log.writerow(row)
        if it % 10 == 0 or it == iterations:
            print(f"it {it}/{iterations} step {global_step:,} sps {sps:,} | episodes {ended['n']} success {row[4]:.3f} "
                  f"spill {row[5]:.3f} timeout {row[6]:.3f} | std min {row[8]:.3f} | kl {row[11]:.4f}", flush=True)
        progress.flush()
        if args.save_every and it % args.save_every == 0:
            checkpoint(it)
    save()
    print(f"saved {run} ({(time.time() - start) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
