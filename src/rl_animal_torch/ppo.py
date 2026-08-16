"""
The PPO update and the batch it reads. Neither piece knows about an environment, a
logger or a training loop, so a driver holding one environment and a driver holding
twenty-four fill the same buffer and call the same update.
"""

import numpy as np
import torch
import torch.nn.functional as F


def swap_and_flatten(array):
    """
    [steps, actors, ...] -> [actors * steps, ...]: every contiguous run of seq_len entries
    then belongs to one environment, which is what the recurrent update assumes.
    """
    shape = array.shape
    return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class Rollout:
    """
    One batch of experience, already flattened and on the training device.
    """

    def __init__(self, visual, vels, dones, actions, values, neglogpacs, returns, states):
        self.visual = visual
        self.vels = vels
        self.dones = dones
        self.actions = actions
        self.values = values
        self.neglogpacs = neglogpacs
        self.returns = returns
        self.states = states


class RolloutBuffer:
    """
    The steps of one epoch as the driver produced them, and the advantage pass that turns
    them into a Rollout. Every entry is [actors, ...] for one step, so a single
    environment is just the width-one case.
    """

    def __init__(self):
        self.steps = {
            name: []
            for name in (
                "visual",
                "vels",
                "rewards",
                "actions",
                "values",
                "dones",
                "neglogpacs",
                "states",
            )
        }

    def __len__(self):
        return len(self.steps["rewards"])

    def add(self, visual, vels, rewards, actions, values, dones, neglogpacs, states):
        """
        `dones` and `states` are the ones that held *before* the action was chosen: they
        are what the recurrent unrolling replays, not the outcome of the step.
        """
        self.steps["visual"].append(visual)
        self.steps["vels"].append(vels)
        self.steps["rewards"].append(rewards)
        self.steps["actions"].append(actions)
        self.steps["values"].append(values)
        self.steps["dones"].append(dones)
        self.steps["neglogpacs"].append(neglogpacs)
        self.steps["states"].append(states)

    def finish(self, last_values, final_dones, gamma, lam, seq_len, device):
        """
        Generalized advantage estimation over the collected steps, then the
        environment-major flattening the recurrent update assumes. `seq_len` is the length
        of the window the update unrolls the LSTM over: one recurrent state is kept per
        window, the one recorded at its first step.
        """
        steps_num = len(self)
        rewards = np.asarray(self.steps["rewards"], dtype=np.float32)
        values = np.asarray(self.steps["values"], dtype=np.float32)
        dones = np.asarray(self.steps["dones"], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(steps_num)):
            if t == steps_num - 1:
                next_non_terminal = 1.0 - final_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            running = delta + gamma * lam * next_non_terminal * running
            advantages[t] = running

        returns = advantages + values

        flat = {
            name: swap_and_flatten(np.asarray(self.steps[name]))
            for name in ("visual", "vels", "actions", "states")
        }
        return Rollout(
            visual=torch.as_tensor(flat["visual"], device=device),
            vels=torch.as_tensor(flat["vels"], device=device),
            dones=torch.as_tensor(swap_and_flatten(dones), device=device),
            actions=torch.as_tensor(flat["actions"].astype(np.int64), device=device),
            values=torch.as_tensor(swap_and_flatten(values), device=device),
            neglogpacs=torch.as_tensor(
                swap_and_flatten(np.asarray(self.steps["neglogpacs"], dtype=np.float32)),
                device=device,
            ),
            returns=torch.as_tensor(swap_and_flatten(returns), device=device),
            states=torch.as_tensor(flat["states"][::seq_len], device=device),
        )


class PPO:
    """
    The network and its update. It is handed the network rather than building one, and it
    holds no environment and no loop: a driver asks it to act on observations and to learn
    from a finished Rollout, and carries the recurrent state between the two itself.
    """

    def __init__(self, agent, config, device):
        self.agent = agent
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)

    def initial_state(self, env_num):
        return self.agent.initial_state(env_num, self.device)

    @torch.no_grad()
    def act(self, visual, vels, state, dones, env_num):
        logits, value, state = self.agent(visual, vels, state, dones, env_num)
        probabilities = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probabilities, 1).squeeze(-1)
        neglogpacs = F.cross_entropy(logits, actions, reduction="none")
        return actions, value.squeeze(-1), neglogpacs, state

    def update(self, rollout):
        config = self.config
        batch_size = rollout.returns.shape[0]
        num_minibatches = batch_size // config.minibatch_size
        assert num_minibatches > 0, (
            f"minibatch_size {config.minibatch_size} exceeds the batch of {batch_size} steps"
        )
        assert config.minibatch_size % config.seq_len == 0

        advantages = rollout.returns - rollout.values
        if config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_sequences = batch_size // config.seq_len
        sequences_per_batch = config.minibatch_size // config.seq_len
        step_indexes = np.arange(total_sequences * config.seq_len).reshape(
            total_sequences, config.seq_len
        )

        losses = {"actor": [], "critic": [], "entropy": [], "kl": []}
        for _ in range(config.mini_epochs):
            order = np.random.permutation(total_sequences)
            for minibatch in range(num_minibatches):
                chosen = order[
                    minibatch * sequences_per_batch : (minibatch + 1) * sequences_per_batch
                ]
                flat = torch.as_tensor(step_indexes[chosen].ravel(), device=self.device)
                reported = self.minibatch_step(
                    rollout,
                    advantages,
                    flat,
                    torch.as_tensor(chosen, device=self.device),
                    len(chosen),
                )
                for name, value in reported.items():
                    losses[name].append(value)

        return {name: float(np.mean(values)) for name, values in losses.items()}

    def minibatch_step(self, rollout, advantages, flat, sequences, env_num):
        config = self.config
        logits, values, _ = self.agent(
            rollout.visual[flat],
            rollout.vels[flat],
            rollout.states[sequences],
            rollout.dones[flat],
            env_num,
        )
        values = values.squeeze(-1)

        neglogpacs = F.cross_entropy(logits, rollout.actions[flat], reduction="none")
        old_neglogpacs = rollout.neglogpacs[flat]
        batch_advantages = advantages[flat]

        ratio = torch.exp(old_neglogpacs - neglogpacs)
        unclipped = -batch_advantages * ratio
        clipped = -batch_advantages * torch.clamp(ratio, 1.0 - config.e_clip, 1.0 + config.e_clip)
        actor_loss = torch.max(unclipped, clipped).mean()

        returns = rollout.returns[flat]
        old_values = rollout.values[flat]
        value_loss = (values - returns) ** 2
        if config.clip_value:
            clipped_values = old_values + torch.clamp(
                values - old_values, -config.e_clip, config.e_clip
            )
            value_loss = torch.max(value_loss, (clipped_values - returns) ** 2)
        critic_loss = value_loss.mean()

        log_probabilities = F.log_softmax(logits, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()

        loss = actor_loss + 0.5 * config.critic_coef * critic_loss - config.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), config.grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            kl = 0.5 * ((old_neglogpacs - neglogpacs) ** 2).mean()

        return {
            "actor": actor_loss.item(),
            "critic": critic_loss.item(),
            "entropy": entropy.item(),
            "kl": kl.item(),
        }

    def save(self, path, progress):
        """
        `progress` is whatever the driver counts (its epoch and frame); it comes back out
        of restore untouched.
        """
        torch.save(
            {
                "model": self.agent.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                **progress,
            },
            path,
        )

    def restore(self, path):
        state = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        return state
