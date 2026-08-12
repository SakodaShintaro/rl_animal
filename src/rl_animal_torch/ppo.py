"""PPO with a recurrent policy, as a2c_discrete.A2CAgent implemented it.

The details that make this the same algorithm rather than a generic PPO:

- The rollout is collected environment-major. Data arrives as [steps, actors, ...] and is
  transposed to [actors, steps, ...] before flattening, so every contiguous run of
  seq_len entries belongs to one environment. The recurrent state carried into each such
  run is the one recorded at its first step.
- The update shuffles whole sequences, never individual steps, and rebuilds the network
  over minibatch_size / seq_len sequences of seq_len steps.
- Both the policy and the value loss are clipped, and the maximum of clipped and
  unclipped is taken. The value loss is halved again by critic_coef * 0.5.
- Advantages are normalized across the whole batch, before it is split.
- Gradients are clipped by global norm.
"""
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from rl_animal_torch.network import LSTM_UNITS


def format_duration(seconds):
    '''
    A full run is tens of hours, so days are worth separating out.
    '''
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 24:
        days, hours = divmod(hours, 24)
        return '%dd %d:%02d:%02d' % (days, hours, minutes, seconds)

    return '%d:%02d:%02d' % (hours, minutes, seconds)


def swap_and_flatten(array):
    '''
    [steps, actors, ...] -> [actors * steps, ...], which puts each environment's steps
    next to each other.
    '''
    shape = array.shape
    return array.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])


class Rollout:
    '''
    One batch of experience, already flattened and on the training device.
    '''
    def __init__(self, visual, vels, dones, actions, values, neglogpacs, returns, states):
        self.visual = visual
        self.vels = vels
        self.dones = dones
        self.actions = actions
        self.values = values
        self.neglogpacs = neglogpacs
        self.returns = returns
        self.states = states


class PPOTrainer:
    def __init__(self, agent, vec_env, config, device, writer):
        self.agent = agent.to(device)
        self.vec_env = vec_env
        self.config = config
        self.device = device
        self.writer = writer
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=config.learning_rate)

        self.batch_size = config.steps_num * config.num_actors
        self.num_minibatches = self.batch_size // config.minibatch_size
        assert self.num_minibatches > 0, (
            'minibatch_size %d exceeds the batch of %d steps'
            % (config.minibatch_size, self.batch_size))
        assert config.minibatch_size % config.seq_len == 0

        self.state = self.agent.initial_state(config.num_actors, device=device)
        self.dones = torch.zeros(config.num_actors, device=device)
        self.current_rewards = np.zeros(config.num_actors, dtype=np.float32)
        self.episode_rewards = deque([], maxlen=1000)
        self.frame = 0
        self.epoch = 0

    def to_device(self, visual, vels):
        return (torch.as_tensor(visual, device=self.device),
                torch.as_tensor(vels, device=self.device))

    @torch.no_grad()
    def act(self, visual, vels):
        logits, value, self.state = self.agent(visual, vels, self.state, self.dones,
                                               self.config.num_actors)
        '''
        models.LSTMModelA2C sampled with a Gumbel-max over a uniform tensor, which is a
        draw from the categorical distribution; torch.multinomial is the same distribution
        with this framework's generator.
        '''
        probabilities = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probabilities, 1).squeeze(-1)
        neglogpacs = F.cross_entropy(logits, actions, reduction='none')
        return actions, value.squeeze(-1), neglogpacs

    @torch.no_grad()
    def collect(self):
        config = self.config
        steps = {name: [] for name in
                 ('visual', 'vels', 'rewards', 'actions', 'values', 'dones', 'neglogpacs',
                  'states')}

        visual, vels = self.to_device(*self.vec_env.observations())
        for _ in range(config.steps_num):
            steps['states'].append(self.state.cpu().numpy())
            actions, values, neglogpacs = self.act(visual, vels)

            steps['visual'].append(visual.cpu().numpy())
            steps['vels'].append(vels.cpu().numpy())
            steps['actions'].append(actions.cpu().numpy())
            steps['values'].append(values.cpu().numpy())
            steps['neglogpacs'].append(neglogpacs.cpu().numpy())
            steps['dones'].append(self.dones.cpu().numpy())

            (next_visual, next_vels), rewards, dones = self.vec_env.step(
                actions.cpu().numpy())
            steps['rewards'].append(rewards)

            self.current_rewards += rewards
            for reward, done in zip(self.current_rewards, dones):
                if done:
                    self.episode_rewards.append(float(reward))
            self.current_rewards = self.current_rewards * (1.0 - dones)

            self.dones = torch.as_tensor(dones.astype(np.float32), device=self.device)
            visual, vels = self.to_device(next_visual, next_vels)

        _, last_values, _ = self.act(visual, vels)
        return self.finish(steps, last_values.cpu().numpy())

    def finish(self, steps, last_values):
        '''
        Generalised advantage estimation over the collected steps, then the same
        environment-major flattening the original applied.
        '''
        config = self.config
        rewards = np.asarray(steps['rewards'], dtype=np.float32)
        values = np.asarray(steps['values'], dtype=np.float32)
        dones = np.asarray(steps['dones'], dtype=np.float32)
        final_dones = self.dones.cpu().numpy()

        advantages = np.zeros_like(rewards)
        running = 0.0
        for t in reversed(range(config.steps_num)):
            if t == config.steps_num - 1:
                next_non_terminal = 1.0 - final_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]
            delta = rewards[t] + config.gamma * next_values * next_non_terminal - values[t]
            running = delta + config.gamma * config.lam * next_non_terminal * running
            advantages[t] = running

        returns = advantages + values

        flat = {name: swap_and_flatten(np.asarray(steps[name]))
                for name in ('visual', 'vels', 'actions', 'states')}
        '''
        states keeps one recurrent state per sequence, the one recorded at the sequence's
        first step.
        '''
        return Rollout(
            visual=torch.as_tensor(flat['visual'], device=self.device),
            vels=torch.as_tensor(flat['vels'], device=self.device),
            dones=torch.as_tensor(swap_and_flatten(dones), device=self.device),
            actions=torch.as_tensor(flat['actions'].astype(np.int64), device=self.device),
            values=torch.as_tensor(swap_and_flatten(values), device=self.device),
            neglogpacs=torch.as_tensor(
                swap_and_flatten(np.asarray(steps['neglogpacs'], dtype=np.float32)),
                device=self.device),
            returns=torch.as_tensor(swap_and_flatten(returns), device=self.device),
            states=torch.as_tensor(flat['states'][::config.seq_len], device=self.device))

    def update(self, rollout):
        config = self.config
        advantages = rollout.returns - rollout.values
        if config.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_sequences = self.batch_size // config.seq_len
        sequences_per_batch = config.minibatch_size // config.seq_len
        step_indexes = np.arange(total_sequences * config.seq_len).reshape(
            total_sequences, config.seq_len)

        losses = {'actor': [], 'critic': [], 'entropy': [], 'kl': []}
        for _ in range(config.mini_epochs):
            order = np.random.permutation(total_sequences)
            for minibatch in range(self.num_minibatches):
                chosen = order[minibatch * sequences_per_batch:
                               (minibatch + 1) * sequences_per_batch]
                flat = torch.as_tensor(step_indexes[chosen].ravel(), device=self.device)
                reported = self.step(rollout, advantages, flat,
                                     torch.as_tensor(chosen, device=self.device),
                                     len(chosen))
                for name, value in reported.items():
                    losses[name].append(value)

        return {name: float(np.mean(values)) for name, values in losses.items()}

    def step(self, rollout, advantages, flat, sequences, env_num):
        config = self.config
        logits, values, _ = self.agent(rollout.visual[flat], rollout.vels[flat],
                                       rollout.states[sequences], rollout.dones[flat],
                                       env_num)
        values = values.squeeze(-1)

        neglogpacs = F.cross_entropy(logits, rollout.actions[flat], reduction='none')
        old_neglogpacs = rollout.neglogpacs[flat]
        batch_advantages = advantages[flat]

        ratio = torch.exp(old_neglogpacs - neglogpacs)
        unclipped = -batch_advantages * ratio
        clipped = -batch_advantages * torch.clamp(ratio, 1.0 - config.e_clip,
                                                  1.0 + config.e_clip)
        actor_loss = torch.max(unclipped, clipped).mean()

        returns = rollout.returns[flat]
        old_values = rollout.values[flat]
        value_loss = (values - returns) ** 2
        if config.clip_value:
            clipped_values = old_values + torch.clamp(values - old_values,
                                                      -config.e_clip, config.e_clip)
            value_loss = torch.max(value_loss, (clipped_values - returns) ** 2)
        critic_loss = value_loss.mean()

        '''
        The entropy of the categorical distribution, which is what
        softmax_cross_entropy_with_logits(logits, softmax(logits)) computes.
        '''
        log_probabilities = F.log_softmax(logits, dim=-1)
        entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1).mean()

        loss = (actor_loss + 0.5 * config.critic_coef * critic_loss
                - config.entropy_coef * entropy)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), config.grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            kl = 0.5 * ((old_neglogpacs - neglogpacs) ** 2).mean()

        return {'actor': actor_loss.item(), 'critic': critic_loss.item(),
                'entropy': entropy.item(), 'kl': kl.item()}

    def train(self, checkpoint_path, best_checkpoint_path):
        config = self.config
        best_reward = -float('inf')
        self.vec_env.reset()
        '''
        The rate is measured over this session only, so a run resumed with --restore
        predicts from its own speed rather than from the whole history.
        '''
        started = time.time()
        epochs_this_session = 0
        while self.epoch < config.max_epochs:
            self.epoch += 1
            epochs_this_session += 1
            self.frame += self.batch_size

            collect_start = time.time()
            rollout = self.collect()
            collect_time = time.time() - collect_start

            update_start = time.time()
            losses = self.update(rollout)
            update_time = time.time() - update_start

            steps_per_second = self.batch_size / (collect_time + update_time)
            self.writer.add_scalar('performance/fps', steps_per_second, self.frame)
            self.writer.add_scalar('performance/play_time', collect_time, self.frame)
            self.writer.add_scalar('performance/update_time', update_time, self.frame)
            for name, value in losses.items():
                self.writer.add_scalar('losses/' + name, value, self.frame)
            self.writer.add_scalar('epochs', self.epoch, self.frame)

            elapsed = time.time() - started
            remaining = (elapsed / epochs_this_session) * (config.max_epochs - self.epoch)
            self.writer.add_scalar('performance/elapsed_hours', elapsed / 3600.0,
                                   self.frame)
            self.writer.add_scalar('performance/eta_hours', remaining / 3600.0, self.frame)

            report = ('epoch %d/%d  frame %d  %.0f steps/s  elapsed %s  eta %s  '
                      'actor %.4f  critic %.4f  entropy %.4f'
                      % (self.epoch, config.max_epochs, self.frame, steps_per_second,
                         format_duration(elapsed), format_duration(remaining),
                         losses['actor'], losses['critic'], losses['entropy']))
            if len(self.episode_rewards) > 0:
                mean_reward = float(np.mean(self.episode_rewards))
                self.writer.add_scalar('mean_rewards', mean_reward, self.frame)
                report += '  mean reward %.4f' % mean_reward
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save(best_checkpoint_path)
                    report += ' (best, saved)'
            print(report, flush=True)

            self.save(checkpoint_path)

    def save(self, path):
        torch.save({'model': self.agent.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    'frame': self.frame}, path)

    def restore(self, path):
        state = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.epoch = state['epoch']
        self.frame = state['frame']
        return self
