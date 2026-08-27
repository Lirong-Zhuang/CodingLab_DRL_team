import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from itertools import compress
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


# ---------------------------------------------------------------------------
# Greedy oracle — used for behavioural cloning warm-start
# Mirrors greedy.py exactly, including variant 2 blocked cell layout
# ---------------------------------------------------------------------------
class GreedyOracle:
    def __init__(self, variant):
        self.variant           = variant
        self.max_response_time = 15 if variant == 2 else 10
        self.reward_val        = 25 if variant == 2 else 15

        if variant == 0 or variant == 1:
            # Full 5x5 grid — 25 nodes
            neighbor_matrix = np.zeros((25, 25), int)
            for i in range(25):
                for j in range(i + 1, 25):
                    iv, ih = divmod(i, 5)
                    jv, jh = divmod(j, 5)
                    if (iv == jv and abs(ih - jh) == 1) or \
                       (ih == jh and abs(iv - jv) == 1):
                        neighbor_matrix[i, j] = 1
            self.mapping = None  # not needed for variants 0/1

        else:
            # Variant 2: blocked cells, only 18 eligible nodes
            # cell indices (from greedy.py):
            # |  0 |  X |  7 | 12 | 13 |
            # |  1 |  X |  8 |  X | 14 |
            # |  2 |  X |  9 |  X | 15 |
            # |  3 |  5 | 10 |  X | 16 |
            # |  4 |  6 | 11 |  X | 17 |
            self.mapping = [
                (0,0), (1,0), (2,0), (3,0), (4,0),
                (3,1), (4,1),
                (0,2), (1,2), (2,2), (3,2), (4,2),
                (0,3),
                (0,4), (1,4), (2,4), (3,4), (4,4)
            ]
            neighbor_matrix = np.zeros((18, 18), int)
            neighbor_matrix[0, 1] = 1
            neighbor_matrix[1, 2] = 1
            neighbor_matrix[2, 3] = 1
            neighbor_matrix[3, 4] = 1
            neighbor_matrix[3, 5] = 1
            neighbor_matrix[4, 6] = 1
            neighbor_matrix[5, 6] = 1
            neighbor_matrix[5, 10] = 1
            neighbor_matrix[6, 11] = 1
            neighbor_matrix[7, 8] = 1
            neighbor_matrix[7, 12] = 1
            neighbor_matrix[8, 9] = 1
            neighbor_matrix[9, 10] = 1
            neighbor_matrix[10, 11] = 1
            neighbor_matrix[12, 13] = 1
            neighbor_matrix[13, 14] = 1
            neighbor_matrix[14, 15] = 1
            neighbor_matrix[15, 16] = 1
            neighbor_matrix[16, 17] = 1

        graph = csr_matrix(neighbor_matrix)
        self.dist_matrix, predecessors = dijkstra(
            csgraph=graph, directed=False,
            return_predecessors=True, unweighted=True
        )
        n = predecessors.shape[0]
        self.successors = np.zeros_like(predecessors)
        for i in range(n):
            for j in range(n):
                if i == j:
                    self.successors[i, j] = i
                else:
                    pred = j
                    while True:
                        new_pred = predecessors[i, pred]
                        if new_pred == i:
                            self.successors[i, j] = pred
                            break
                        pred = new_pred

    def _idx(self, coord):
        if self.mapping is None:
            return coord[0] * 5 + coord[1]
        return self.mapping.index(coord)

    def _coord(self, idx):
        if self.mapping is None:
            return (idx // 5, idx % 5)
        return self.mapping[idx]

    def _dist(self, c1, c2):
        return self.dist_matrix[self._idx(c1), self._idx(c2)]

    def _action(self, cur, tar):
        if cur == tar:
            return 0
        nxt = self._coord(self.successors[self._idx(cur), self._idx(tar)])
        dv  = cur[0] - nxt[0]
        if dv != 0:
            return 1 if dv > 0 else 3
        return 4 if cur[1] - nxt[1] > 0 else 2

    def act(self, env):
        if env.agent_load == env.agent_capacity:
            return self._action(env.agent_loc, env.target_loc)
        if not env.item_locs:
            return 0
        dist = np.array([self._dist(env.agent_loc, loc) for loc in env.item_locs])
        mask = dist <= self.max_response_time - np.array(env.item_times)
        item_locs = list(compress(env.item_locs, mask))
        if not item_locs:
            return 0
        dist    = dist[mask]
        dist   += np.array([self._dist(loc, env.target_loc) for loc in item_locs])
        profit  = self.reward_val - dist
        mask2   = profit > 0
        item_locs = list(compress(item_locs, mask2))
        if not item_locs:
            return 0
        profit = profit[mask2]
        return self._action(env.agent_loc, item_locs[int(np.argmax(profit))])


# ---------------------------------------------------------------------------
# Actor-Critic: CNN over item grids + MLP over scalars
#
# Obs layout (62 features total):
#   [0]      step_count (normalised)
#   [1,2]    agent location (normalised)
#   [3]      agent load (normalised)
#   [4:29]   item presence grid 5x5
#   [29:54]  item urgency grid  5x5
#   [54:62]  8 engineered scalar features
# ---------------------------------------------------------------------------
class HybridActorCritic(nn.Module):
    """CNN grid encoder plus a separate MLP scalar encoder."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.cnn_presence = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()    # 16*5*5 = 400
        )
        self.cnn_urgency = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten()    # 400
        )
        scalar_dim = 4 + (obs_dim - 54)
        self.scalar_mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        self.shared = nn.Sequential(
            nn.Linear(400 + 400 + 64, 256), nn.ReLU(),
            nn.Linear(256, 128),                    nn.ReLU(),
            nn.Linear(128, 64),                     nn.ReLU(),
        )
        self.actor_head  = nn.Linear(64, act_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, obs):
        scalars  = torch.cat([obs[:, :4], obs[:, 54:]], dim=-1)
        presence = obs[:, 4:29].view(-1, 1, 5, 5)
        urgency  = obs[:, 29:54].view(-1, 1, 5, 5)
        feat = self.shared(torch.cat([
            self.cnn_presence(presence),
            self.cnn_urgency(urgency),
            self.scalar_mlp(scalars)
        ], dim=-1))
        return self.actor_head(feat), self.critic_head(feat)


class MLPActorCritic(nn.Module):
    """Pure MLP baseline operating on the complete observation vector."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor_head = nn.Linear(64, act_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, obs):
        feat = self.shared(obs)
        return self.actor_head(feat), self.critic_head(feat)


class CNNActorCritic(nn.Module):
    """Pure CNN encoder; scalar state is represented as constant spatial maps."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.scalar_dim = 4 + (obs_dim - 54)
        channels = 3 + self.scalar_dim  # presence, urgency, agent, scalar maps
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.actor_head = nn.Linear(64, act_dim)
        self.critic_head = nn.Linear(64, 1)

    def forward(self, obs):
        presence = obs[:, 4:29].view(-1, 1, 5, 5)
        urgency = obs[:, 29:54].view(-1, 1, 5, 5)
        agent = torch.zeros_like(presence)
        rows = torch.round(obs[:, 1] * 4).long().clamp(0, 4)
        cols = torch.round(obs[:, 2] * 4).long().clamp(0, 4)
        agent[torch.arange(obs.shape[0], device=obs.device), 0, rows, cols] = 1.0
        scalars = torch.cat([obs[:, :4], obs[:, 54:]], dim=-1)
        scalar_maps = scalars[:, :, None, None].expand(-1, -1, 5, 5)
        feat = self.encoder(torch.cat([presence, urgency, agent, scalar_maps], dim=1))
        return self.actor_head(feat), self.critic_head(feat)


# Backwards-compatible name used by older checkpoints and scripts.
ActorCritic = HybridActorCritic


# ---------------------------------------------------------------------------
# PPO_v1  (name kept for test_ppo_leaderboard.py compatibility)
#
# Step overview:
#   1. GreedyOracle creates demonstration actions for behavioural cloning only.
#   2. run_bc_phase() trains the neural actor-critic to imitate those actions.
#   3. select_action() always chooses from the neural network at evaluation time.
#   4. update() contains the PPO clipped objective for optional fine-tuning.
# ---------------------------------------------------------------------------
class PPO_v1:
    def __init__(self, env, architecture="hybrid"):
        self.env          = env
        self.variant      = env.variant
        self.data_dir     = env.data_dir
        self.file_name    = "PPO_v1."
        self.architecture = architecture
        self.network_name = f"PPO {architecture.upper()} Network"

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        initial_obs  = self.env.reset("training")
        self.obs_dim = len(initial_obs)
        self.act_dim = 5

        self.episode_steps      = self.env.episode_steps
        self.num_episodes       = 10000
        self.bc_episodes        = 2000
        self.bc_epochs          = 20
        self.bc_batch_size      = 2048
        self.bc_lr              = 1e-3      # supervised BC warm-start
        self.gamma              = 0.99
        self.learning_rate      = 2e-4
        self.clip_epsilon       = 0.1
        self.update_epochs      = 6
        self.gae_lambda         = 0.98
        self.entropy_coef_start = 0.03    # BC already taught useful actions
        self.entropy_coef_end   = 0.002
        self.entropy_coef       = self.entropy_coef_start
        self.value_coef         = 0.5
        self.max_grad_norm      = 0.5

        networks = {
            "hybrid": HybridActorCritic,
            "cnn": CNNActorCritic,
            "mlp": MLPActorCritic,
        }
        if architecture not in networks:
            raise ValueError(f"Unknown architecture: {architecture}")
        self.network = networks[architecture](self.obs_dim, self.act_dim).to(self.device)
        self.bc_optimizer = optim.Adam(self.network.parameters(), lr=self.bc_lr)
        self.optimizer    = optim.Adam(self.network.parameters(), lr=self.learning_rate)
        self.oracle       = GreedyOracle(self.variant)

    def action_mask_from_obs(self, obs_t):
        # Step 0: mask impossible moves so the policy never selects blocked/off-grid actions.
        row = torch.round(obs_t[:, 1] * (self.env.vertical_cell_count - 1)).long()
        col = torch.round(obs_t[:, 2] * (self.env.horizontal_cell_count - 1)).long()

        mask = torch.zeros((obs_t.shape[0], self.act_dim), dtype=torch.bool, device=obs_t.device)
        mask[:, 0] = True

        eligible = set(self.env.eligible_cells)
        for i, (r, c) in enumerate(zip(row.tolist(), col.tolist())):
            candidates = {
                1: (r - 1, c),
                2: (r, c + 1),
                3: (r + 1, c),
                4: (r, c - 1),
            }
            for action, loc in candidates.items():
                if loc in eligible:
                    mask[i, action] = True
        return mask


    # Behavioural cloning warm-start
    def run_bc_phase(self):
        # Step 1: collect training states and greedy actions as supervised examples.
        print(f"Behavioural cloning warm-start ({self.bc_episodes} episodes)...")
        all_obs, all_acts = [], []
        for ep in range(self.bc_episodes):
            obs = self.env.reset("training")
            for _ in range(self.episode_steps):
                act = self.oracle.act(self.env)
                all_obs.append(obs)
                all_acts.append(act)
                _, obs, done = self.env.step(act)
                if done:
                    break

            if (ep + 1) % 100 == 0:
                print(f"  collected {ep+1}/{self.bc_episodes} episodes")

        obs_t = torch.FloatTensor(np.array(all_obs))
        act_t = torch.LongTensor(all_acts)
        dataset = TensorDataset(obs_t, act_t)
        loader = DataLoader(dataset, batch_size=self.bc_batch_size, shuffle=True)

        # Step 2: fit the actor logits to the greedy labels with cross-entropy.
        for epoch in range(self.bc_epochs):
            total_loss = 0.0
            total_correct = 0
            total_count = 0
            for batch_obs, batch_act in loader:
                batch_obs = batch_obs.to(self.device)
                batch_act = batch_act.to(self.device)
                logits, _ = self.network(batch_obs)
                mask = self.action_mask_from_obs(batch_obs)
                logits = logits.masked_fill(~mask, -1e9)
                loss = nn.CrossEntropyLoss()(logits, batch_act)

                self.bc_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.bc_optimizer.step()

                with torch.no_grad():
                    preds = torch.argmax(logits, dim=-1)
                    total_correct += int((preds == batch_act).sum().item())
                    total_count += int(batch_act.numel())
                    total_loss += float(loss.item()) * int(batch_act.numel())

            print(
                f"  BC epoch {epoch+1}/{self.bc_epochs}  "
                f"loss: {total_loss / total_count:.4f}  "
                f"acc: {total_correct / total_count:.2%}"
            )

        print("Behavioural cloning done. Starting PPO...\n")


    # Action selection
    def select_action(self, obs, deterministic=False):
        # Step 3: use the neural policy for both PPO rollouts and test evaluation.
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.network(obs_t)
            mask = self.action_mask_from_obs(obs_t)
            logits = logits.masked_fill(~mask, -1e9)
        dist   = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()


    # GAE returns
    def compute_returns(self, rewards, dones, values):
        # Step 4: estimate advantages with GAE for PPO fine-tuning.
        advantages, gae, next_val = [], 0.0, 0.0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values)):
            if d:
                next_val = gae = 0.0
            delta = r + self.gamma * next_val - v
            gae   = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
            next_val = v
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = adv_t + torch.FloatTensor(values).to(self.device)
        if len(adv_t) > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        return ret_t, adv_t


    # PPO update
    def update(self, obs, actions, log_probs, returns, advantages):
        # Step 5: apply PPO's clipped policy loss plus value and entropy terms.
        obs_t    = torch.FloatTensor(np.array(obs)).to(self.device)
        act_t    = torch.LongTensor(actions).to(self.device)
        old_lp_t = torch.FloatTensor(log_probs).to(self.device)

        tl = ta = tc = te = 0.0
        for _ in range(self.update_epochs):
            logits, vals = self.network(obs_t)
            mask = self.action_mask_from_obs(obs_t)
            logits = logits.masked_fill(~mask, -1e9)
            dist    = torch.distributions.Categorical(logits=logits)
            new_lp  = dist.log_prob(act_t)
            entropy = dist.entropy().mean()
            ratio   = torch.exp(new_lp - old_lp_t)

            actor_loss  = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - self.clip_epsilon,
                             1 + self.clip_epsilon) * advantages
            ).mean()
            critic_loss = nn.MSELoss()(vals.squeeze(), returns)
            loss        = actor_loss + self.value_coef * critic_loss \
                          - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()

            tl += loss.item(); ta += actor_loss.item()
            tc += critic_loss.item(); te += entropy.item()

        n = self.update_epochs
        return tl/n, ta/n, tc/n, te/n


    # Entropy decay
    def decay_entropy(self, episode):
        frac = min(episode / (self.num_episodes * 0.8), 1.0)
        self.entropy_coef = (
            self.entropy_coef_start
            + frac * (self.entropy_coef_end - self.entropy_coef_start)
        )

    # Save / load
    def save(self, path):
        # Step 6: save only the neural network weights and PPO metadata.
        torch.save({
            "model_state_dict": self.network.state_dict(),
            "obs_dim": self.obs_dim, "act_dim": self.act_dim,
            "variant": self.variant, "agent": self.network_name,
            "architecture": self.architecture,
            "gamma": self.gamma, "learning_rate": self.learning_rate,
            "clip_epsilon": self.clip_epsilon, "update_epochs": self.update_epochs,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.network.load_state_dict(
            ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        )
        self.network.eval()

