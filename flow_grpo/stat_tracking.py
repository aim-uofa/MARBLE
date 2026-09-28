import numpy as np
from collections import deque


class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.full_like(rewards, np.nan, dtype=np.float64)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            # Only accumulate non-NaN rewards into stats
            valid = ~np.isnan(prompt_rewards)
            if prompt not in self.stats:
                self.stats[prompt] = []
            if valid.any():
                self.stats[prompt].extend(prompt_rewards[valid].tolist())
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if not self.stats.get(prompt):
                # No valid rewards ever seen for this prompt — leave NaN
                continue
            self.stats[prompt] = np.array(self.stats[prompt])
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            if self.global_std:
                # Use global std from non-NaN rewards only
                valid_rewards = rewards[~np.isnan(rewards)]
                std = (np.std(valid_rewards, axis=0, keepdims=True) + 1e-4) if len(valid_rewards) > 0 else 1e-4
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
            # NaN prompt_rewards naturally produce NaN advantages (correct)
            advantages[prompts == prompt] = (prompt_rewards - mean) / std
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def main():
    tracker = PerPromptStatTracker()
    prompts = ["a", "b", "a", "c", "b", "a"]
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(prompts, rewards)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()
