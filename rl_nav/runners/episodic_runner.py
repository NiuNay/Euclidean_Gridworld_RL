import os
import numpy as np
from typing import Any, Dict, Optional, Union

from rl_nav import constants
from rl_nav.runners import base_runner


class EpisodicRunner(base_runner.BaseRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._episode_count = 0

    def _get_runner_specific_data_columns(self):
        columns = [
            constants.TRAIN_EPISODE_REWARD,
            constants.TRAIN_EPISODE_LENGTH,
        ]
        return columns

    def _train_rollout(self):
        self._train_environment.visualise_episode_history(
            save_path=os.path.join(
                self._rollout_folder_path,
                f"{constants.INDIVIDUAL_TRAIN_RUN}_{self._step_count}.mp4",
            )
        )
        while self._next_rollout_step <= self._step_count:
            self._next_rollout_step += self._rollout_frequency

    def train(self):
        self._model.train()
        self._model.env_transition_matrix = self._train_environment.transition_matrix
        if self._file_path is None:
            while self._step_count < self._num_steps:
                self._train_episode()
        else:
            training_data = np.load(self._file_path)
            num_trials = training_data.shape[2]
            for i in range(num_trials):
                a = training_data[:,:,i]
                a = a[np.insert(np.invert(np.all(np.diff(a,axis=0)==0,axis=1)), 0, True)]
                self._num_steps = len(a)
                self._episode_timeout=self._num_steps
                self._trial_num = i+1;
                self._trial_step_count = 0
                while self._trial_step_count+1 < self._num_steps:
                    self._train_episode_from_file(a)

    def _train_episode(self) -> Dict[str, Any]:
        """Perform single training loop.

        Args:
            episode: index of episode

        Returns:
            logging_dict: dictionary of items to log (e.g. episode reward).
        """
        self._episode_count += 1
        episode_reward = 0

        state = self._train_environment.reset_environment()

        while self._train_environment.active and self._step_count < self._num_steps:

            state, reward, logging_dict = self._train_step(state=state)
            episode_reward += reward

            logging_dict[constants.STEP] = self._step_count

            if not self._train_environment.active:
                logging_dict[constants.TRAIN_EPISODE_REWARD] = episode_reward
                logging_dict[
                    constants.TRAIN_EPISODE_LENGTH
                ] = self._train_environment.episode_step_count

            self._log_episode(step=self._step_count, logging_dict=logging_dict)
            if (
                self._step_count % self._checkpoint_frequency == 0
                and self._step_count != 1
            ):
                self._data_logger.checkpoint()

    def _train_episode_from_file(self,data) -> Dict[str, Any]:
        """Perform single training loop using data from 1 trial of the file provided.

        Args:
            episode: index of episode

        Returns:
            logging_dict: dictionary of items to log (e.g. episode reward).
        """
        self._episode_count += 1
        episode_reward = 0

        state = tuple(np.int_(data[self._trial_step_count,0:2]))
        self._train_environment.reset_environment(start_position=state)

        while self._train_environment.active and self._trial_step_count+1 < self._num_steps:

            states = data[self._trial_step_count:self._trial_step_count+2,:]
            reward, logging_dict = self._train_step_from_file(states=states)
            episode_reward += reward

            logging_dict[constants.STEP] = self._step_count

            if not self._train_environment.active or self._step_count == 1:
                logging_dict[constants.TRAIN_EPISODE_REWARD] = episode_reward
                logging_dict[
                    constants.TRAIN_EPISODE_LENGTH
                ] = self._train_environment.episode_step_count

            self._log_episode(step=self._step_count, logging_dict=logging_dict)

            if (
                self._step_count % self._checkpoint_frequency == 0
                and self._step_count != 1
            ):
                self._data_logger.checkpoint()