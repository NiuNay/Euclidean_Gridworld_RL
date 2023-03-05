import numpy as np
from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongQLearningRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = False

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=self._epsilon.value)
        reward, new_state = self._train_environment.step(action)

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _runner_specific_visualisations(self):
        pass


class EpisodicQLearningRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = False
        self._deltas_ = {
            (-1, 0): 0,
            (0, 1): 1,
            (1, 0): 2,
            (0, -1): 3,
            (-1, 1): 4,
            (1, 1): 5,
            (1, -1): 6,
            (-1, -1): 7,
        }

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=self._epsilon.value)
        reward, new_state = self._train_environment.step(action)

        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )

        return new_state, reward

    def _model_train_step_from_file(self, states) -> float:
        """perform single training step."""
        state = tuple(np.int_(states[0]))
        new_state = tuple(np.int_(states[1]))
        diff = tuple(map(lambda i, j: i - j, new_state, state))
        action = self._deltas_[diff]
        reward, new_state = self._train_environment.step(action)
        self._model.step(
            state=state,
            action=action,
            reward=reward,
            new_state=new_state,
            active=self._train_environment.active,
        )
        return reward

    def _runner_specific_visualisations(self):
        pass
