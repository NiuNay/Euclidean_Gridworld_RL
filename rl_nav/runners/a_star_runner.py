import numpy as np
from rl_nav.runners import episodic_runner, lifelong_runner


class LifelongAStarRunner(lifelong_runner.LifelongRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._deltas = self._train_environment.action_deltas
        self._model.deltas = self._deltas

        self._planner = True

    def _model_train_step(self, state) -> float:
        """Perform single training step."""
        action = self._model.select_behaviour_action(state, epsilon=1)
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


class EpisodicAStarRunner(episodic_runner.EpisodicRunner):
    def __init__(self, config, unique_id: str):

        super().__init__(config=config, unique_id=unique_id)

        self._planner = True
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
        if states.shape[1]==4:
            state = tuple(np.int_(states[0,0:2]))
            new_state_actual = tuple(np.int_(states[1,0:2]))
            attempted_state = tuple(np.int_(states[1,2:4]))
            diff = tuple(map(lambda i, j: i - j, attempted_state, state))
        elif states.shape[1]==2:
            state = tuple(np.int_(states[0]))
            new_state_actual = tuple(np.int_(states[1]))
            diff = tuple(map(lambda i, j: i - j, new_state_actual, state))

        action = self._deltas_[diff]
        reward, new_state = self._train_environment.step(action)
        if new_state_actual != new_state:
            raise ValueError(
                f"The next state {new_state_actual} (state number {self._trial_step_count+1} of trial {self._trial_num}) specified by your training data does not match a permissible state in the training map you have specified."
                )
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
