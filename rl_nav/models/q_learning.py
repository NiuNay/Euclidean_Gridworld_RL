import copy
from typing import Dict, List, Tuple, Type

import numpy as np
from rl_nav import constants
from rl_nav.models import tabular_learner
from rl_nav.utils import learning_rate_schedules


class QLearner(tabular_learner.TabularLearner):
    """Q-learning (Watkins)."""

    def __init__(
        self,
        action_space: List[int],
        state_space: List[Tuple[int, int]],
        learning_rate: Type[learning_rate_schedules.LearningRateSchedule],
        gamma: float,
        initialisation_strategy: Dict,
        behaviour: str,
        target: str,
        imputation_method: str,
        update_no_op: bool,
    ):
        """Class constructor.

        Args:
            action_space: list of actions available.
            state_space: list of states.
            learning_rate: learning_rage.
            gamma: discount factor.
            initialisation_strategy: name of network initialisation strategy.
            behaviour: name of behaviour type e.g. epsilon_greedy.
            target: name of target type e.g. greedy.
            imputation_method: name of method to impute values at test time
                for states not present during training,
                e.g. near_neighbours or random.
        """
        super().__init__(
            action_space=action_space,
            state_space=state_space,
            learning_rate=learning_rate,
            gamma=gamma,
            initialisation_strategy=initialisation_strategy,
            behaviour=behaviour,
            target=target,
            imputation_method=imputation_method,
            update_no_op=update_no_op,
        )

        self._state_action_values = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )
        self._latest_state_action_values = {
            self._id_state_mapping[i]: action_values
            for i, action_values in enumerate(self._state_action_values)
        }

    @property
    def state_action_values(self) -> Dict[Tuple[int, int], np.ndarray]:
        if self._training:
            values = {
                self._id_state_mapping[i]: action_values
                for i, action_values in enumerate(self._state_action_values)
            }
            return values
        else:
            return self._latest_state_action_values

    def _impute_near_neighbours(
        self,
        state: Tuple[int, int],
        excess_state_mapping: Dict[Tuple[int, int], List[Tuple[int, int]]],
        store_imputation: bool,
    ):
        """method to impute values for new state that has no entry in table
        using average of values in table that are near neighbours (directly reachable).

        Args:
            state: new state for which value is being imputed.
            excess_state_mapping: mapping from state to near neighbours.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.

        Returns:
            imputed_value for state.
        """
        near_neighbour_ids = [
            self._state_id_mapping[s] for s in excess_state_mapping[state]
        ]
        neighbour_state_action_values = [
            copy.deepcopy(self._state_action_values[s_id])
            for s_id in near_neighbour_ids
        ]
        state_action_values = np.mean(neighbour_state_action_values, axis=0)
        return state_action_values

    def _impute_randomly(self, state: Tuple[int, int], store_imputation: bool):
        """method to impute values for new state that has no entry in table
        by initialising randomly

        Args:
            state: new state for which value is being imputed.
            store_imputation: whether to compute for single-use or store
                as part of model for future use.

        Returns:
            imputed_value for state.
        """
        state_action_values = np.random.normal(size=len(self._action_space))

        if store_imputation:
            self._store_imputation(state=state, imputation=state_action_values)
        return state_action_values

    def _store_imputation(self, state, imputation):
        self._state_id_mapping[state] = len(self._state_id_mapping)
        self._id_state_mapping[len(self._id_state_mapping)] = state
        self._state_space.append(state)

        self._state_action_values = np.vstack(
            (self._state_action_values, imputation.reshape(1, len(self._action_space)))
        )
        self._state_visitation_counts[state] = 0

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        initialisation_strategy_name = list(initialisation_strategy.keys())[0]
        if isinstance(initialisation_strategy_name, (int, float)):
            return initialisation_strategy_name * np.ones(
                (len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.RANDOM_UNIFORM:
            return np.random.rand(len(self._state_space), len(self._action_space))
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(
                loc=0, scale=0.1, size=(len(self._state_space), len(self._action_space))
            )
        elif initialisation_strategy_name == constants.ZEROS:
            return np.zeros((len(self._state_space), len(self._action_space)))
        elif initialisation_strategy_name == constants.ONES:
            return np.ones((len(self._state_space), len(self._action_space)))

    def step(
        self,
        state: Tuple[int, int],
        action: int,
        reward: float,
        new_state: Tuple[int, int],
        active: bool,
    ) -> None:
        """Update state-action values.

        Make q-learning update:
        Q(s_t, a_t) <- Q(s_t, a_t) + alpha * [
                            r_{t+1}
                            + gamma * max_a(Q(s_{t+1}, a))
                            - Q(s_t, at_t)
                            ]

        Args:
            state: state before update.
            action: action taken by agent.
            reward: scalar reward received from environment.
            new_state: next state.
            active: whether episode is still ongoing.
        """
        self._state_visitation_counts[state] += 1

        if state == new_state and not self._update_no_op:
            return

        if active:
            discount = self._gamma
        else:
            discount = 0

        state_id = self._state_id_mapping[state]
        self._step(
            state_id=state_id,
            action=action,
            reward=reward,
            discount=discount,
            new_state=new_state,
        )

        next(self._learning_rate)

    def _step(
        self,
        state_id,
        action,
        reward,
        discount,
        new_state,
    ):
        initial_state_action_value = self._state_action_values[state_id][action]

        if new_state not in self._state_id_mapping and self._allow_state_instantiation:
            self._impute_randomly(state=new_state, store_imputation=True)

        updated_state_action_value = (
            initial_state_action_value
            + self._learning_rate.value
            * (
                reward
                + discount * self._max_state_action_value(state=new_state)
                - initial_state_action_value
            )
        )
        self._state_action_values[state_id][action] = updated_state_action_value
