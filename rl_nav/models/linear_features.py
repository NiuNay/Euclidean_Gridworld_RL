import itertools
from typing import Any, Dict, List, Tuple, Type

import numpy as np
from rl_nav import constants
from rl_nav.models import tabular_learner
from rl_nav.utils import feature_utils
from rl_nav.utils import learning_rate_schedules


class LinearFeatureLearner(tabular_learner.TabularLearner):
    """Model with linear combination of features."""

    def __init__(
        self,
        features: Dict[str, Dict[str, Any]],
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

        self._state_action_id_mapping = {
            state + tuple([a]): i
            for i, (state, a) in enumerate(
                itertools.product(self._state_space, self._action_space)
            )
        }

        (
            self._feature_extractors,
            self._feature_dim,
        ) = self._setup_feature_extractors(features=features)

        self._weight_matrix = self._initialise_values(
            initialisation_strategy=initialisation_strategy
        )

        self._state_action_features = np.zeros(
            (len(self._state_space), len(self._action_space), self._feature_dim)
        )

        for s, state in enumerate(self._state_space):
            for a, act in enumerate(self._action_space):
                self._state_action_features[s, a, :] = self._extract_features(
                    state + tuple([act])
                )

        self._wm_change: bool
        self._compute_state_action_values()

    @property
    def state_action_id_mapping(self):
        return self._state_action_id_mapping

    @property
    def _state_action_values(self):
        if self._wm_change:
            self._compute_state_action_values()
        return self.__state_action_values

    @property
    def state_action_values(self):
        if self._training:
            values = {
                self._id_state_mapping[i]: action_values
                for i, action_values in enumerate(self._state_action_values)
            }
            return values
        else:
            return self._latest_state_action_values

    def _setup_feature_extractors(self, features: Dict[str, Dict[str, Any]]):
        if constants.STATE_ACTION_ID in features:
            features[constants.STATE_ACTION_ID][
                constants.STATE_ACTION_ID_MAPPING
            ] = self._state_action_id_mapping

        if constants.COARSE_CODING in features:
            features[constants.COARSE_CODING][constants.STATE_SPACE] = self._state_space
            if features[constants.COARSE_CODING][constants.AUGMENT_ACTIONS]:
                features[constants.COARSE_CODING][
                    constants.AUGMENT_ACTIONS
                ] = self._action_space
            else:
                features[constants.COARSE_CODING][constants.AUGMENT_ACTIONS] = None

        if constants.HARD_CODED_GEOMETRY in features:
            if features[constants.HARD_CODED_GEOMETRY][constants.AUGMENT_ACTIONS]:
                features[constants.HARD_CODED_GEOMETRY][
                    constants.AUGMENT_ACTIONS
                ] = self._action_space
            else:
                features[constants.HARD_CODED_GEOMETRY][
                    constants.AUGMENT_ACTIONS
                ] = None

        if constants.ACTION_ONE_HOT in features:
            features[constants.ACTION_ONE_HOT][
                constants.ACTION_SPACE
            ] = self._action_space

        return feature_utils.get_feature_extractors(features=features)

    def _extract_features(self, state: Tuple[int, int]):
        feature_vector = np.concatenate(
            [extractor(state) for extractor in self._feature_extractors]
        )
        return feature_vector

    def _compute_state_action_values(self):
        self.__state_action_values = np.tensordot(
            self._weight_matrix, self._state_action_features, axes=([0], [2])
        )
        self._wm_change = False

    def _initialise_values(self, initialisation_strategy: str) -> np.ndarray:
        """Initialise values for each state, action pair in state-action space.

        Args:
            initialisation_strategy: name of method used to initialise.

        Returns:
            initial_values: matrix containing state-action id / value mapping.
        """
        initialisation_strategy_name = list(initialisation_strategy.keys())[0]
        if isinstance(initialisation_strategy_name, (int, float)):
            return initialisation_strategy_name * np.ones(self._feature_dim)
        elif initialisation_strategy_name == constants.RANDOM_UNIFORM:
            return np.random.rand(len(self._feature_dim))
        elif initialisation_strategy_name == constants.RANDOM_NORMAL:
            return np.random.normal(loc=0, scale=0.1, size=(self._feature_dim))
        elif initialisation_strategy_name == constants.ZEROS:
            return np.zeros(self._feature_dim)
        elif initialisation_strategy_name == constants.ONES:
            return np.ones(self._feature_dim)

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
        state_action_features = self._state_action_features[state_id][action]
        q_target = self._max_state_action_value(state=new_state)
        delta = reward + discount * q_target - initial_state_action_value
        self._weight_matrix += self._learning_rate.value * delta * state_action_features
        self._wm_change = True
