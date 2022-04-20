from config_manager import config_field, config_template
from rl_nav import constants


class RLNavConfigTemplate:

    _gaussian_statistics_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.MEAN, key=constants.GAUSSIAN_MEAN, types=[float, int]
            ),
            config_field.Field(
                name=constants.VARIANCE,
                key=constants.GAUSSIAN_VARIANCE,
                types=[float, int],
                requirements=[lambda x: x >= 0],
            ),
        ],
        level=[constants.REWARD_ATTRIBUTES, constants.GAUSSIAN],
        dependent_variables=[constants.STATISTICS],
        dependent_variables_required_values=[[constants.GAUSSIAN]],
    )

    _reward_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.AVAILABILITY,
                types=[int, str],
                requirements=[
                    lambda x: x == constants.INFINITE or (isinstance(x, int) and x > 0)
                ],
            ),
            config_field.Field(
                name=constants.STATISTICS,
                types=[str],
                requirements=[lambda x: x in [constants.GAUSSIAN]],
            ),
        ],
        level=[constants.REWARD_ATTRIBUTES],
        nested_templates=[_gaussian_statistics_template],
    )

    _random_uniform_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.LOWER_BOUND, types=[float, int]),
            config_field.Field(name=constants.UPPER_BOUND, types=[float, int]),
        ],
        level=[constants.INITIALISATION, constants.RANDOM_UNIFORM],
        dependent_variables=[constants.INITIALISATION_TYPE],
        dependent_variables_required_values=[[constants.RANDOM_UNIFORM]],
    )

    _random_normal_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.MEAN, types=[float, int]),
            config_field.Field(name=constants.VARIANCE, types=[float, int]),
        ],
        level=[constants.INITIALISATION, constants.RANDOM_NORMAL],
        dependent_variables=[constants.INITIALISATION_TYPE],
        dependent_variables_required_values=[[constants.RANDOM_NORMAL]],
    )

    _initialisation_template = config_template.Template(
        fields=[
            config_field.Field(
                name=constants.TYPE,
                types=[str],
                requirements=[
                    lambda x: x
                    in [
                        constants.RANDOM_UNIFORM,
                        constants.RANDOM_NORMAL,
                        constants.ZEROS,
                        constants.ONES,
                    ]
                    or isinstance(x, (int, float))
                ],
                key=constants.INITIALISATION_TYPE,
            ),
        ],
        level=[constants.INITIALISATION],
        nested_templates=[
            _random_uniform_template,
            _random_normal_template,
        ],
    )

    base_template = config_template.Template(
        fields=[
            config_field.Field(name=constants.SEED, types=[int]),
            config_field.Field(
                name=constants.MODEL,
                types=[str],
                requirements=[lambda x: x in [constants.Q_LEARNING]],
            ),
            config_field.Field(
                name=constants.ENVIRONMENT,
                types=[str],
                requirements=[lambda x: x in [constants.ESCAPE_ENV]],
            ),
            config_field.Field(name=constants.MAP_PATH, types=[str]),
            config_field.Field(
                name=constants.REPRESENTATION,
                types=[str],
                requirements=[
                    lambda x: x in [constants.AGENT_POSITION, constants.PIXEL]
                ],
            ),
            config_field.Field(name=constants.START_POSITION, types=[type(None), list]),
            config_field.Field(
                name=constants.REWARD_POSITIONS,
                types=[list],
                requirements=[lambda x: all([isinstance(y, list) for y in x])],
            ),
            config_field.Field(
                name=constants.BEHAVIOUR,
                types=[str],
                requirements=[lambda x: x in [constants.EPSILON_GREEDY]],
            ),
            config_field.Field(
                name=constants.TARGET,
                types=[str],
                requirements=[lambda x: x in [constants.GREEDY]],
            ),
            config_field.Field(
                name=constants.LEARNING_RATE,
                types=[float, int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.DISCOUNT_FACTOR,
                types=[float, int],
                requirements=[lambda x: x > 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.EPSILON,
                types=[float, int],
                requirements=[lambda x: x >= 0 and x <= 1],
            ),
            config_field.Field(
                name=constants.NUM_STEPS, types=[int], requirements=[lambda x: x > 0]
            ),
            config_field.Field(
                name=constants.EPISODE_TIMEOUT,
                types=[int, type(None)],
                requirements=[lambda x: x is None or x > 0],
            ),
            config_field.Field(
                name=constants.VISUALISATION_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.TEST_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
            config_field.Field(
                name=constants.ROLLOUT_FREQUENCY,
                types=[int],
                requirements=[lambda x: x > 0],
            ),
        ],
        nested_templates=[
            _initialisation_template,
            _reward_template,
        ],
    )