"""
This module includes algorithms for various purposes.
"""
from . import online_update, outliers
from ..utility import do_call

class OnlineUpdateRuleFactory:
    """ Factory for update rules.
    """

    @staticmethod
    def create_update_rule(update_rule: str, **construct_args) -> online_update.OnlineUpdateRule:
        """ Create an update rule.

        :param construct_args: The arguments to construct the update rule.
        :return: The update rule.
        """
        if update_rule == 'IncrementalBatchGradientDescent':
            return do_call(online_update.IncrementalBatchGradientDescent, **construct_args)
        else:
            raise ValueError(f'Update rule {update_rule} not supported.')
