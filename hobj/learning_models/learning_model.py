class LearningModel(object):
    """
    This class is meant to be subclassed. As long as the functions below are implemented (and returns the correct dtype),
    the LearningModel can be scored on any benchmark in this repo.
    """

    def __init__(self, learner_id: str):
        """
        :param learner_id:  a string that identifies the learning model.
        """
        self.learner_id = learner_id
        return

    def reset(self) -> None:
        """
        This function is called at the beginning of each simulated behavioral session.
        It should reset the learning model to an initial state.
        :return:
        """
        return

    def respond(self, image_url: str) -> int:
        """
        This function takes the current stimulus image (given by its image_url) and returns an action (parameterized by an integer).
        :param image_url: A public URL to the image.
        :return:
        """

        action = 0
        return action

    def learn(self, reward: float) -> None:
        """
        This function takes a reward from the environment. One may use this reward to update the learning model.
        This function is called by the environment, after the call to respond().
        :return:
        """

        return
