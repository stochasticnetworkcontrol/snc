from src.snc import BigStepHedgehogGTOAgent
from src.snc import StrategicIdlingGTO
import src.snc.environments.examples as examples
from src.snc.simulation.utils import get_hedgehog_hyperparams


class TestBigStepGTOAgent:

    @staticmethod
    def get_klimov_environment():
        return examples.klimov_model(alpha1=0.2, alpha2=.3, alpha3=.4, alpha4=.5, mu1=1.1,
                                     mu2=2.2, mu3=3.3, mu4=4.4)

    def test_initialised_si_object(self):
        env = self.get_klimov_environment()

        ac_params, wk_params, si_params, po_params, hh_params, _, dp_params, name \
            = get_hedgehog_hyperparams()

        agent = BigStepHedgehogGTOAgent(
            env,
            1,
            wk_params,
            hh_params,
            ac_params,
            po_params,
            dp_params,
            name
        )
        agent.perform_offline_calculations()

        assert isinstance(agent.strategic_idling_object, StrategicIdlingGTO)
