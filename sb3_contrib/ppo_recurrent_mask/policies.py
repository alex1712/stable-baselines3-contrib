from sb3_contrib.common.RecurrentMaskable.policies import (
    RecurrentMaskableActorCriticCnnPolicy,
    RecurrentMaskableActorCriticPolicy,
    RecurrentMaskableMultiInputActorCriticPolicy,
)

MlpLstmPolicy = RecurrentMaskableActorCriticPolicy
CnnLstmPolicy = RecurrentMaskableActorCriticCnnPolicy
MultiInputLstmPolicy = RecurrentMaskableMultiInputActorCriticPolicy
