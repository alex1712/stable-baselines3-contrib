from sb3_contrib.common.recurrent_maskable.policies import (
    RecurrentMaskableActorCriticCnnPolicy,
    RecurrentMaskableActorCriticPolicy,
    RecurrentMaskableMultiInputActorCriticPolicy,
)

MlpLstmMaskPolicy = RecurrentMaskableActorCriticPolicy
CnnLstmMaskPolicy = RecurrentMaskableActorCriticCnnPolicy
MultiInputLstmMaskPolicy = RecurrentMaskableMultiInputActorCriticPolicy
