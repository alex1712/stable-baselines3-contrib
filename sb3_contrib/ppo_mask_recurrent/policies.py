from sb3_contrib.common.recurrent.maskable.policies import (
    MaskableRecurrentActorCriticCnnPolicy,
    MaskableRecurrentActorCriticPolicy,
    MaskableRecurrentMultiInputActorCriticPolicy,
)

MlpLstmPolicy = MaskableRecurrentActorCriticPolicy
CnnLstmPolicy = MaskableRecurrentActorCriticCnnPolicy
MultiInputLstmPolicy = MaskableRecurrentMultiInputActorCriticPolicy
