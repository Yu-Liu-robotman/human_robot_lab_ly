from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg

from legged_lab.rsl_rl import RslRlDualPolicyDistillationAlgorithmCfg, RslRlDualTeacherMoEStudentCfg


@configclass
class G1RslRlDualDistillRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "DualPolicyDistillationRunner"
    num_steps_per_env = 32
    max_iterations = 10000
    save_interval = 100
    experiment_name = "g1_amp_dual_distill"

    obs_groups = {
        "policy": ["policy"],
        "teacher_0": ["teacher_0"],
        "teacher_1": ["teacher_1"],
        "selector": ["selector"],
    }

    policy = RslRlDualTeacherMoEStudentCfg(
        class_name="DualTeacherMoEStudent",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        activation="elu",
        num_experts=2,
        moe_gate_hidden_dims=[512, 256, 128],
        critic_moe=False,
        teacher_policy_class_name="ActorCritic",
        teacher_policy={
            "init_noise_std": 1.0,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "activation": "elu",
        },
        teacher_0_obs_group="teacher_0",
        teacher_1_obs_group="teacher_1",
        selector_obs_group="selector",
        selector_obs_index=0,
        selector_threshold=0.5,
    )

    algorithm = RslRlDualPolicyDistillationAlgorithmCfg(
        class_name="DualPolicyDistillation",
        num_learning_epochs=5,
        gradient_length=1,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        mse_loss_coef=0.1,
        kl_loss_coef=0.5,
        optimizer="adam",
        teacher_0_checkpoint_path="",
        teacher_1_checkpoint_path="",
        strict_teacher_loading=True,
    )
