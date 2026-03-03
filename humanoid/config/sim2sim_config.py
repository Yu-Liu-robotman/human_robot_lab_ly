# Copyright (c) 2024-2024, fudeRobot All rights reserved.
#

class Sim2simCfg():
    """Sim configuration."""
    class sim_config:
        # path: supported placeholder : workspace_dir  experiment_name(such as FdBot_ppo)  currentdir(path of the sim2sim_cfg file)
        mjcf_path = '{workspace_dir}/assets/t1pro/mjcf/t1pro.xml'
        model_path = '{workspace_dir}/logs/{experiment_name}/{load_run}/exported/policy.pt'
        param_path = '{workspace_dir}/logs/{experiment_name}/{load_run}/exported/params.yaml'

        sim_duration = 10000.0  # 100000.0
        dt = 0.005      # 实时
        decimation = 4  # 分频

    class robot_config:
    #     kps = [250., 200., 1000., 420, 80., 150.,
    #            250., 200., 1000., 420., 80., 150.,
    #            250, 200, 1000,
    #         #    100, 100, 100, 100, 100, 100, 100,
    #         #    100, 100, 100, 100, 100, 100, 100,
    #            ]
    #     kds = [ 12, 12.0, 50, 23, 4, 7.5,
    #             12, 12.0, 50, 23, 4, 7.5,
    #             12, 12.0, 50,
    #             # 10, 10, 10, 10, 10, 10, 10,
    #             # 10, 10, 10, 10, 10, 10, 10, 
    #            ]

    #     tau_limit = [80, 100, 200, 150, 40, 80,
    #                  80, 100, 200, 150, 40, 80,
    #                  80, 100, 200,
    #                 #  50, 50, 50, 50, 50, 50, 50, 
    #                 #  50, 50, 50, 50, 50, 50, 50,
    #                  ]
        kps = [120., 250., 700., 350, 60., 150.,
               120., 250., 700., 350, 60., 150.,
               120., 250., 700.
               ]
        kds = [ 5, 12.0, 40, 18, 3, 8,
                5, 12.0, 40, 18, 3, 8,
                5, 12.0, 40
               ]

        tau_limit = [80, 100, 200, 150, 40, 80,
                     80, 100, 200, 150, 40, 80,
                     80, 100, 200
                     ]
