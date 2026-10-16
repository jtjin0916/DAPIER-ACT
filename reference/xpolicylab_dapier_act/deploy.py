import numpy as np


TEMPORAL_AGG_K = 0.01


def temporal_ensemble(action_list):
    if len(action_list) == 0:
        raise RuntimeError(
            "No actions available for temporal ensembling"
        )

    weights = np.exp(
        -TEMPORAL_AGG_K
        * np.arange(
            len(action_list),
            dtype=np.float32,
        )
    )

    weights = (
        weights
        / weights.sum()
    )

    result = {}

    keys = action_list[0].keys()

    for key in keys:
        values = np.stack(
            [
                np.asarray(
                    action[key],
                    dtype=np.float32,
                )
                for action in action_list
            ],
            axis=0,
        )

        weight_shape = (
            len(weights),
        ) + (
            1,
        ) * (
            values.ndim - 1
        )

        result[key] = (
            values
            * weights.reshape(
                weight_shape
            )
        ).sum(
            axis=0
        ).astype(
            np.float32
        )

    return result


def eval_one_episode(
    TASK_ENV,
    model_client,
):
    model_client.call(
        func_name="reset"
    )

    action_history = []
    timestep = 0

    while not TASK_ENV.is_episode_end():
        obs = TASK_ENV.get_obs()

        model_client.call(
            func_name="update_obs",
            obs=obs,
        )

        action_chunk = model_client.call(
            func_name="get_action"
        )

        if len(action_chunk) == 0:
            raise RuntimeError(
                "Policy returned empty action chunk"
            )

        action_history.append(
            (
                timestep,
                action_chunk,
            )
        )

        valid_history = []
        current_actions = []

        for (
            start_timestep,
            chunk,
        ) in action_history:
            chunk_index = (
                timestep
                - start_timestep
            )

            if (
                chunk_index >= 0
                and chunk_index
                < len(chunk)
            ):
                valid_history.append(
                    (
                        start_timestep,
                        chunk,
                    )
                )

                current_actions.append(
                    chunk[
                        chunk_index
                    ]
                )

        action_history = (
            valid_history
        )

        action = temporal_ensemble(
            current_actions
        )

        if (
            timestep < 5
            or timestep % 50 == 0
        ):
            print(
                "[DAPIER_ACT TEMPORAL]",
                f"t={timestep}",
                f"candidates={len(current_actions)}",
                f"history={len(action_history)}",
            )

        TASK_ENV.take_action(
            action
        )

        timestep += 1


def eval_one_episode_batch(
    TASK_ENV,
    model_client,
):
    model_client.call(
        func_name="reset"
    )

    while not TASK_ENV.is_episode_end():
        env_idx_list = (
            TASK_ENV
            .get_running_env_idx_list()
        )

        obs_list = (
            TASK_ENV
            .get_obs_batch(
                env_idx_list
            )
        )

        model_client.call(
            func_name="update_obs_batch",
            obs=obs_list,
        )

        actions = model_client.call(
            func_name="get_action_batch",
            obs=env_idx_list,
        )

        chunk_size = len(
            actions[0]
        )

        for action_idx in range(
            chunk_size
        ):
            current_action_list = [
                env_actions[
                    action_idx
                ]
                for env_actions
                in actions
            ]

            TASK_ENV.take_action_batch(
                current_action_list,
                env_idx_list,
            )

            if (
                TASK_ENV.is_episode_end()
                or action_idx + 1
                == chunk_size
            ):
                break

            running = set(
                TASK_ENV
                .get_running_env_idx_list()
            )

            active_batch_idx = [
                i
                for i, env_idx
                in enumerate(
                    env_idx_list
                )
                if env_idx
                in running
            ]

            actions = [
                actions[i]
                for i
                in active_batch_idx
            ]

            env_idx_list = [
                env_idx_list[i]
                for i
                in active_batch_idx
            ]

            model_client.call(
                func_name="update_obs_batch",
                obs=(
                    TASK_ENV
                    .get_obs_batch(
                        env_idx_list
                    )
                ),
            )
