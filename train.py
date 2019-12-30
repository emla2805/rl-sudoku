import tensorflow as tf

from tf_agents.networks import network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from sudoku_env import SudokuEnvironment


if __name__ == "__main__":
    train_py_env = SudokuEnvironment()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_py_env = SudokuEnvironment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100,)
    num_iterations = 20000

    initial_collect_steps = 1000
    collect_steps_per_iteration = 1
    replay_buffer_max_length = 100000

    batch_size = 64
    learning_rate = 1e-3
    log_interval = 200

    num_eval_episodes = 30
    eval_interval = 1000

    class MyQNetwork(network.Network):
        def __init__(self, input_tensor_spec, action_spec, name="MyQnet"):
            super(MyQNetwork, self).__init__(
                input_tensor_spec=input_tensor_spec,
                state_spec=(),
                name=name)
            action_spec = tf.nest.flatten(action_spec)[0]
            num_actions = action_spec.maximum - action_spec.minimum + 1

            self._forward = tf.keras.Sequential([
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(num_actions),

            ])

        def call(self, observation, step_type=None, network_state=()):
            logits = self._forward(
                observation)
            return logits, network_state

    q_net = MyQNetwork(train_env.observation_spec(), train_env.action_spec())

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize() 

    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

    def compute_avg_return(environment, policy, num_episodes=10):
        total_return = 0.0
        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    def collect_step(environment, policy, buffer):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        buffer.add_batch(traj)

    def collect_data(env, policy, buffer, steps):
        for _ in range(steps):
            collect_step(env, policy, buffer)

    collect_data(train_env, random_policy, replay_buffer, steps=100)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)
    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)
