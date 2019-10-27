import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MountainCar:
    def __init__(self, start_pos=-0.5, start_vel=0, mass=0.2, friction=0.3, delta_t=0.1):
        self.mass = mass
        self.friction = friction
        self.delta_t = delta_t
        self.gravity = -9.81
        self.position_t = start_pos
        self.velocity_t = start_vel
        self.position_list = []

        self.actions = [-1, 0, 1]# throttle
        self.pos_bound = (-1.2, 0.6)
        self.vel_bound = (-0.07, 0.07)
        self.goal_pos = 0.5
        self.ε = 0 # use optimistic initial value, so it's ok to set epsilon to 0

    def goal(self, state):
        if state[0] >= self.goal_pos: return True
        return False

    # def step(self, state, action):
    #     position, velocity = state
    #     new_velocity = velocity + 0.001 * action - 0.0025 * np.cos(3 * position)
    #     new_velocity = min(max(self.vel_bound[0], new_velocity), self.vel_bound[1])
    #     new_position = position + new_velocity
    #     new_position = min(max(self.pos_bound[0], new_position), self.pos_bound[1])
    #     reward = -1.0
    #     if new_position == self.pos_bound[0]:
    #         new_velocity = 0.0
    #     return (new_position, new_velocity), reward

    def step(self, state, action):
        self.position_t, self.velocity_t = state
        done = False
        # reward = -0.01
        # action_t = 0.2*action
        # velocity_t1 = self.velocity_t + \
        #               (self.gravity * self.mass * np.cos(3*self.position_t)
        #                + (action_t/self.mass)
        #                - (self.friction*self.velocity_t)) * self.delta_t
        # position_t1 = self.position_t + (velocity_t1 * self.delta_t)

        # # Richard Suttons update:
        reward = -1
        velocity_t1 = self.velocity_t + 0.001 * action - 0.0025 * np.cos(3 * self.position_t)
        position_t1 = self.position_t + velocity_t1
        # Check the limit condition (car outside frame)
        if position_t1 < self.pos_bound[0]:
            position_t1 = self.pos_bound[0]
            velocity_t1 = 0
        # Assign the new position and velocity
        self.position_t = position_t1
        self.velocity_t = velocity_t1
        self.position_list.append(position_t1)
        # Reward and done when the car reaches the goal
        if position_t1 >= self.goal_pos:
            reward = +1.0
            done = True
        # Return state_t1, reward, done
        return [position_t1, velocity_t1], reward#, done

    def render(self, file_path='./mountain_car.mp4', mode='mp4'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.

        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """
        # Plot init
        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 0.5), ylim=(-1.1, 1.1))
        ax.grid(False)  # disable the grid
        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
        y_sin = np.sin(3 * x_sin)
        # plt.plot(x, y)
        ax.plot(x_sin, y_sin)  # plot the sine wave
        # line, _ = ax.plot(x, y, 'o-', lw=2)
        dot, = ax.plot([], [], 'ro')
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        _position_list = self.position_list
        _delta_t = self.delta_t

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
            x = _position_list[i]
            y = np.sin(3 * x)
            dot.set_data(x, y)
            time_text.set_text("Time: " + str(np.round(i*_delta_t, 1)) + "s" + '\n' + "Frame: " + str(i))
            return dot, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(self.position_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1/self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1/self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)

# if __name__=="__main__":# test dynamics
#     car = MountainCar()
#     cumulated_reward = 0
#     print("Starting random agent...")
#     for step in range(100):
#         action = np.random.choice(car.actions)
#         observation, reward, done = car.step(action)
#         cumulated_reward += reward
#         if done: break
#     print("Finished after: " + str(step+1) + " steps")
#     print("Cumulated Reward: " + str(cumulated_reward))
#     car.render(file_path='./mountain_car.gif', mode='gif')
