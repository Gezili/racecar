import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, MultiLineString, LinearRing

#np.random.seed(0)

COLLUSION_REWARD = -5
CROSS_CHECKPOINT_REWARD = 1

class Car:

    def __init__(self, starting_x, starting_y, starting_angle):

        self.vel = 0
        self.x = starting_x
        self.y = starting_y
        self.facing_direction = starting_angle
        self.ang_vel = 0
        self.max_ang_vel = 0.3
        self.accel_factor = 0.005
        self.ang_accel_factor = 0.01
        self.d_x = 0.008
        self.d_y = 0.004

        #Generate body and position of car at rest
        self.run_time_step(pre_gen=True)

        self.max_vel = 0.05
        self.max_ang_vel = 0.1

    '''
    Input is a one-hot vector
    [input_forward, input_turn_right, input_backward, input_turn_left]
    '''

    def run_time_step(self, input=None, pre_gen=False):

        if not pre_gen:

            vel = self.vel + self.accel_factor*(input[0] - input[2])

            if abs(vel) > self.max_vel:
                vel = self.max_vel*np.sign(vel)

            self.vel = vel
            ang_vel = self.ang_vel + self.ang_accel_factor*(input[1] - input[3])

            if abs(ang_vel) > self.max_ang_vel:
                ang_vel = self.max_ang_vel*np.sign(ang_vel)

            self.ang_vel = ang_vel

            self.facing_direction = (self.facing_direction + self.ang_vel) % (2*np.pi)
            self.x += self.vel*np.cos(self.facing_direction)
            self.y += self.vel*np.sin(self.facing_direction)

        polygon_points = np.array([
            [self.d_x, self.d_y],
            [self.d_x, -self.d_y],
            [-self.d_x, -self.d_y],
            [-self.d_x, self.d_y]
        ])

        cos_theta = np.cos(self.facing_direction)
        sin_theta = np.sin(self.facing_direction)

        rotation_matrix = np.array([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ])

        polygon = np.matmul(rotation_matrix, polygon_points.T).T

        self.body = Polygon(polygon + (self.x, self.y))
        plt.plot(*self.body.exterior.xy)

class Environment:

    def __init__(self, temperature, num_checkpoints = 50):

        self.temperature = temperature
        self.num_checkpoints = num_checkpoints
        self.max_reward_reached = 0
        self._generate_shape()
        self.game_over_flag = False
        self.time_step = 0

    def _generate_shape(self):

        init_coords_thetas = np.linspace(0, 2*np.pi, self.num_checkpoints + 1)[:-1]
        init_coords_thetas = init_coords_thetas + \
            (self.temperature/self.num_checkpoints)*np.random.random(self.num_checkpoints)

        init_coords_cartesian = [(np.cos(x), np.sin(x)) for x in init_coords_thetas]
        outer_points = init_coords_cartesian*np.array(
            [
                [
                    list(
                        1 - self.temperature*np.random.rand(
                            self.num_checkpoints
                        )
                    )
                ]*2
            ]
        )[0].T

        inner_points = outer_points*(0.6 - (self.temperature/self.num_checkpoints)*self.temperature*np.random.rand(self.num_checkpoints, 2))


        self.checkpoints = [
            LineString([outer_points[i], inner_points[i]])
            for i in range(self.num_checkpoints)
        ]

        self.inner_polygon = Polygon(inner_points)
        self.outer_polygon = Polygon(outer_points)

        starting_line_y = outer_points[0][1] - inner_points[0][1]
        starting_line_x = outer_points[0][0] - inner_points[0][0]

        starting_angle = np.arctan(starting_line_x/starting_line_y)
        starting_y = (outer_points[0][1] + inner_points[0][1])/2
        starting_x = (outer_points[0][0] + inner_points[0][0])/2

        self.car = Car(starting_x, starting_y, starting_angle)

    def plot_map(self):

        #plt.close()
        plt.plot(*self.inner_polygon.exterior.xy)
        plt.plot(*self.outer_polygon.exterior.xy)

        for checkpoint in self.checkpoints:
            coords = np.array(checkpoint.coords)
            u = coords[:,0]
            v = coords[:,1]

            plt.plot(u, v)

        plt.xticks([])
        plt.yticks([])

        plt.show()
        #plt.savefig(f'./track_temp_{self.temperature}.jpg')

    def _check_inside(self):
        return self.outer_polygon.contains(self.car.body) and not self.car.body.intersects(self.inner_polygon)

    def assign_reward(self, prev_car, cur_car):

        if not self._check_inside():
            self.game_over_flag = True
            return COLLUSION_REWARD

        car_travel_line = LineString([
            (prev_car[0], prev_car[1]),
            (cur_car[0], cur_car[1])
        ])

        coords = np.array(car_travel_line.coords)
        u = coords[:,0]
        v = coords[:,1]

        plt.plot(u, v)

        checkpoint_reached = -1

        for checkpoint in range(1, 6):

            checkpoint_of_interest = self.checkpoints[
                (self.max_reward_reached + checkpoint) % self.num_checkpoints
            ]

            if car_travel_line.intersects(checkpoint_of_interest):
                checkpoint_reached = checkpoint

        if checkpoint_reached != -1:
            self.max_reward_reached += checkpoint_reached

        return CROSS_CHECKPOINT_REWARD*checkpoint_reached if checkpoint_reached != -1 else 0

    def run_time_step(self):

        prev_car = (self.car.x, self.car.y)

        #Make measurement
        inputs = self.return_measurements()

        #generate input sequence
        #TODO self.model.generate_output
        output = np.random.choice(2,4)

        #Run simulator one time step
        self.car.run_time_step(output)

        #Get reward
        reward = self.assign_reward(prev_car, (self.car.x, self.car.y))

        if reward != 0:
            print(reward)

        #Update model
        #TODO self.model.give(reward)
        #TODO self.model.run_backprop()

        self.time_step += 1

    def return_measurements(self):

        max_line_length = 2
        num_lines_traced = 32

        angles = np.linspace(0, 2*np.pi, num_lines_traced + 1)[:-1]
        rays = []

        for angle in angles:

            endpoints = (
                self.car.x + max_line_length*np.cos(angle),
                self.car.y + max_line_length*np.sin(angle)
            )

            rays.append(LineString([
                (self.car.x, self.car.y), endpoints
            ]))

        min_distances = []

        for ray in rays:

            ray_intersects = []
            inner_intersect = ray.intersection(self.inner_polygon.boundary)
            if inner_intersect.__class__.__name__ == 'Point':
                ray_intersects.append(inner_intersect)
            elif inner_intersect.__class__.__name__ == 'MultiPoint':
                ray_intersects += list(inner_intersect)

            outer_intersect = ray.intersection(self.outer_polygon.boundary)
            if outer_intersect.__class__.__name__ == 'Point':
                ray_intersects.append(outer_intersect)
            elif outer_intersect.__class__.__name__ == 'MultiPoint':
                ray_intersects += list(outer_intersect)

            distances = []

            for intersect in ray_intersects:

                distances.append(
                    np.sqrt(
                        (intersect.coords[0][0] - self.car.x)**2 +
                        (intersect.coords[0][1] - self.car.y)**2
                    )
                )

            if self.time_step % 1000 == 0:
                min_ray_dot = ray_intersects[distances.index(min(distances))]
                plt.scatter(min_ray_dot.coords[0][0], min_ray_dot.coords[0][1])

                coords = np.array(ray.coords)
                u = coords[:,0]
                v = coords[:,1]

                plt.plot(u,v)

            min_distances.append(min(distances))

        return min_distances

if __name__ == '__main__':

    env = Environment(temperature=0)


    #while not env.game_over_flag:
    for i in range(1):
        env.run_time_step()
        #print(env.time_step)

    env.plot_map()