"""Creates data set from a defined car-following model

"""
import numpy as np
import pandas as pd
import random


class IDMDataGenerator():
    """Intelligent Driver Model (IDM) controller.

    Attributes
    ----------
    sample_size : int
        size of the sample
    v0 : float
        desirable speed, in m/s
    T : float
        safe time headway, in s
    a : float
        maximum acceleration, in m/s2
    b : float
        comfortable deceleration, in m/s2
    delta : float
        acceleration exponent
    s0 : float
        minimum gap, in m
    dt : float
        time step, in s
    """

    def __init__(self,
                 sample_size=1e6,
                 v0=30,
                 T=1,
                 a=1,
                 b=1.5,
                 delta=4,
                 s0=2,
                 dt=0.1):

        self.sample_size = sample_size
        self.v0 = v0
        self.T = T
        self.a = a
        self.b = b
        self.delta = delta
        self.s0 = s0
        self.dt = dt

    def get_accel(self, s, v, lead_vel):
        """Returns IDM acceleration.

        Parameters:
        -----------
        s : float
            bumper to bumper distance between the vehicle and its leading
            vehicle in m
        v : float
            speed of the vehicle in m/s
        lead_v : float
            speed of the leading vehicle in m/s

        Returns:
        --------
        float
            acceleration of the vehicle in m/s2
        """
        if abs(s) < 1e-3:
            s = 1e-3

        if lead_vel is None:  # no leading vehicle
            s_star = 0
        else:
            s_star = self.s0 + max(
                0, v * self.T + v * (v - lead_vel) /
                   (2 * np.sqrt(self.a * self.b)))

        return self.a * (1 - (v / self.v0) ** self.delta - (s_star / s) ** 2)

    def get_rand(self):
        """Returns random numbers generated for gap, speed, and lead speed.

        Returns:
        --------
        s : float
            bumper to bumper distance between the vehicle and its leading
            vehicle in m
        v : float
            speed of the vehicle in m/s
        lead_v : float
            speed of the leading vehicle in m/s
        """
        #a random value for gap in the range of [self.s0, 50]
        s = random.uniform(self.s0, 50)
        # a random value for velocity in the range of [0,35]
        v = random.uniform(0, 35)
        # a random value for leading vehicle velocity in the range of [0,35]
        lead_v = random.uniform(0, 35)

        return s, v, lead_v

    def generate_data(self):
        """Generates a dataset and dumps it into data.dat.
        """
        data = np.zeros(shape=(self.sample_size, 6))
        for i in range(self.sample_size):
            s, v, lead_v = self.get_rand()
            acc = self.get_accel(s, v, lead_v)
            new_v = v + acc * self.dt
            delta_x = v * self.dt + 0.5 * acc * self.dt ** 2
            data[i] = [s, v, lead_v, acc, new_v, delta_x]

        # # dump the np array
        # data.dump('data.dat')

        # save as a csv file as well
        columns = ["gap", "speed", "lead_speed", "acceleration", "new_speed",
                   "delta_x"]
        df = pd.DataFrame(data, columns=columns)
        df.to_csv("data.csv")


if __name__ == '__main__':
    generator = IDMDataGenerator(sample_size=3000000)
    generator.generate_data()
