import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from gnss_ins_sim.sim import ins_sim
from gnss_ins_sim.sim import imu_model

# %% [OpenIMUSimulation] class 使用OpenIMU的IMU模型进行数据模拟

ideal_imu_err = {
    'gyro_b': np.array([0.0, 0.0, 0.0]),
    'gyro_arw': np.array([0.0, 0.0, 0.0]),
    'gyro_b_stability': np.array([0.0, 0.0, 0.0]),
    'gyro_b_corr': np.array([np.inf, np.inf, np.inf]),
    'accel_b': np.array([0.0, 0.0, 0.0]),
    'accel_vrw': np.array([0.0, 0.0, 0.0]),
    'accel_b_stability': np.array([0.0, 0.0, 0.0]),
    'accel_b_corr': np.array([np.inf, np.inf, np.inf]),
}

class IMUSimulation(ABC):
    """
    Abstract class to handle IMU data simulation, saving and visualization.
    """
    @abstractmethod
    def run_simulation(self, run_count=1):
        """
        Run the simulation for a specified number of iterations.
        :param run_count: Number of times to run the simulation
        """
        pass

    @abstractmethod
    def read_results(self, file_path=None):
        """
        读取保存的模拟结果。

        :param file_path: 保存结果的文件路径。
        :return: 保存的模拟结果数据。
        """
        pass


class OpenIMUSimulation(IMUSimulation):
    """
    Class to handle IMU data simulation, saving and visualization.
    """

    def __init__(self, imu_accuracy=ideal_imu_err, motion_profile=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim', 'motion_def.csv'),
                 fs_imu=100.0, fs_gps=1.0, fs_mag=10.0, output_dir=os.path.join(os.path.dirname(__file__), 'sim', 'datasets')):
        """
        Initialize IMU Simulation object with given parameters.
        :param imu_accuracy: The accuracy of the IMU model (low, mid, high)
            'low-accuracy':
            'mid-accuracy':
            'high-accuracy':
        :param motion_profile: Path to the motion profile definition file
        :param fs_imu: IMU sampling frequency in Hz
        :param fs_gps: GPS sampling frequency in Hz
        :param fs_mag: Magnetometer sampling frequency in Hz
        :param output_dir: Directory to save the generated data
        """
        self.imu_accuracy = imu_accuracy
        self.motion_profile = motion_profile
        self.fs_imu = fs_imu
        self.fs_gps = fs_gps
        self.fs_mag = fs_mag
        self.output_dir = output_dir
        self.sim = None
        self.data_dict = None

    def setup_simulation(self):
        """
        Set up the simulation with the IMU model and motion profile.
        """
        imu = imu_model.IMU(accuracy=self.imu_accuracy, axis=6, gps=True)
        self.sim = ins_sim.Sim(
            [self.fs_imu, self.fs_gps, self.fs_mag],
            self.motion_profile,
            ref_frame=1,  # NED frame
            imu=imu,
            env=None,
            algorithm=None  # No external algorithm for this simulation
        )

    def run_simulation(self, run_count=1):
        """
        Run the simulation for a specified number of iterations.
        :param run_count: Number of times to run the simulation
        """
        if self.sim is None:
            raise RuntimeError("Simulation not set up. Call setup_simulation() first.")
        
        self.sim.run(run_count)

    def plot_results(self, data_types=['ref_pos', 'gyro'], plot_options={'ref_pos': '3d'}):
        """
        绘制轨迹和传感器数据。

        :param data_types: 要绘制的数据类型列表，如['ref_pos', 'gyro']。
        :param plot_options: 绘图选项。
        """
        self.sim.plot(data_types, opt=plot_options)

    def save_results(self, save_dir=None):
        """
        保存模拟结果到文件。

        :param save_dir: 保存结果的文件夹路径。
        """
        if save_dir is None:
            save_dir = self.output_dir
        self.sim.results(save_dir)

    def read_results(self, dataset_dir=None):
        """
        读取保存的模拟结果。

        :param file_path: 保存结果的文件路径。
        :return: 保存的模拟结果数据。
        """
        # 定义磁场模型函数，输入位置，返回磁场
        def mag_model(x):
            return 200 * np.sin(2 * np.pi * x / max(x))

        if dataset_dir is None:
            dataset_dir = self.output_dir
        pos_path = os.path.join(dataset_dir, 'ref_pos.csv')
        vel_path = os.path.join(dataset_dir, 'ref_vel.csv')
        att_quat_path = os.path.join(dataset_dir, 'ref_att_quat.csv')
        att_euler_path = os.path.join(dataset_dir, 'ref_att_euler.csv')
        gyro_path = os.path.join(dataset_dir, 'ref_gyro.csv')
        accel_path = os.path.join(dataset_dir, 'ref_accel.csv')
        time_path = os.path.join(dataset_dir, 'time.csv')

        ref_pos = pd.read_csv(pos_path)
        ref_vel = pd.read_csv(vel_path)
        ref_att_quat = pd.read_csv(att_quat_path)
        ref_att_euler = pd.read_csv(att_euler_path)
        ref_gyro = pd.read_csv(gyro_path)
        ref_accel = pd.read_csv(accel_path)
        acc = pd.read_csv(os.path.join(dataset_dir, 'accel-0.csv'))
        gyro = pd.read_csv(os.path.join(dataset_dir, 'gyro-0.csv'))
        t = pd.read_csv(time_path)

        # 初始位置：ref_pos 需要令初始位置为(0, 0, 0)
        ref_pos = ref_pos - ref_pos.iloc[0]

        # 模拟磁场数据
        mag_x = mag_model(ref_pos['ref_pos_x (m)'].values)
        
        self.data_dict = pd.DataFrame({
            'timestamp(s)': t['time (sec)'],
            'ref_accel_x (m/s^2)': ref_accel['ref_accel_x (m/s^2)'],
            'ref_accel_y (m/s^2)': ref_accel['ref_accel_y (m/s^2)'],
            'ref_accel_z (m/s^2)': ref_accel['ref_accel_z (m/s^2)'],
            'accel_x (m/s^2)': acc['accel_x (m/s^2)'],
            'accel_y (m/s^2)': acc['accel_y (m/s^2)'],
            'accel_z (m/s^2)': acc['accel_z (m/s^2)'],
            'ref_gyro_x (rad/s)': np.deg2rad(ref_gyro['ref_gyro_x (deg/s)']),
            'ref_gyro_y (rad/s)': np.deg2rad(ref_gyro['ref_gyro_y (deg/s)']),
            'ref_gyro_z (rad/s)': np.deg2rad(ref_gyro['ref_gyro_z (deg/s)']),
            'gyro_x (rad/s)': np.deg2rad(gyro['gyro_x (deg/s)']),
            'gyro_y (rad/s)': np.deg2rad(gyro['gyro_y (deg/s)']),
            'gyro_z (rad/s)': np.deg2rad(gyro['gyro_z (deg/s)']),
            # 磁场数据为0
            'mag_x (uT)': mag_x,
            'mag_y (uT)': np.zeros(len(t)),
            'mag_z (uT)': np.zeros(len(t)),
            'ref_pos_x (m)': ref_pos['ref_pos_x (m)'],
            'ref_pos_y (m)': ref_pos['ref_pos_y (m)'],
            'ref_pos_z (m)': ref_pos['ref_pos_z (m)'],
            'ref_vel_x (m/s)': ref_vel['ref_vel_x (m/s)'],
            'ref_vel_y (m/s)': ref_vel['ref_vel_y (m/s)'],
            'ref_vel_z (m/s)': ref_vel['ref_vel_z (m/s)'],
            'ref_Yaw (deg)': ref_att_euler['ref_Yaw (deg)'],
            'ref_Pitch (deg)': ref_att_euler['ref_Pitch (deg)'],
            'ref_Roll (deg)': ref_att_euler['ref_Roll (deg)'],
            'q0 ()': ref_att_quat['q0 ()'],
            'q1': ref_att_quat['q1'],
            'q2': ref_att_quat['q2'],
            'q3': ref_att_quat['q3'],
        })

        return self.data_dict
    
# %% [SimpleIMUSimulation] class 生成一段简单的IMU数据（直线运动）
class SimpleIMUSimulation(IMUSimulation):
    """
    Class to handle simple IMU data simulation, saving and visualization.
    """

    def __init__(self, num_steps=1000, dt=0.01, accel=np.array([0, 0, 9.81]), gyro=np.array([0, 1, 0])):
        """
        Initialize Simple IMU Simulation object with given parameters.
        :param num_steps: Number of time steps to simulate
        :param dt: Time step size
        :param accel: Constant acceleration vector
        :param gyro: Constant angular velocity vector
        """
        self.num_steps = num_steps
        self.dt = dt
        self.accel = accel
        self.gyro = gyro
        self.data = None

    def run_simulation(self, run_count=1):
        """
        Run the simulation for a specified number of iterations.
        :param run_count: Number of times to run the simulation
        """
        self.data = pd.DataFrame({
            'time': np.arange(0, self.num_steps * self.dt, self.dt),
            'accel_x (m/s^2)': np.full(self.num_steps, self.accel[0]),
            'accel_y (m/s^2)': np.full(self.num_steps, self.accel[1]),
            'accel_z (m/s^2)': np.full(self.num_steps, self.accel[2]),
            'gyro_x (rad/s)': np.full(self.num_steps, self.gyro[0]),
            'gyro_y (rad/s)': np.full(self.num_steps, self.gyro[1]),
            'gyro_z (rad/s)': np.full(self.num_steps, self.gyro[2])
        })

    def plot_results(self):
        """
        绘制IMU数据。
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data['time'], self.data['accel_x (m/s^2)'], label="Accelerometer X")
        plt.plot(self.data['time'], self.data['accel_y (m/s^2)'], label="Accelerometer Y")
        plt.plot(self.data['time'], self.data['accel_z (m/s^2)'], label="Accelerometer Z")
        plt.plot(self.data['time'], self.data['gyro_x (rad/s)'], label="Gyroscope X")
        plt.plot(self.data['time'], self.data['gyro_y (rad/s)'], label="Gyroscope Y")
        plt.plot(self.data['time'], self.data['gyro_z (rad/s)'], label="Gyroscope Z")
        plt.title("IMU Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    def read_results(self):
        """
        读取保存的模拟结果。
        :return: 保存的模拟结果数据。
        """
        return self.data
    
    def save_results(self, save_dir=None):
        """
        保存模拟结果到文件。
        :param save_dir: 保存结果的文件夹路径。
        """
        pass

    def setup_simulation(self):
        """
        Set up the simulation with the IMU model and motion profile.
        """
        pass

# %% [IMU] 兼容IMU的数据读取和绘制
from IMU_base import IMU, save_imu_data
    
class RecordFileIMU(IMU):
    def __init__(self, file_path: str=os.path.join(os.path.dirname(os.path.abspath(__file__)),'output', 'imu_data.csv')):
        super().__init__()
        self.file_path = file_path
        self.imu_datasets = pd.read_csv(file_path)
        self.imu_iter = 0

        # assert 是否为空
        assert not self.imu_datasets.empty, "IMU数据为空"

    def init(self):
        pass

    def deinit(self):
        pass

    @save_imu_data
    def get_data(self):
        if self.imu_iter < len(self.imu_datasets):
            data = self.imu_datasets.iloc[[self.imu_iter]]
            self.imu_iter += 1
            return data
        return None
    
    @save_imu_data
    def get_all_data(self):
        return self.imu_datasets
    
    def static_plot(self):
        self.init()
        self.get_all_data()
        # Plot
        self.plotter.update_plots(self.imu_frames['timestamp(s)'].values, [self.imu_frames[['accel_x (m/s^2)', 'accel_y (m/s^2)', 'accel_z (m/s^2)']].values.T,
                                                                          self.imu_frames[['gyro_x (rad/s)', 'gyro_y (rad/s)', 'gyro_z (rad/s)']].values.T,
                                                                          self.imu_frames[['mag_x (uT)', 'mag_y (uT)', 'mag_z (uT)']].values.T])
        self.plotter.show()


def data_generate(sim: IMUSimulation):
    # Initialize the IMU simulation with default settings
    imu_sim = sim

    # Setup and run the simulation
    imu_sim.setup_simulation()
    imu_sim.run_simulation()

    # Save results to CSV
    imu_sim.save_results()

    # Visualize the results
    imu_sim.plot_results()

    # Read the saved results
    results = imu_sim.read_results()
    print(results.head())

def data_test(sim: IMUSimulation):
    # 从模拟数据集中加载IMU数据
    imu_sim = sim
    imu_sim.setup_simulation()
    imu_sim.run_simulation()
    data = imu_sim.read_results()

    # 创建IMU模型并进行轨迹推导
    dt = np.mean(np.diff(data['time']))  # 计算时间步长
    if isinstance(imu_sim, OpenIMUSimulation):
        imu_model = SimpleIMUModel(imu_data=data,
                                   dt=dt,
                                   position=[data['pos_x (m)'][0], data['pos_y (m)'][0], data['pos_z (m)'][0]],
                                    velocity=[data['vel_x (m/s)'][0], data['vel_y (m/s)'][0], data['vel_z (m/s)'][0]],
                                    quaternion=[data['q0 ()'][0], data['q1'][0], data['q2'][0], data['q3'][0]])
    else:
        imu_model = SimpleIMUModel(data, dt)
    imu_model.integrate()

    # 获取真实轨迹用于对比
    if 'pos_x (m)' in data.columns and 'pos_y (m)' in data.columns and 'pos_z (m)' in data.columns:
        true_positions = data[['pos_x (m)', 'pos_y (m)', 'pos_z (m)']].to_numpy()
    else:
        true_positions = None

    # 绘制结果与真实值对比
    imu_model.plot_results(true_positions)

# Example usage of the class
if __name__ == "__main__":
    # data_test(SimpleIMUSimulation())
    # data_test(OpenIMUSimulation())
    # data_generate(OpenIMUSimulation())
    # data_generate(SimpleIMUSimulation())

    imu = SimuIMU()
    imu.init()
    # imu.print_test()
    imu.position_plot(interval=1)

