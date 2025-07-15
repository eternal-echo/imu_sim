import os
import math
from gnss_ins_sim.sim import imu_model, ins_sim

# IMU / 传感器参数
fs = 100.0           # IMU 频率
fs_gps = 0.0         # 不生成 GPS
fs_mag = 0.0         # 不生成 Magnetometer


motion_def_path = os.path.abspath('.//datasets//motion_defs')
raw_data_path = os.path.abspath('.//datasets//gnss_ins_sim_test//')
results_path = os.path.abspath('.//datasets//datasets//')

# 运动定义文件（CSV），格式同官方 my_test.csv
motion_def_file = os.path.join(motion_def_path, 'my_test.csv')

def gen_imu():
    # 选择官方 mid-accuracy 6 轴 IMU（仅 accel + gyro）
    imu = imu_model.IMU(accuracy='mid-accuracy', axis=9, gps=False)

    # 创建仿真：ref_frame=0 表示 NED 导航系，IMU 输出仍在 body frame
    sim = ins_sim.Sim([fs, fs_gps, fs_mag],
                      motion_def_file,
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,
                      algorithm=None)

    # num_times 就是告诉这个方法“要把仿真跑几次”
    sim.run(1)

    # 查看可用数据名称
    available_data = sim.get_names_of_available_data()
    print("Available Data:", available_data)

    # 修改位置初始值为0
    ref_pos = sim.get_data(['ref_pos'])[0]
    ref_pos -= ref_pos[0]
    print("Modified ref_pos:\n", ref_pos[:5])  # 打印前5行修改后的参考位置
    # 将修改后的数据写回数据管理器
    sim.dmgr.add_data(sim.dmgr.ref_pos.name, ref_pos)

    # 保存仿真结果到文件
    os.makedirs(raw_data_path, exist_ok=True)
    sim.results(raw_data_path)

    # 可视化仿真数据：绘制陀螺仪、加速度计和参考位置数据
    sim.plot(['gyro', 'accel', 'ref_pos', 'ref_vel', 'ref_att_euler', 'ref_att_quat'], opt={'ref_pos': '3d'})

if __name__ == '__main__':
    gen_imu()
