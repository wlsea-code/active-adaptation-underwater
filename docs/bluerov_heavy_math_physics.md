# BlueROVHeavy 显式控制与循迹：数学和物理原理

本文档参考主程序中的 `docs/bluerov_explicit_controller.md`，但内容对应当前
`aa-projects/uderwater-control` 中已经通过仿真验收的 BlueROVHeavy 实现。重点是
说明代码实际采用的模型、参数和控制链，而不是给出一套脱离实现的通用 ROV 理论。

对应文件：

- `src/underwater_control/assets_heavy.py`
- `src/underwater_control/actions.py`
- `src/underwater_control/controllers.py`
- `src/underwater_control/commands.py`
- `src/underwater_control/trajectories.py`
- `cfg/task/BlueROVHeavyExplicit.yaml`
- `cfg/task/BlueROVHeavyLemniscate.yaml`
- `/home/wlsea2/workspace/aa-robot-models/underwater/BlueROVHeavy.usd`
- `active_adaptation/envs/robots/underwater.py`

完整数据流为：

```text
目标位姿或 Lemniscate 解析参考
  -> 六字段 Pose Reference Contract
  -> 世界系位置/机体系姿态 PD 和前馈
  -> 刚体质量、惯量及附加质量模型
  -> 重力与浮力补偿
  -> 六维目标 wrench
  -> 8 推进器有界加权分配
  -> 目标推力 -> RPM -> throttle
  -> 推进器一阶响应和水动力
  -> Isaac 刚体运动
```

## 1. 坐标系、状态和符号

代码后缀约定：

- `_w`：世界坐标系；
- `_b`：BlueROVHeavy 机体坐标系；
- `_wb`：机体相对于世界的姿态四元数；
- 四元数顺序为 $(w,x,y,z)$；
- `quat_rotate_inverse(q_wb, x_w)` 实现 $R_{bw}x_w$。

状态定义为：

$$
\eta=(p_w,q_{wb}),
\qquad
\nu_b=\begin{bmatrix}v_b\\\omega_b\end{bmatrix}.
$$

代码直接读取的主要状态和参考量为：

| 数学符号 | 代码字段 | 坐标系与含义 |
|---|---|---|
| $p_w$ | `root_link_pos_w` | 当前世界系位置 |
| $q_{wb}$ | `root_link_quat_w` | 当前姿态四元数 |
| $v_w$ | `root_link_lin_vel_w` | 当前世界系线速度 |
| $\omega_b$ | `root_link_ang_vel_b` | 当前机体系角速度 |
| $p_{d,w}$ | `target_position_w` | 目标世界系位置 |
| $q_{d,wb}$ | `target_orientation_wb` | 目标姿态 |
| $v_{d,w}$ | `target_linear_velocity_w` | 目标世界系线速度 |
| $\omega_{d,b}$ | `target_angular_velocity_b` | 目标机体系角速度 |
| $a_{ff,w}$ | `feedforward_linear_acceleration_w` | 世界系线加速度前馈 |
| $\alpha_{ff,b}$ | `feedforward_angular_acceleration_b` | 机体系角加速度前馈 |

六维加速度和 wrench 均按以下顺序排列：

$$
\dot\nu_b=
\begin{bmatrix}a_x&a_y&a_z&\alpha_x&\alpha_y&\alpha_z\end{bmatrix}^{T},
$$

$$
\tau_b=
\begin{bmatrix}F_x&F_y&F_z&M_x&M_y&M_z\end{bmatrix}^{T}.
$$

## 2. BlueROVHeavy 刚体与流体参数

### 2.1 质量和惯量

资产包含一个 `base_link` 和 8 个推进器刚体。原始模型参数为：

| 部件 | 数量 | 单体质量 |
|---|---:|---:|
| `base_link` | 1 | $11.5\ \mathrm{kg}$ |
| `rotor_i` | 8 | $0.01\ \mathrm{kg}$ |

因此 Action 从 PhysX 读取到的总质量为：

$$
m=11.5+8\times0.01=11.58\ \mathrm{kg}.
$$

基体对角惯量为：

$$
J_b=\operatorname{diag}(0.21,\ 0.245,\ 0.245)\ \mathrm{kg\,m^2}.
$$

当前控制器在 `inertia_b: null` 时读取 `base_link` 的惯量，而质量使用 articulation
所有刚体质量之和。这是当前实现边界，不等同于重新把 8 个推进器通过平行轴定理
折算到整机质心后的完整复合惯量。

### 2.2 排水体积、浮力和重力

Heavy 配置使用：

$$
V=0.0116499\ \mathrm{m^3},\qquad
\rho=997\ \mathrm{kg/m^3},\qquad
g_{hydro}=9.8\ \mathrm{m/s^2}.
$$

静水浮力大小为：

$$
F_B=\rho g_{hydro}V
=997\times9.8\times0.0116499
\approx113.827\ \mathrm{N}.
$$

控制 Action 使用 $g_{ctrl}=9.81\ \mathrm{m/s^2}$ 计算整机重力：

$$
F_G=mg_{ctrl}=11.58\times9.81\approx113.600\ \mathrm{N}.
$$

水平姿态下两者相差约：

$$
F_B-F_G\approx0.227\ \mathrm{N}.
$$

所以该参数组合接近中性浮力，但存在很小的正浮力。浮心相对质心的偏置大小为：

$$
d_{BM}=0.01\ \mathrm{m}.
$$

它使倾斜后的浮力产生恢复 roll/pitch 力矩。具体分量和符号由
`UnderwaterRobot` 的水动力坐标约定转换后写入机体系。

### 2.3 附加质量

配置给出的附加质量对角向量为：

$$
M_A=\operatorname{diag}
(5.5,\ 12.7,\ 14.57,\ 0.12,\ 0.12,\ 0.12).
$$

单位前三项为 kg，后三项为 $\mathrm{kg\,m^2}$。启用 `use_added_mass: true`
后，控制器的等效惯性矩阵为：

$$
M=M_{RB}+M_A,
$$

$$
M_{RB}=
\begin{bmatrix}
mI_3&0\\
0&J_b
\end{bmatrix}.
$$

代入当前数值：

$$
M=\operatorname{diag}
(17.08,\ 24.28,\ 26.15,\ 0.33,\ 0.365,\ 0.365).
$$

这说明水体随动效应在 sway 和 heave 方向尤其明显。同样的目标加速度下，控制器
要求的横向和垂向推力会显著大于只考虑干质量时的结果。

### 2.4 线性和二次阻尼

水动力执行层使用：

$$
D_L=\operatorname{diag}
(4.03,\ 6.22,\ 5.18,\ 0.07,\ 0.07,\ 0.07),
$$

$$
D_Q=\operatorname{diag}
(18.18,\ 21.66,\ 36.99,\ 1.55,\ 1.55,\ 1.55).
$$

设相对水流的机体系速度为 $\nu_r$，代码构造的阻尼近似可概括为：

$$
\tau_D=(D_L+D_Q|\nu_r|)\nu_r,
$$

随后作为反向水动力的一部分施加。实际代码还在若干平移和转动耦合位置填入
`hydro_twist_matrix_b`，并执行水动力数据源所需的 Y、Z、pitch、yaw 符号转换，
因此不能在调试时把配置系数未经坐标转换直接与 PhysX wrench 分量比较。

### 2.5 附加质量科氏项

执行层根据：

$$
p_A=M_A\nu_r
$$

计算附加质量动量，并用叉积构造平移和转动科氏项。整体水动力写成：

$$
\tau_{hydro}=-(\tau_A+\tau_C+\tau_D),
$$

其中 $\tau_A=M_A\dot\nu_r$。加速度来自离散差分，并用
`acc_filter_alpha=0.3` 进行一阶滤波。

当前两个 Heavy 任务设置 `compensate_hydrodynamics: false`。因此控制器显式补偿
重力和浮力，并把 $M_A$ 加入目标惯性矩阵，但不会逐步抵消执行层估计出的阻尼和
科氏 wrench。PD 反馈负责修正剩余误差。

## 3. 两种参考输入

Heavy 的普通位姿控制和 Lemniscate 循迹共用完全相同的后端，只在参考生成器不同。

### 3.1 六字段 Pose Reference Contract

两类 Command 均输出：

$$
\mathcal R=
\{p_{d,w},q_{d,wb},v_{d,w},\omega_{d,b},a_{ff,w},\alpha_{ff,b}\}.
$$

Action 只消费这六个字段，不关心参考来自鼠标拖拽、脚本目标还是解析曲线。

### 3.2 鼠标目标和最小加加速度轨迹

拖动目标点后，平移参考采用五次多项式：

$$
p(t)=a_0+a_1t+a_2t^2+a_3t^3+a_4t^4+a_5t^5.
$$

边界条件为：

$$
p(0)=p_0,\quad \dot p(0)=v_0,\quad \ddot p(0)=a_0^{ref},
$$

$$
p(T)=p_g,\quad \dot p(T)=0,\quad \ddot p(T)=0.
$$

重新拖动目标时，规划起点取当前参考轨迹的 $p,\dot p,\ddot p$，所以参考保持
$C^2$ 连续。当前配置使用：

$$
T=4.0\ \mathrm{s}.
$$

姿态参考为：

$$
q_d(t)=\operatorname{SLERP}(q_0,q_g,h(t/T)),
$$

$$
h(u)=10u^3-15u^4+6u^5.
$$

该模式当前不解析计算 SLERP 的角速度和角加速度，因此
$\omega_{d,b}=0$、$\alpha_{ff,b}=0$。

## 4. Lemniscate 3D 解析轨迹

### 4.1 几何曲线

局部曲线定义为：

$$
r(\theta)=
\begin{bmatrix}
a\sin\theta\\
b\sin2\theta\\
h\cos\theta
\end{bmatrix}.
$$

当前 Heavy 配置为：

$$
a=1.5\ \mathrm{m},\qquad
b=0.9\ \mathrm{m},\qquad
h=0.45\ \mathrm{m}.
$$

对相位的一、二阶导数为：

$$
r'(\theta)=
\begin{bmatrix}
a\cos\theta\\
2b\cos2\theta\\
-h\sin\theta
\end{bmatrix},
$$

$$
r''(\theta)=
\begin{bmatrix}
-a\sin\theta\\
-4b\sin2\theta\\
-h\cos\theta
\end{bmatrix}.
$$

曲线先绕世界 Z 轴旋转 $\psi_c$，再平移到中心 $c_w$：

$$
p_{d,w}=c_w+R_z(\psi_c)r(\theta).
$$

reset 时通过：

$$
c_w=p_w(0)-R_z(\psi_c)r(\theta_0)
$$

保证轨迹第一点恰好等于机器人当前位置，不发生参考位置跳变。

### 4.2 相位、速度和加速度

稳定相位角速度为：

$$
\Omega=\frac{2\pi}{T_L},\qquad T_L=45\ \mathrm{s}.
$$

方向参数 $s\in\{-1,1\}$，因此稳定状态：

$$
\dot\theta=s\Omega.
$$

启动前 $T_e=6\ \mathrm{s}$ 内，代码使用五次平滑速度比例：

$$
\sigma(u)=10u^3-15u^4+6u^5,
\qquad u=\operatorname{clip}(t/T_e,0,1).
$$

于是：

$$
\dot\theta=s\Omega\sigma(u),
$$

$$
\ddot\theta=\frac{s\Omega}{T_e}
(30u^2-60u^3+30u^4).
$$

相位通过速度比例的解析积分得到，启动结束后与匀速相位连续。世界系参考速度和
加速度由链式法则得到：

$$
v_{d,w}=R_z(\psi_c)r'(\theta)\dot\theta,
$$

$$
a_{ff,w}=R_z(\psi_c)
\left[r''(\theta)\dot\theta^2+r'(\theta)\ddot\theta\right].
$$

所以 Lemniscate 路径向控制器提供解析速度与加速度前馈，不需要通过有限差分估计。

### 4.3 切线航向

`heading_mode: tangent_yaw` 使机头跟随水平运动切线。设旋转后的曲线导数为
$r'_w=(x',y',z')$，则：

$$
\psi_d=\operatorname{atan2}(s y',s x').
$$

航向对相位的导数为：

$$
\frac{d\psi_d}{d\theta}
=\frac{x'y''-y'x''}{x'^2+y'^2},
$$

目标 yaw rate 为：

$$
\dot\psi_d=\frac{d\psi_d}{d\theta}\dot\theta.
$$

reset 时还会选择 $\psi_c$，使初始水平切线与机器人初始机头对齐。roll 和 pitch
参考保持 reset 时的姿态加配置偏移；当前 `rpy_offset=[0,0,0]`。角加速度前馈仍为
零，但 yaw rate 已写入 $\omega_{d,b}$ 的 Z 分量。

## 5. 位姿误差到目标加速度

### 5.1 平移控制

世界系误差为：

$$
e_{p,w}=p_{d,w}-p_w,
\qquad
e_{v,w}=v_{d,w}-v_w.
$$

命令加速度为：

$$
a_{cmd,w}=K_p^p\odot e_{p,w}
+K_d^p\odot e_{v,w}+a_{ff,w}.
$$

当前参数：

$$
K_p^p=(0.7,0.7,0.8),
\qquad
K_d^p=(2.0,2.0,2.2).
$$

逐轴限幅：

$$
|a_{cmd,w,i}|\le0.2\ \mathrm{m/s^2}.
$$

随后转换到机体系：

$$
a_{cmd,b}=R_{bw}(q_{wb})a_{cmd,w}.
$$

### 5.2 姿态控制

姿态误差四元数与轴角误差为：

$$
q_{e,b}=q_{wb}^{-1}\otimes q_{d,wb},
$$

$$
e_{R,b}=\operatorname{Log}_{SO(3)}(q_{e,b}).
$$

角速度误差为：

$$
e_{\omega,b}=\omega_{d,b}-\omega_b.
$$

角加速度命令：

$$
\alpha_{cmd,b}=K_p^R\odot e_{R,b}
+K_d^R\odot e_{\omega,b}+\alpha_{ff,b}.
$$

Heavy 参数为：

$$
K_p^R=(8,8,12),
\qquad
K_d^R=(3,3,3),
$$

$$
\alpha_{max}=(1.5,1.5,2.0)\ \mathrm{rad/s^2}.
$$

与 6 推进器 BlueROV 不同，Heavy 的 pitch 增益和 pitch 加速度上限不为零，因为
8 推进器布局可以独立产生 $M_y$。

## 6. 目标加速度到推进器 wrench

控制器首先计算：

$$
\tau_{inertia,b}=M
\begin{bmatrix}a_{cmd,b}\\\alpha_{cmd,b}\end{bmatrix}.
$$

然后加入刚体角动量耦合项：

$$
\tau_{gyro,b}=
\begin{bmatrix}
0_3\\
\omega_b\times(J_b\omega_b)
\end{bmatrix}.
$$

Action 组装非推进器外部 wrench：

$$
\tau_{ext,b}=\tau_{buoyancy,b}+\tau_{gravity,b}.
$$

由于动力学约定为：

$$
M\dot\nu_b=\tau_{thruster,b}+\tau_{ext,b},
$$

目标推进器 wrench 为：

$$
\boxed{
\tau_{d,b}=M\dot\nu_{cmd,b}
+\tau_{gyro,b}-\tau_{ext,b}
}.
$$

当前 `compensate_hydrodynamics: false`，因此这里不加入实时 `uw.hydro`。同时
$M_A$ 已经进入 $M$，Action 不会再次把同一个附加质量惯性项作为外力重复补偿。

## 7. 8 推进器几何

### 7.1 位置与方向

Action 不把几何硬编码进控制器，而是在仿真初始化时读取各推进器刚体相对
`base_link` 的位置和局部 +X 推力方向。由当前资产得到：

| 推进器 | $r_i=[x,y,z]$ m | $d_i=[d_x,d_y,d_z]$ | 主要作用 |
|---|---|---|---|
| 0 | $[0.1355,-0.1000,-0.0725]$ | $[c,c,0]$ | surge/sway/yaw |
| 1 | $[0.1355,0.1000,-0.0725]$ | $[c,-c,0]$ | surge/sway/yaw |
| 2 | $[-0.1475,-0.1000,-0.0725]$ | $[-c,c,0]$ | surge/sway/yaw |
| 3 | $[-0.1475,0.1000,-0.0725]$ | $[-c,-c,0]$ | surge/sway/yaw |
| 4 | $[0.1200,-0.2200,-0.0050]$ | $[0,0,1]$ | heave/roll/pitch |
| 5 | $[0.1200,0.2200,-0.0050]$ | $[0,0,1]$ | heave/roll/pitch |
| 6 | $[-0.1200,-0.2200,-0.0050]$ | $[0,0,1]$ | heave/roll/pitch |
| 7 | $[-0.1200,0.2200,-0.0050]$ | $[0,0,1]$ | heave/roll/pitch |

其中：

$$
c=\frac{1}{\sqrt2}\approx0.70710678.
$$

前 4 个是水平 45 度矢量推进器，后 4 个是垂向推进器。

### 7.2 单推进器 wrench

第 $i$ 个推进器标量推力为 $f_i$，则：

$$
F_i=d_i f_i,
$$

$$
M_i=(r_i\times d_i)f_i+k_i d_i f_i.
$$

当前 `reaction_torque_per_thrust: 0.0`，所以 $k_i=0$。对应分配列为：

$$
b_i=
\begin{bmatrix}
d_i\\r_i\times d_i
\end{bmatrix}.
$$

## 8. Heavy 的完整分配矩阵与可控性

令：

$$
f=\begin{bmatrix}f_0&f_1&\cdots&f_7\end{bmatrix}^T,
\qquad
\tau_b=Bf.
$$

代入上一节几何，可得当前 Heavy 的 $6\times8$ 分配矩阵：

$$
B\approx
\begin{bmatrix}
 0.707107& 0.707107&-0.707107&-0.707107& 0&0&0&0\\
 0.707107&-0.707107& 0.707107&-0.707107& 0&0&0&0\\
 0&0&0&0&1&1&1&1\\
 0.051265&-0.051265& 0.051265&-0.051265&-0.22&0.22&-0.22&0.22\\
-0.051265&-0.051265& 0.051265& 0.051265&-0.12&-0.12&0.12&0.12\\
 0.166524&-0.166524&-0.175009& 0.175009&0&0&0&0
\end{bmatrix}.
$$

数值计算得到：

$$
\operatorname{rank}(B)=6.
$$

奇异值约为：

$$
\sigma(B)=
(2.0000,\ 1.41835,\ 1.41803,\ 0.43873,\ 0.34153,\ 0.23935).
$$

因此当前布局可以独立控制全部六个 wrench 通道：

$$
[F_x,F_y,F_z,M_x,M_y,M_z].
$$

这也是 Heavy 配置使用：

```yaml
wrench_command_mask: [1, 1, 1, 1, 1, 1]
```

的物理原因。6 推进器 BlueROV 的矩阵 rank 为 5，需要屏蔽 $M_y$；Heavy 的四个
垂向推进器在前后和左右方向都有力臂，可以分别组合出 heave、roll 和 pitch。

最小奇异值明显小于三个平移相关的大奇异值，表示某些转矩方向的控制权弱于纯
平移方向，但矩阵并不降秩。

## 9. 有界加权推力分配

分配器求解：

$$
\min_f\ \|W(Bf-\tau_d)\|_2^2+\lambda\|f\|_2^2,
$$

约束为：

$$
f_{min}\le f_i\le f_{max}.
$$

Heavy 参数：

$$
W=\operatorname{diag}(1,1,1,5,5,5),
\qquad
\lambda=10^{-6}.
$$

较大的转矩权重使饱和或过驱动情况下优先维护姿态稳定。无边界时的阻尼最小二乘
解为：

$$
f=(B^TW^2B+\lambda I)^{-1}B^TW^2\tau_d.
$$

实际实现不显式求逆，而用 `torch.linalg.solve()`。如果某个推力超出上下限，
active-set 过程会：

1. 把越界推进器固定在边界；
2. 从目标 wrench 中扣除固定推进器贡献；
3. 用其余自由推进器重新分配残余 wrench；
4. 最多迭代 $N+1=9$ 次。

诊断残差为：

$$
\tau_{res}=\tau_d-Bf.
$$

残差大意味着目标超出当前几何、推力范围或死区能够实现的能力。

## 10. Heavy 推进器模型和力常数缩放

### 10.1 throttle 到 RPM

归一化油门 $u\in[-1,1]$，死区：

$$
\delta=0.075.
$$

RPM 分段模型为：

$$
n(u)=
\begin{cases}
3659.9u+345.21,&u>0.075,\\
0,&|u|\le0.075,\\
3494.4u-433.50,&u<-0.075.
\end{cases}
$$

并限制在：

$$
-3900\le n\le3900\ \mathrm{RPM}.
$$

### 10.2 标称 T200 曲线

原始双向二次曲线为：

$$
T_+(n)=9.81
(4.7368\times10^{-7}n^2-1.9275\times10^{-4}n+8.4452\times10^{-2}),
$$

$$
T_-(n)=9.81
(-3.8442\times10^{-7}n^2-1.6186\times10^{-4}n-3.9139\times10^{-2}).
$$

它们对应标称力常数：

$$
k_{nom}=4.4\times10^{-7}.
$$

Heavy 资产为每台推进器配置：

$$
k_H=0.8\times10^{-7}.
$$

执行层实际推力为：

$$
T_H(n)=\frac{k_H}{k_{nom}}T_{T200}(n)
=\frac{0.8}{4.4}T_{T200}(n).
$$

### 10.3 为什么控制配置使用较小的 `thrust_scale`

控制器必须按 Heavy 的实际能力分配推力，否则会认为每台推进器比仿真实际能力
强 5.5 倍。因此控制器配置直接合并比例：

$$
s_H=9.81\frac{0.8}{4.4}
=1.78363636.
$$

控制器内部使用：

$$
T_{ctrl,\pm}(n)=s_H(a_\pm n^2+b_\pm n+c_\pm).
$$

而执行层使用 $9.81$ 的标称曲线后再乘 $k_H/k_{nom}$。两条计算路径数学等价：

$$
1.78363636\,P(n)
=\frac{0.8}{4.4}\times9.81\,P(n).
$$

因此不存在重复缩小；这是让控制器预测推力与 Isaac 实际推力一致的必要处理。

当前单推进器近似极限为：

$$
f_{min}\approx-9.373\ \mathrm{N},
\qquad
f_{max}\approx11.660\ \mathrm{N}.
$$

死区刚外侧的非零推力约为反向 $-0.201\ \mathrm{N}$、正向
$0.262\ \mathrm{N}$。因此很小的目标推力会在零推力与死区边缘推力之间量化。

### 10.4 推力反解

控制器按照：

```text
目标推力 -> 二分法求 RPM -> 分段仿射反解 throttle
```

工作。二次曲线的有效分支经初始化检查保证单调，代码执行 40 次二分。死区附近还
会比较活动分支和零 RPM 的误差，选择更接近目标推力的结果。

## 11. 推进器动态和 Isaac 施力

控制器输出的是目标 throttle。执行层对每台推进器使用时间常数：

$$
\tau_i=0.01\ \mathrm{s}.
$$

离散一阶响应为：

$$
\gamma_i=\exp(-\Delta t_{phys}/\tau_i),
$$

$$
u_{i,k}=\gamma_i u_{i,k-1}+(1-\gamma_i)u_{cmd,i,k}.
$$

Heavy 配置的物理步长为：

$$
\Delta t_{phys}=0.01\ \mathrm{s},
$$

控制/环境步长为：

$$
\Delta t_{ctrl}=0.02\ \mathrm{s}.
$$

每台推进器的力沿其刚体局部 +X 方向写入 PhysX。安装姿态把局部 +X 转换成第 7
节给出的 $d_i$，力臂力矩由物理引擎自然产生。

## 12. 总闭环方程

把当前实现概括为：

$$
\mathcal R
\xrightarrow{PD+FF}
\dot\nu_{cmd,b}
\xrightarrow{M,J,\tau_{ext}}
\tau_{d,b}
\xrightarrow{B,W,\lambda}
f_d
\xrightarrow{T^{-1}}
u_{cmd}.
$$

仿真执行侧：

$$
u_{cmd}
\xrightarrow{\text{一阶滞后}}
u
\xrightarrow{n(u),T_H(n)}
f
\xrightarrow{B}
\tau_{thruster}.
$$

机器人受到的实际 wrench 还包括：

$$
\tau_{actual}
=\tau_{thruster}
+\tau_{gravity}
+\tau_{buoyancy}
+\tau_{hydro}.
$$

控制循环中，物理推进后 Command 先用新状态更新误差，再推进下一时刻参考，下一
控制周期重新计算 throttle。

## 13. 配置字段与数学量

| 配置字段 | 当前值 | 数学含义 |
|---|---:|---|
| `position_kp` | `[0.7,0.7,0.8]` | $K_p^p$ |
| `position_kd` | `[2.0,2.0,2.2]` | $K_d^p$ |
| `attitude_kp` | `[8,8,12]` | $K_p^R$ |
| `attitude_kd` | `[3,3,3]` | $K_d^R$ |
| `max_linear_acceleration` | `[0.2,0.2,0.2]` | $a_{max}$ |
| `max_angular_acceleration` | `[1.5,1.5,2.0]` | $\alpha_{max}$ |
| `wrench_weights` | `[1,1,1,5,5,5]` | $W$ 的对角元素 |
| `wrench_command_mask` | `[1,1,1,1,1,1]` | 六维通道掩码 |
| `allocation_damping` | $10^{-6}$ | $\lambda$ |
| `reaction_torque_per_thrust` | 0 | $k_i$ |
| `use_added_mass` | true | 是否在 $M$ 中加入 $M_A$ |
| `compensate_hydrodynamics` | false | 是否实时补偿 `uw.hydro` |
| `gravity_w` | `[0,0,-9.81]` | 控制器重力加速度 |
| `throttle_deadband` | 0.075 | $\delta$ |
| `thrust_scale` | 1.78363636 | 已含 Heavy 力常数比例的 $s_H$ |
| `inversion_iterations` | 40 | 推力反解二分次数 |

## 14. 仿真验收结果的物理含义

当前实现已在 Isaac Sim 5.1、单环境、随机种子 42 下通过：

| 指标 | 结果 |
|---|---:|
| 推进器数量 | 8 |
| 分配矩阵 rank | 6 |
| 位姿目标最终位置误差 | $0.083902\ \mathrm{m}$ |
| 位姿目标最终姿态误差 | $0.767717^\circ$ |
| Lemniscate 位置 RMSE | $0.126527\ \mathrm{m}$ |
| Lemniscate 最大位置误差 | $0.200337\ \mathrm{m}$ |
| Lemniscate 平均全姿态误差 | $2.484329^\circ$ |
| Lemniscate 最大全姿态误差 | $5.784646^\circ$ |
| Lemniscate 推进器饱和率 | 0 |
| Lemniscate 峰值绝对 throttle | 0.971281 |

这些结果说明在当前 45 秒周期和当前轨迹尺度下，六自由度分配、推力缩放和闭环
参数能够实现稳定跟踪。峰值 throttle 接近 1，但没有达到验收脚本定义的饱和阈值
0.99。若缩短周期、增大曲线或加入强水流，推力裕量会首先成为限制。

## 15. 当前模型边界

- 位姿环是 PD，没有积分项，未建模恒定扰动可能留下稳态误差。
- 刚体控制器不是完整的六自由度反馈线性化；阻尼和附加质量科氏项主要在执行层。
- 当前 `compensate_hydrodynamics: false`，水动力不作为逐步前馈完全抵消。
- 普通姿态 SLERP 不输出角速度和角加速度；Lemniscate 只输出 yaw rate。
- 控制质量包含 8 个 rotor 质量，但默认惯量读取 `base_link`，不是严格复合惯量。
- 浮力模型用 $g=9.8$，控制器重力用 $g=9.81$，两者存在小差异。
- 推进器反作用扭矩设置为零；yaw 主要来自水平推进器力臂。
- active-set 分配是工程化有界阻尼最小二乘，不是通用最优控制或 QP 求解器。
- 分配器当前逐环境求解，适合验证和少量环境，不适合直接扩展到数千环境。
- USD/PhysX 实际几何是运行时真值；若以后修改资产推进器姿态，本文矩阵也应重算。

## 16. 调试顺序

1. 检查参考 $p_d,v_d,a_{ff},q_d,\omega_d$ 是否连续和有限。
2. 检查位置误差、轴角误差以及加速度命令是否长期触及限幅。
3. 检查 `desired_wrench_b`，确认重力和浮力补偿符号正确。
4. 检查推进器数为 8、`allocator.rank == 6`，并记录奇异值。
5. 检查 `allocation.residual_wrench_b`，区分几何不可实现与推力饱和。
6. 检查目标推力、RPM 和 throttle，注意 $\pm0.075$ 死区。
7. 比较控制器 `realized_wrench_b` 与执行层 `uw.thrusts_b`，确认 Heavy 比例一致。
8. 最后再分析水动力、流速、一阶滞后和 PhysX 刚体响应。

## 17. 代码导航

| 阶段 | 类或方法 |
|---|---|
| Heavy 资产和物理参数 | `assets_heavy.make_isaaclab_cfg` |
| 鼠标目标平滑参考 | `MinimumJerkPoseTrajectory` |
| Lemniscate 几何和相位 | `Lemniscate3DTrajectory` |
| 切线航向和六字段参考 | `Lemniscate3DCommand` |
| 位姿误差到加速度 | `PoseAccelerationController.compute` |
| 加速度到 wrench | `RigidBodyWrenchController.compute` |
| 几何读取和外力组装 | `PoseReferenceTrackingActionBase` |
| 8 推进器完整性检查 | `BlueROVHeavyPoseTrackingAction` |
| Lemniscate Heavy Action | `BlueROVHeavyLemniscateTrackingAction` |
| 有界推力分配 | `ThrusterAllocator.allocate` |
| 推力/RPM/throttle | `ParameterizedThrusterModel` |
| 总控制链 | `ExplicitPoseController.compute` |
| 水动力和实际施力 | `UnderwaterRobot.write_data_to_sim` |
