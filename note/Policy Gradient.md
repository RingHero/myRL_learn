## <center><font face='黑体' font color=black size=6>策略梯度算法(Policy Gradient)</font></center> 
### 一、策略梯度算法的本质：
**策略优化的目标是最大化期望累积回报**
学习参数$\theta$使得代价函数$J(\theta)$最大

$J(\theta) = \mathbb{E}_\pi [ R(\tau) ]$

其中，$R(\tau)$ 是从某个策略$\pi_\theta$生成的轨迹$\tau$的回报。故:

$R(\tau)=R_t + \gamma ·R_{t+1} + \gamma_{}^{2}·R_{t+2}+···.$

又因为

$Q_\pi(s_t,a_t)=\mathbb{E}_\pi[R(\tau)|S_t=s_t,A_t=a_t],$
$V_\pi(s_t)=\mathbb{E}_A[(Q_\pi(s_t,A))]=\sum_{a}\pi(a|s;\theta)·Q_\pi(s,a).$

故有:

$J(\theta) = \mathbb{E}_\pi [V(S;\theta)]$

策略梯度定理给出了梯度的计算公式:

$\nabla_\theta J(\theta) = \mathbb{E}\pi [ \nabla\theta \log \pi_\theta(a|s) R(\tau) ]$


### 二、推导：
##### 1、问题定义：
本质也是为了使得整体期望回报最大化
由：

$J(\theta) = \mathbb{E}_\pi [ R(\tau) ]$

$J(\theta) = $$\int P(\tau|\theta)R(\tau) d\tau$
即把所有可能的轨迹$\tau$乘上发生概率再求和，即期望值。

##### 2、梯度上升法与对数求导技巧
由梯度上升法来更新参数，我们需要对$J(\theta)$求关于$\theta$的梯度，即$\nabla_\theta$。
即：
$\nabla_\theta J(\theta) =  \int \nabla_\theta P(\tau|\theta)R(\tau) d\tau$
注意这时候$\nabla_\theta P(\tau|\theta)$不是一个合理的概率分布，无法用蒙特卡洛方法（也就是让 AI 自己去玩游戏采样）来计算这个积分。这个时候就打死结了，无法计算这个积分，也就无法计算梯度了。


##### 3、对数求导技巧

那该怎么办呢？人们想到可以利用高数里最基础的复合函数求导法则：$\nabla \log x = \frac{\nabla x}{x}$，我们可以反推出：
$$\nabla x = x \nabla \log x$$
把 $x$ 替换成 $P(\tau|\theta)$，代入上面的死结公式：
$$\nabla_\theta J(\theta) = \int \underbrace{P(\tau|\theta)}_{\text{概率分布又回来了！}} \nabla_\theta \log P(\tau|\theta) R(\tau) d\tau$$
既然前面又有了概率分布 $P(\tau|\theta)$，我们就可以理直气壮地把积分符号变回期望符号 $\mathbb{E}$：
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \big[ \nabla_\theta \log P(\tau|\theta) R(\tau) \big]$$

##### 4、拆解轨迹概率
上面公式里还有一个大麻烦：$\log P(\tau|\theta)$。我们怎么算一条轨迹发生的总概率呢？根据概率论的链式法则，一条轨迹发生的概率 = 初始状态概率 × (第一步选动作概率 × 第一步环境转移概率) × (第二步选动作概率 × 第二步环境转移概率) $\dots$$$P(\tau|\theta) = P(s_0) \prod_{t=0}^T \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)$$现在，我们要对这个极其复杂的连乘公式取对数 $\log$。得益于对数的性质（乘法变加法），它瞬间变得极其清爽：$$\log P(\tau|\theta) = \log P(s_0) + \sum_{t=0}^T \log \pi_\theta(a_t|s_t) + \sum_{t=0}^T \log P(s_{t+1}|s_t, a_t)$$

我们把这个展开式代回去，并对它求 $\theta$ 的梯度 $\nabla_\theta$：
- $\log P(s_0)$ 里面有 $\theta$ 吗？没有。梯度为 0。
- $\log P(s_{t+1}|s_t, a_t)$ 里面有 $\theta$ 吗？没有（这是环境的物理规律）。梯度为 0。
  
所以，那个让我们头疼的未知环境黑盒，在对 $\theta$ 求导的瞬间，全部灰飞烟灭了！只剩下了我们的策略网络：
$$\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t)$$

##### 5、策略梯度定理
把第 4 步极其干净的结果，代回到第 3 步的期望公式里，我们就得到了最终的定理：$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t|s_t) \right) R(\tau) \right]$$

### 三、策略梯度的优缺点
策略梯度方法的核心是直接参数化策略函数 
$\pi(a|s;\theta)$,其中$θ$是策略参数（例如神经网络的权重）。算法的目标是通过调整参数$θ$，最大化累积回报的期望值。
与基于价值函数的方法（如 Q-Learning）不同，策略梯度直接对策略进行优化，避免了因价值函数估计误差导致的策略退化问题，且天然支持连续动作空间。
同时，它也有容易陷入**局部最优**的缺陷，并且总体更新效率略低于基于函数值的方法。

### 四、算法细节
优于策略梯度算法的本质是优化$J(\theta)$使之最大，而优化$J(\theta)$则需使用梯度上升法：

$\theta ← \theta + \beta·\frac{\partial J(\theta)}{\partial \theta}$

此时优化$\theta$需要计算状态价值函数关于$\theta$的导数$\frac{\partial V(s;\theta)}{\partial \theta}$。根据环境的数据形式不同，$\frac{\partial V(s;\theta)}{\partial \theta}$的计算可以分为两种形式，分别应对离散环境和连续环境。
#### 4.1 策略梯度形式
##### 4.1.1 形式一(Fomula 1，适用于离散环境):

$\frac{\partial V(s;\theta)}{\partial \theta} = \frac{\partial \sum_{a}\pi(a|s;\theta)·Q_\pi(s,a)}{\partial \theta}\\
=\sum_{a} \frac{\partial \pi(a|s;\theta)·Q_\pi(s,a)}{\partial \theta}\\
=\sum_{a} \frac{\partial \pi(a|s;\theta)}{\partial \theta}·Q_\pi(s,a)$

此处假设$Q_\pi$与参数$\theta$无关，其实是不严谨的。但无大碍。
最后得出离散下的形式为：

$\frac{\partial V(s;\theta)}{\partial \theta}=\sum_{a} \frac{\partial \pi(a|s;\theta)}{\partial \theta}·Q_\pi(s,a)$

##### 4.1.2 形式二(Fomula 2，适用于连续环境)
根据形式一变形:
$\frac{\partial V(s;\theta)}{\partial \theta}=\sum_{a} \frac{\partial \pi(a|s;\theta)}{\partial \theta}·Q_\pi(s,a)\\
\\
=\sum_{a} \pi(a|s;\theta) \frac{\partial log\pi(a|s;\theta)}{\partial \theta}·Q_\pi(s,a)$

最后整理得：

$\frac{\partial V(s;\theta)}{\partial \theta}=\mathbb{E}_{A-\pi(·|s;\theta)}[\frac{\partial log\pi(a|s;\theta)}{\partial \theta}·Q_\pi(s,a)]$

#### 3.2 流程
* 观测当前状态$s_t$.
* 根据策略函数$\pi(·|s_t;\theta)$随机采样出动作$a_t$.
* 计算动作价值函数$q_t=Q_\pi(s_t,a_t)$，只能估计出来.(**根据此处$q_t$的计算方法可以分出几种策略梯度算法！**)
* 求梯度$d_{\theta,t}=\frac{\partial log\pi(a|s;\theta)}{\partial \theta}|_{\theta = \theta_t}$.(pytorch、tenserflow**可自动计算**)
* 近似计算出策略梯度:
$g(a_t,\theta_t)=q_t·d_{\theta,t}$.
* 更新策略网络的参数:$\theta_{t+1}=\theta_t + \beta · g(a_t,\theta_t)$.

这里可以注意到第三步计算$q_t$的方法，根据计算方法的不同可以分出几种不同的策略梯度算法，在下一小节将涉及。

#### 3.3 几种不同的策略梯度算法
由上一节我们知道根据算法流程的第三步我们可以得到不同的策略梯度算法，接下来将介绍几种方法。
##### 3.3.1 REFORCEMENT
这是最简单的策略梯度算法。该算法基于蒙特卡洛(Monte Carlo)原理，根据现有的策略网络去进行一次实践训练(执行一次episodic)，生成并记录一次轨迹(trajectory):

$s_1,a_1,r_1,s_2,a_2,r_2,s_3,a_3,r_3,...,s_t,a_t,r_t$.

随后计算折扣收益$\~R(\tau)=\sum_{k}^{t}\gamma_{}^{k-t} r_k$。
由于$Q_\pi(s_t,a_t)=\mathbb{E}[R(\tau)]$,那么就可以用我们上面计算出来的$\~R(\tau)$来近似代替$q_t=Q_\pi(s_t,a_t)$
这样就可以根据后面的流程来计算出策略梯度$g(a_t,\theta_t)=q_t·d_{\theta,t}$，并更新参数$\theta$.

##### 3.3.2 Actor-Critic方法
使用神经网络来近似$q_t$。另一张md讲