# git3
import numpy as np


class CHMMForward(object):
    '''
    隐马尔可夫模型求观测序列的前向算法
    '''

    def __init__(self, A, B, pai):
        '''
        :param A:状态转移概率分布
        :param B:观察概率分布
        :param pai:初始状态概率分布
        '''
        self.A = np.array(A)  # 状态转移概率分布
        self.B = B  # 观测概率分布
        self.pai = pai  # 初始状态概率分布

    def forward(self, A, B, pai, O):
        N, T = np.shape(A)[0], np.shape(O)[0]  # 隐马尔可夫模型的状态个数N，观测序列的时刻个数T
        print('HMM的状态个数N=%d' % N, '观测序列的时刻总数T=%d' % T)

        alpha = np.zeros((T, N))  # 保存每个时刻每个状体的观测序列出现的概率
        for t in range(T):  # [T-1, T-2, ..., 0]
            if 0 == t:  # 计算初值
                alpha[t] = np.multiply(pai.T, B[:, O[t]])
                print('alpha_t0:', alpha[t])
            else:  # 递推计算时刻t每个状态的观测序列出现的概率
                for i in range(N):
                    alpha_t_i = np.multiply(alpha[t - 1], A[:, i]) * B[i, O[t]]
                    alpha[t, i] = sum(alpha_t_i)  # 时刻t状态i所有观测序列出现的概率之和
            print('\n时刻%d' % t, '每个状态的观测序列的概率:', alpha[t])
            print('update alpha:\n', alpha)
        # 时刻T所有状态的观测序列概率之和，就是目标观测序列出现的概率
        return sum(alpha[-1])

    def GetProb(self, O):
        '''
        :param O:观测序列，求此观测序列出现的概率
        '''
        return self.forward(self.A, self.B, self.pai, O)


def CHMMForward_manual():
    # 隐马尔可夫模型λ=(A, B, pai)
    # A是状态转移概率分布，状态集合Q的大小N=np.shape(A)[0]
    # 从下给定A可知：Q={晴天, 多云, 雨天}, N=3
    A = np.array([[0.5, 0.375, 0.125],
                  [0.25, 0.125, 0.625],
                  [0.25, 0.375, 0.375]])
    # B是观测概率分布，观测集合V的大小T=np.shape(B)[1]  矩阵第一维是行的个数
    # 从下面给定的B可知：V={干透，稍干，潮湿，湿透}，T=4
    B = np.array([[0.6, 0.2,0.15,0.05],
                  [0.25, 0.25,0.25,0.25],
                  [0.05, 0.10,0.35,0.5]])
    # pai是初始状态概率分布，初始状态个数=np.shape(pai)[0]
    pai = np.array([[0.63],
                    [0.17],
                    [0.20]])
    hmm = CHMMForward(A, B, pai)

    O = [0, 2, 3]  # 0表示干透，2表示稍干，3表示湿透，就是(干透，稍干，湿透)观测序列
    prob = hmm.GetProb(O)
    print('\n通过HMM的前向算法计算得到：观测序列', O, '出现的概率为:', prob)


if __name__ == '__main__':
    CHMMForward_manual()
