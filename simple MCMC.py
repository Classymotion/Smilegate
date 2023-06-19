import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    # 타겟 분포 (정규분포)를 정의합니다.
    mu = 0
    sigma = 1
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

def metropolis_hastings(n_samples, initial_value, proposal_std):
    samples = [initial_value]
    current_value = initial_value

    for _ in range(n_samples):
        # 샘플링 대상 분포로부터 샘플을 추출합니다.
        proposal_value = np.random.normal(current_value, proposal_std)

        # 알파 값 계산 (샘플링 대상 분포의 확률을 계산합니다.)
        alpha = min(1, target_distribution(proposal_value) / target_distribution(current_value))

        # 거절 또는 샘플을 받아들일지 결정합니다.
        u = np.random.uniform(0, 1)
        if u < alpha:
            current_value = proposal_value

        samples.append(current_value)

    return samples

# MCMC로 샘플링할 횟수와 초기값, 제안 분포의 표준편차를 설정합니다.
n_samples = 10000
initial_value = 0
proposal_std = 1

# MCMC로 샘플링합니다.
samples = metropolis_hastings(n_samples, initial_value, proposal_std)

# 결과를 시각화합니다.
plt.hist(samples, bins=30, density=True, alpha=0.7, label='MCMC samples')
x = np.linspace(-5, 5, 100)
plt.plot(x, target_distribution(x), 'r', label='Target distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
