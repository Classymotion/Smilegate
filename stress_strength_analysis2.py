import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import simps

# 요기에 원하는 응력값을 정해줌
user_position = 0

# 두 정규분포의 평균과 표준편차
mu1, sigma1 = user_position, 1  # 첫 번째 정규분포
mu2, sigma2 = 2, 1  # 두 번째 정규분포

# 파손 확률 계산
x = np.linspace(-5, 10, 1000)
stress_pdf = norm.pdf(x, mu1, sigma1)
strength_pdf = norm.pdf(x, mu2, sigma2)
''' Overlap area 절대면적 계산
overlap_area = np.minimum(stress_pdf, strength_pdf)
area = simps(overlap_area, x)
'''
# Overlap area를 두 정규분포를 곱해서 계산
result = stress_pdf * strength_pdf
area = np.trapz(result, x)

# 겹침 영역의 면적의 확률밀도 함수
area_mu = mu2 - mu1
area_sigma = np.sqrt(sigma1**2 + sigma2**2)
area_pdf = norm.pdf(x, area_mu, area_sigma)

# 두 번째 그래프를 위한 데이터
z0 = user_position - mu2
z_left = x[x <= z0]
area_pdf_left = area_pdf[x <= z0]

# 그래프 그리기
plt.figure(figsize=(8, 3))

# 첫 번째 그래프: 두 정규분포와 겹침 영역
plt.subplot(1, 2, 1)
plt.plot(x, stress_pdf, label='Stress PDF', color='blue')
plt.plot(x, strength_pdf, label='Strength PDF', color='red')
plt.fill_between(x, result, color='gray', alpha=0.5, label=f'Overlap Area = {area:.3f}')
plt.axvline(user_position, color='black', linestyle='--', label=f'x={user_position}')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.title('Stress-Strength Analysis')
plt.legend()
plt.grid(True)

# 두 번째 그래프: 겹침 영역의 면적의 확률밀도 함수와 좌측 면적 표시
plt.subplot(1, 2, 2)
plt.plot(x, area_pdf, label='Product Area PDF', color='green')
plt.fill_between(z_left, area_pdf_left, color='green', alpha=0.5, label=f'PDF = {area:.3f}, CDF = {simps(area_pdf_left, z_left):.3f}')
plt.axvline(z0, color='black', linestyle='--', label=f'z={z0}')
plt.xlabel('z')
plt.ylabel('Probability Density')
plt.title('Probability of Failure')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
