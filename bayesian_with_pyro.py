import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 결과 출력
print("data mean:", data.mean())
print("data std:", data.std())
print("Inferred posterior mean:", posterior_mean.mean())
print("Inferred posterior std:", posterior_std.mean())

plt.figure(figsize=(18, 4))
OFFSET = 0.0001
'''
## No. 1 사전 분포와 관측 데이터
plt.subplot(1, 3, 1)
x = torch.linspace(0, 1000, 1000)
prior_dist = dist.Normal(400, 80)
pdf = prior_dist.log_prob(x).exp()
plt.plot(x.numpy(), pdf.numpy(), color = 'red', label='Prior Distribution')
plt.fill_between(x.numpy(), pdf.numpy(), color='red', alpha=0.3)
plt.scatter(data, [OFFSET] * len(data), color='g', marker='x', label='Measured Data')  # 측정 데이터
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Prior and Measured Data')
plt.legend()

## No. 2 사후 분포 시각화
plt.subplot(1, 3, 2)
plt.hist2d(posterior_mean.numpy(), posterior_std.numpy(), bins=30, cmap='Blues')
plt.colorbar(label='Frequency')
plt.scatter(posterior_mean.mean(), posterior_std.mean(), color='red', marker='o', label='Posterior Mean')
plt.xlabel('Posterior Mean')
plt.ylabel('Posterior Std')
plt.legend()
plt.title('Posterior Distribution')
'''
## No. 3 사전 분포와 사후 분포 비교
plt.subplot(1, 3, 3)
posterior_dist = dist.Normal(posterior_mean.mean(), posterior_std.mean())
posterior_pdf = posterior_dist.log_prob(x).exp()
plt.plot(x.numpy(), posterior_pdf.numpy(), color = 'blue', label='Posterior Distribution')
plt.fill_between(x.numpy(), posterior_pdf.numpy(), color='blue', alpha=0.3)
# 사전 분포
plt.plot(x.numpy(), pdf.numpy(), color = 'red', label='Prior Distribution')
plt.fill_between(x.numpy(), pdf.numpy(), color='red', alpha=0.3)
plt.scatter(data, [OFFSET] * len(data), color='g', marker='x', label='Measured Data')  # 측정 데이터
plt.xlim(0, 1000)
plt.xlabel('Value')
# 사후 분포
plt.axvline(x=posterior_mean.mean(), color='b', linestyle='--', label='Posterior Mean')  # 사후 평균
plt.title('Prior and Posterior')
plt.legend()
