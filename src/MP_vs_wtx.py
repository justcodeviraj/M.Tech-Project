import torch
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

def MP(x, gamma):
    sorted = torch.sort(x, descending=True)[0]
    cum_sum = torch.cumsum(sorted, dim=0)
    k = None
    for i in range(len(x)):
      alpha = (cum_sum[i]- gamma) / (i+1)
      if sorted[i] - alpha > 0 :
        k = i + 1
      else:
        break
    if k is None:
      return 0
    alpha = (cum_sum[k-1] - gamma) / k
    return alpha

def g(x, gamma):
  D = len(x)
  x = torch.cat([x, -x])
  lse = gamma**2*D*torch.logsumexp(x/gamma, dim=0)
  return lse

def lse_vs_wtx(norm, gamma, D, trials=10):
    errors = []
    for _ in range(trials):
      x = torch.rand((D), dtype=torch.float32)
      w = torch.rand((D), dtype=torch.float32)
      x = norm*x/torch.norm(x)
      w = norm*w/torch.norm(w)

      lse = 0.5*(g(w+x, gamma) - g(w-x, gamma))
      wtx = w@x

      error = abs(lse-wtx)
      errors.append(error.item())

    return np.mean(errors)


def mp_vs_lse(norm, gamma, D, trials=10):
    errors = []
    for _ in range(trials):
      # x = torch.zeros((D), dtype=torch.float32)
      # w = torch.zeros((D), dtype=torch.float32)
      x = torch.rand((D), dtype=torch.float32)
      w = torch.rand((D), dtype=torch.float32)
      x = norm*x/torch.norm(x)
      w = norm*w/torch.norm(w)

      p = torch.cat([w+x,-w-x])
      q = torch.cat([w-x,-w+x])
      mp = 0.5*(MP(gamma + p, gamma) - MP(gamma + q, gamma))
      # mp = MP(gamma + p, gamma)
      lse = 0.5*(g(w+x, gamma) - g(w-x, gamma))
      # lse = g(w+x, gamma)
      error = abs(lse-mp)
      errors.append(error.item())

    return np.mean(errors)

def mp_vs_wtx(norm, gamma, D, trials=10):
    errors = []
    for _ in range(trials):
      x = torch.rand((D), dtype=torch.float32)
      w = torch.rand((D), dtype=torch.float32)
      x = norm*x/torch.norm(x)
      w = norm*w/torch.norm(w)

      p = torch.cat([w+x,-w-x])
      q = torch.cat([w-x,-w+x])
      mp = 0.5*(MP(gamma + p, gamma) - MP(gamma + q, gamma))
      # lse = 0.5*(g(w+x, gamma) - g(w-x, gamma))
      wtx = w@x
      error = abs(mp-wtx)
      errors.append(error.item())

    return np.mean(errors)

###########################################  LSE VS WTX
D = 100
gamma = np.linspace(0.5, 2, 50)
norm = 1
errors = [lse_vs_wtx(norm, g, D) for g in gamma]
bound = [norm**4/(g**2*D) for g in gamma]
plt.figure(figsize=(6, 4))
plt.plot(gamma, errors, label='Actual Error')
plt.plot(gamma, bound, label='Theoretical Error', linestyle='--')
plt.xlabel('gamma')
plt.ylabel('Error')
plt.title('Error vs gamma (K={}, D={})'.format(norm, D))
plt.legend()
plt.grid(True)
plt.show()

D = 10*np.arange(1, 21)
gamma = 1
norm = 1
errors = [lse_vs_wtx(norm, gamma, d) for d in D]
bound = [norm**4/(gamma**2*d) for d in D]
plt.figure(figsize=(6, 4))
plt.plot(D, errors, label='Actual Error')
plt.plot(D, bound, label='Theoretical Error', linestyle='--')
plt.xlabel('Dimension (D)')
plt.ylabel('Error')
plt.title('Error vs Dimension (D) (K={}, gamma={})'.format(norm, gamma))
plt.legend()
plt.grid(True)
plt.show()

D = 100
gamma = 1
norm = 0.1*np.arange(1, 21)
errors = [lse_vs_wtx(n, gamma, D) for n in norm]
bound = [n**4/(gamma**2*D) for n in norm]
plt.figure(figsize=(6, 4))
plt.plot(norm, errors, label='Actual Error')
plt.plot(norm, bound, label='Theoretical Error', linestyle='--')
plt.xlabel('Norm')
plt.ylabel('Error')
plt.title('Error vs Norm (D={}, gamma={})'.format(D, gamma))
plt.legend()
plt.grid(True)
plt.show()

########################################### LSE VS MP


# D = 100
# gamma = np.linspace(0.01, 1, 10)
# norm = 1
# errors = [mp_vs_lse(norm, y, D) for y in gamma]
# bound = [y**2*D*(math.log(2*D)- 1 + 1/(2*D)) for y in gamma]
# plt.figure(figsize=(6, 4))
# plt.plot(gamma, errors, label='Actual Error')
# # plt.plot(gamma, bound, label='Theoretical Error', linestyle='--')
# plt.xlabel('gamma')
# plt.ylabel('Error')
# plt.title('Error vs gamma (K={}, D={})'.format(norm, D))
# plt.legend()
# plt.grid(True)
# plt.show()

# D = 10*np.arange(1, 21)
# gamma = 1
# norm = 1
# errors = [mp_vs_lse(norm, gamma, d) for d in D]
# bound = [gamma**2*d*(math.log(2*d)- 1 + 1/(2*d)) for d in D]
# plt.figure(figsize=(6, 4))
# plt.plot(D, errors, label='Actual Error')
# # plt.plot(D, bound, label='Theoretical Error', linestyle='--')
# plt.xlabel('Dimension (D)')
# plt.ylabel('Error')
# plt.title('Error vs Dimension (D) (K={}, gamma={})'.format(norm, gamma))
# plt.legend()
# plt.grid(True)
# plt.show()


######################################### MP VS WTX

# D = 100
# gamma = np.linspace(0.5, 2, 50)
# norm = 1
# errors = [mp_vs_wtx(norm, g, D) for g in gamma]
# bound = [norm**4/(y**2*D) for y in gamma]
# plt.figure(figsize=(6, 4))
# plt.plot(gamma, errors, label='Actual Error')
# plt.plot(gamma, bound, label='Theoretical Error', linestyle='--')
# plt.xlabel('gamma')
# plt.ylabel('Error')
# plt.title('Error vs gamma (K={}, D={})'.format(norm, D))
# plt.legend()
# plt.grid(True)
# plt.show()