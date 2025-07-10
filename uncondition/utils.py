import torch 
import os
# os.environ["GEOMSTATS_BACKEND"] = "pytorch"
# from geomstats.geometry.spd_matrices import SPDAffineMetric, SPDMatrices



def exp(mat):
    if mat.dim()==2:
        eigvals, eigvecs = torch.linalg.eigh(mat)
        return eigvecs @ torch.diag(torch.exp(eigvals)) @ eigvecs.T
    else:
        return torch.stack([exp(mat[i]) for i in range(mat.shape[0])])

def log(mat):
    if mat.dim() == 2:
        eigvals, eigvecs = torch.linalg.eigh(mat)
        return eigvecs @ torch.diag(torch.log(eigvals)) @ eigvecs.T
    else:
        return torch.stack([log(mat[i]) for i in range(mat.shape[0])])
    
def RieGauss_sample(d, mu, cov, n_samples, base = False):
    # mu = log(mu)
    # vec = (mu * torch.tril(torch.ones((d, d)))).flatten()[(mu * torch.tril(torch.ones((d, d)))).flatten() != 0]
    if base:
        vec = torch.ones(int(d * (d + 1) * 0.5))
    mean = vec.to(torch.float64)
    
    covariance = (cov * torch.eye(int(d * (d + 1) * 0.5))).to(torch.float64)
    mvn = torch.distributions.MultivariateNormal(mean, covariance)
    sample_flat = mvn.sample((n_samples,))
    
    
    sample = []
    for i in range(sample_flat.shape[0]):
        vec = sample_flat[i]
        n = int((torch.sqrt(torch.tensor(8.0 * len(vec) + 1.0)) - 1) / 2)        
        out = torch.zeros((n, n)).to(torch.float64)
        tril_indices = torch.tril_indices(n, n)
        out[tril_indices[0], tril_indices[1]] = vec
        ex = out.T * (1 - torch.eye(out.shape[0]))
        sample.append(exp(out + ex))
    return torch.stack(sample)

def invsqrtm(A):
    eigvals, eigvecs = torch.linalg.eigh(A)
    invsqrt_eigvals = torch.rsqrt(eigvals)  # 1 / sqrt
    return eigvecs @ torch.diag(invsqrt_eigvals) @ eigvecs.T

    
def half_rev(mat):
    if mat.dim()==1:
        d = int(((8*mat.shape[0] +1)**(0.5)-1)/2)
        out = torch.zeros(d,d).to(torch.float64)
        tril_indices = torch.tril_indices(d, d)
        out[tril_indices[0], tril_indices[1]] = mat.to(torch.float64)
        return out + (out.T * (1 - torch.tril(torch.ones(d,d)))) 
    else:
        return torch.stack([half_rev(mat[i]) for i in range(mat.shape[0])])
    
def half_fla(mat):
    d = mat.shape[0]
    if mat.dim()==2:
        return torch.tensor([mat[pos[0], pos[1]] for pos in torch.tril_indices(d, d).T])
    else:
        return torch.stack([half_fla(mat[i]) for i in range(mat.shape[0])])

def Affine_met(A, B):
    if A.dim() == 2:
        metric = SPDAffineMetric(SPDMatrices(n=A.shape[0], equip=False))
        return (metric.squared_dist(A, B))**(0.5)
    else:
        metric = SPDAffineMetric(SPDMatrices(n=A.shape[2], equip=False))
        return ((metric.squared_dist(A, B))**(0.5)).mean()


