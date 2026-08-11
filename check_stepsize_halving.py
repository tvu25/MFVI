import numpy as np, time, json
from pathlib import Path

def sech2(x):
    t=np.tanh(x); return 1-t*t

def build_target(a=2., radius=8., size=200001):
    grid=np.linspace(-radius,radius,size)
    ld=-.5*grid*grid-a*np.cos(grid); ld-=ld.max(); den=np.exp(ld)
    dx=grid[1]-grid[0]; cdf=np.cumsum(den)*dx; cdf/=cdf[-1]
    return grid,cdf

def drift(X,a,J):
    means=np.mean(np.tanh(X),axis=2)
    inter=means@J.T
    return X-a*np.sin(X)+sech2(X)*inter[:,:,None]

def rms_se(vals):
    sq=vals*vals; ms=sq.mean(); rms=np.sqrt(ms)
    se_ms=sq.std(ddof=1)/np.sqrt(len(vals)); se=se_ms/(2*rms)
    return float(rms),float(se)

def w2_rows(A,B):
    sa=np.sort(A,axis=2); sb=np.sort(B,axis=2)
    return np.sqrt(np.sum(np.mean((sa-sb)**2,axis=2),axis=1))

def triple(h, seed, n=1024, reps=12, final=.512, a=2., href=.0005, hfine=.00025):
    assert np.isclose(h/href,round(h/href)) and np.isclose(href/hfine,round(href/hfine))
    J=np.array([[0.,.5],[.5,0.]])
    grid,cdf=build_target(a)
    rng=np.random.default_rng(seed)
    init=np.interp(rng.random((reps,2,n)),cdf,grid)
    C=init.copy(); M=init.copy(); F=init.copy()
    coarse_steps=int(round(final/h)); meds_per=int(round(h/href)); fine_per=int(round(href/hfine))
    t0=time.perf_counter()
    for _ in range(coarse_steps):
        dWc=np.zeros_like(C)
        for _ in range(meds_per):
            dWm=np.zeros_like(M)
            for _ in range(fine_per):
                dW=np.sqrt(hfine)*rng.normal(size=F.shape)
                F = F - hfine*drift(F,a,J) + np.sqrt(2.)*dW
                dWm += dW
            M = M - href*drift(M,a,J) + np.sqrt(2.)*dWm
            dWc += dWm
        C = C - h*drift(C,a,J) + np.sqrt(2.)*dWc
    eCM=w2_rows(C,M); eCF=w2_rows(C,F); eMF=w2_rows(M,F)
    return {"h":h,"coarse_vs_5e-4":rms_se(eCM),"coarse_vs_2p5e-4":rms_se(eCF),"5e-4_vs_2p5e-4":rms_se(eMF),"ratio_refdiff_to_cf":float(rms_se(eMF)[0]/rms_se(eCF)[0]),"seconds":time.perf_counter()-t0}

seed0=20260717
hs=[.008,.004,.002,.001]
out=[]
for idx,h in enumerate(hs):
    r=triple(h, seed0+900+idx)
    print(r, flush=True); out.append(r)
output_path = Path(__file__).resolve().with_name('stepsize_halving_check.json')
with output_path.open('w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(f'Wrote {output_path}')
# slopes
cm=np.array([r['coarse_vs_5e-4'][0] for r in out]); cf=np.array([r['coarse_vs_2p5e-4'][0] for r in out]); hs=np.array(hs)
print('slope medium-ref',np.polyfit(np.log(hs),np.log(cm),1)[0])
print('slope fine-ref',np.polyfit(np.log(hs),np.log(cf),1)[0])
