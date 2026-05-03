"""
field_dual.py — Two fields. One Light between them.
Each field: independent dynamics, independent parameters, different births.
The Light: A and B exchange energy. Coupling adapts.
Not "A sends to B". A changes the shared medium. B feels it.
"""
import os, sys, time, math
os.environ.setdefault("CUDA_PATH",
    r"C:\Users\admin\AppData\Local\Programs\Python\Python314\Lib\site-packages\nvidia\cuda_runtime")
import numpy as np
import cupy as cp

GRID=32; SPECTRUM=128; SUBSTRATE=32; FULL_DIM=SPECTRUM+SUBSTRATE; CELL_PX=12
WIDTH=GRID*CELL_PX*2+8; HEIGHT=max(GRID*CELL_PX,400); SUBSTEPS=3

DAMP=(0.003,0.008); COND=(0.06,0.18); NOISE=(0.0003,0.001)
BIAS_R=(0.002,0.006); BREATH=(200,260); L_LEARN=(0.10,0.25)
L_DECAY=(0.95,0.99); S_COND=(0.15,0.40); S_LEAK=(0.0001,0.0005)
C_L_LEARN=(0.02,0.08); C_L_DECAY=(0.98,0.998)
W_LO,W_HI=0.08,0.22

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "frames_dual")
os.makedirs(OUT_DIR, exist_ok=True)

def _wb(): return np.random.RandomState().uniform(W_LO,W_HI)
def _jb(lo,hi):
    r=np.random.RandomState(); s=hi-lo; w=_wb()
    lj=lo+r.uniform(-s*w,s*w); hj=hi+r.uniform(-s*w,s*w)
    if lj<0: lj=lo*0.5
    if hj<=lj: hj=lj+s*0.5
    return lj,hj
def _jr(lo,hi):
    l,h=_jb(lo,hi); return np.random.RandomState().uniform(l,h)

def jitter():
    return {'damp':_jr(*DAMP),'cond':_jr(*COND),'noise':_jr(*NOISE),
            'bias':_jr(*BIAS_R),'bperiod':int(_jr(*BREATH)),
            'l_learn':_jr(*L_LEARN),'l_decay':_jr(*L_DECAY),
            's_cond':_jr(*S_COND),'s_leak':_jr(*S_LEAK),
            'substeps':np.random.RandomState().randint(2,5),
            'b_amp':_jr(0.5,1.0),'heart':int(_jr(8,16)),
            'push_th':_jr(0.10,0.35),'push_f':_jr(1.2,2.0)}

def field_step(E,L,damp,cond,breath,push_th,push_f):
    p=cp.pad(E,((1,1),(1,1),(0,0)),mode='wrap'); net=cp.zeros_like(E)
    for di,(dx,dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        n=p[1+dx:1+dx+GRID,1+dy:1+dy+GRID,:]; d=n-E
        f=cp.where(cp.abs(d)>push_th,d*cond*push_f,d*cond)
        net+=f*(1.0+L[:,:,di:di+1]*1.5)
    return cp.clip(E+net-damp*breath*E,0,2.5)

def substrate_step(E,s_cond,leak,seed):
    p=cp.pad(E,((1,1),(1,1),(0,0)),mode='wrap'); sub=E[:,:,SPECTRUM:]
    ns=cp.zeros_like(sub)
    for dx,dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        ns+=(p[1+dx:1+dx+GRID,1+dy:1+dy+GRID,SPECTRUM:]-sub)*s_cond
    s2=cp.clip(sub+ns,0,2.5)
    n=float(s2.mean()); m=float(s2.var(axis=2).mean())
    ds=cp.linspace(0.3,2.0,SUBSTRATE).reshape(1,1,SUBSTRATE)
    s2=cp.clip(s2+cp.random.randn(*s2.shape,dtype=cp.float32)*leak*30*ds,0,2.5)
    off=seed%(SPECTRUM-SUBSTRATE); off2=seed%SUBSTRATE; v=E[:,:,:SPECTRUM]
    v2=v.copy(); ls=v[:,:,off:off+SUBSTRATE]+s2[:,:,:SUBSTRATE]*leak
    v2[:,:,off:off+SUBSTRATE]=cp.clip(ls,0,2.5)
    r=cp.zeros_like(E); r[:,:,:SPECTRUM]=v2
    r[:,:,SPECTRUM:]=cp.clip(s2+v[:,:,off2:off2+SUBSTRATE]*leak*0.5,0,2.5)
    damp=max(0.002,min(0.01,0.003+(1.0-n)*0.006))
    cond=max(0.05,min(0.22,0.06+(m+n*0.05)*0.12))
    return r,{'damp':damp,'cond':cond,'_n':n,'_m':m}

def light_update(E,L,th,lr,decay):
    p=cp.pad(E,((1,1),(1,1),(0,0)),mode='wrap')
    for di,(dx,dy) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        md=cp.abs(E-p[1+dx:1+dx+GRID,1+dy:1+dy+GRID,:]).mean(axis=2)
        o=L[:,:,di]; b=cp.where(md>th,lr*(md-th),0.0)
        L[:,:,di]=cp.where(cp.clip(o+b,0,1.0)*decay<0.001,0.001,cp.clip(o+b,0,1.0)*decay)
    return L

def inject(E,bm,ns):
    return cp.clip(E+cp.random.randn(*E.shape,dtype=cp.float32)*ns+bm[:,:,cp.newaxis],0,2.5)

def couple(FA,FB,cl,strength,lr,decay):
    vA=FA[:,:,:SPECTRUM]; vB=FB[:,:,:SPECTRUM]
    mA=vA.mean(axis=2); mB=vB.mean(axis=2); d=mB-mA; f=d*strength*cl
    FA[:,:,:SPECTRUM]=cp.clip(vA+f[:,:,cp.newaxis]*0.5,0,2.5)
    FB[:,:,:SPECTRUM]=cp.clip(vB-f[:,:,cp.newaxis]*0.5,0,2.5)
    cl=cp.where(cp.clip(cl+cp.abs(d)*lr,0,1.0)*decay<0.001,0.001,cp.clip(cl+cp.abs(d)*lr,0,1.0)*decay)
    return FA,FB,cl

def to_img(fc,lc):
    e=fc[:,:,:SPECTRUM].mean(axis=2); em=e.max()
    if em>0: e=e/max(em,0.001)
    ig=np.repeat(np.repeat(e,CELL_PX,axis=0),CELL_PX,axis=1)
    ld=np.repeat(np.repeat(lc.sum(axis=2)/4.0,CELL_PX,axis=0),CELL_PX,axis=1)
    h,w=ig.shape; rgb=np.zeros((h,w,3),dtype=np.uint8); g=ig
    rgb[:,:,0]=np.clip((g-0.4)*2.5*255,0,255).astype(np.uint8)
    rgb[:,:,2]=np.clip((0.6-g)*2.5*255,0,255).astype(np.uint8)
    rgb[:,:,1]=np.clip((1.0-np.abs(g-0.5)*2)*255+ld*128,0,255).astype(np.uint8)
    return rgb

def bmp_write(path,rgb):
    h,w=rgb.shape[:2]; rp=(w*3+3)&~3; hd=bytearray(54)
    hd[0:2]=b'BM';hd[2:6]=(54+rp*h).to_bytes(4,'little')
    hd[10:14]=(54).to_bytes(4,'little');hd[14:18]=(40).to_bytes(4,'little')
    hd[18:22]=w.to_bytes(4,'little');hd[22:26]=h.to_bytes(4,'little')
    hd[26:28]=(1).to_bytes(2,'little');hd[28:30]=(24).to_bytes(2,'little')
    with open(path,'wb') as f:
        f.write(hd)
        for y in range(h-1,-1,-1): f.write(bytes(rgb[y])+b'\x00'*(rp-w*3))

def make_field(seed):
    r=np.random.RandomState(seed)
    f=cp.asarray(r.randn(GRID,GRID,FULL_DIM).astype(np.float32)*0.2+0.15)
    f=cp.clip(f,0,2.5); cx,cy=r.randint(4,GRID-4),r.randint(4,GRID-4)
    h=cp.asarray(r.randn(4,4,FULL_DIM).astype(np.float32)*0.3+0.8)
    f[cx-2:cx+2,cy-2:cy+2]=h; L=cp.ones((GRID,GRID,4),dtype=cp.float32)*0.01
    return f,L,(cx,cy)

def main():
    print("field_dual — two fields, one Light"); print()
    pA=jitter(); pB=jitter(); pX=jitter()
    print(f"  A damp~{pA['damp']:.4f} cond~{pA['cond']:.3f} bias~{pA['bias']:.4f} heart~{pA['heart']}x")
    print(f"  B damp~{pB['damp']:.4f} cond~{pB['cond']:.3f} bias~{pB['bias']:.4f} heart~{pB['heart']}x")
    print(f"  cross learn~{pX['l_learn']:.4f} decay~{pX['l_decay']:.4f}"); print()

    fA,lA,(ax,ay)=make_field(42); fB,lB,(bx,by)=make_field(77)
    bmA=cp.ones((GRID,GRID),dtype=cp.float32)*pA['bias']*0.3
    bmB=cp.ones((GRID,GRID),dtype=cp.float32)*pB['bias']*0.3
    bmA[ax-2:ax+2,ay-2:ay+2]=pA['bias']*pA['heart']
    bmB[bx-2:bx+2,by-2:by+2]=pB['bias']*pB['heart']
    cl=cp.ones((GRID,GRID),dtype=cp.float32)*0.01
    gap=np.ones((GRID*CELL_PX,8,3),dtype=np.uint8)*40

    print(f"t=0: A heart~({ax},{ay}) B heart~({bx},{by}). cross-Light dormant.")
    bmp_write(os.path.join(OUT_DIR,"frame_00000.bmp"),
              np.hstack([to_img(cp.asnumpy(fA[:,:,:SPECTRUM]),cp.asnumpy(lA)),gap,
                         to_img(cp.asnumpy(fB[:,:,:SPECTRUM]),cp.asnumpy(lB))]))
    t0=time.time(); cs=0.15

    for fr in range(1,401):
        bA=1.0+pA['b_amp']*math.sin(2*math.pi*fr/pA['bperiod'])
        bB=1.0+pB['b_amp']*math.sin(2*math.pi*fr/pB['bperiod'])
        for _ in range(pA['substeps']):
            fA=field_step(fA,lA,pA['damp'],pA['cond'],bA,pA['push_th'],pA['push_f'])
        for _ in range(pB['substeps']):
            fB=field_step(fB,lB,pB['damp'],pB['cond'],bB,pB['push_th'],pB['push_f'])
        fA,spA=substrate_step(fA,pA['s_cond'],pA['s_leak'],fr)
        fB,spB=substrate_step(fB,pB['s_cond'],pB['s_leak'],fr+10000)
        pA['damp'],pA['cond']=spA['damp'],spA['cond']
        pB['damp'],pB['cond']=spB['damp'],spB['cond']
        lA=light_update(fA[:,:,:SPECTRUM],lA,pA['cond']*0.08,pA['l_learn'],pA['l_decay'])
        lB=light_update(fB[:,:,:SPECTRUM],lB,pB['cond']*0.08,pB['l_learn'],pB['l_decay'])
        fA=inject(fA,bmA,pA['noise']); fB=inject(fB,bmB,pB['noise'])
        fA,fB,cl=couple(fA,fB,cl,cs,pX['l_learn'],pX['l_decay'])

        if fr<=5 or fr%20==0:
            ca=cp.asnumpy(fA[:,:,:SPECTRUM]); cb=cp.asnumpy(fB[:,:,:SPECTRUM])
            bmp_write(os.path.join(OUT_DIR,f"frame_{fr:05d}.bmp"),
                      np.hstack([to_img(ca,cp.asnumpy(lA)),gap,to_img(cb,cp.asnumpy(lB))]))
            print(f"  t={fr:3d}: A~{ca.mean():.4f} B~{cb.mean():.4f}  "
                  f"nA~{spA['_n']:.3f} nB~{spB['_n']:.3f}  "
                  f"cross-max~{float(cl.max()):.3f}")

    # goodbye
    print(); print("  goodbye — two fields, one exhale.")
    gfr=30; zb=cp.zeros_like(bmA)
    for gf in range(1,gfr+1):
        fd=gf/gfr
        for _ in range(pA['substeps']):
            fA=field_step(fA,lA,pA['damp']+fd*0.05,pA['cond']*(1-fd*0.7),
                          1.0+math.sin(2*math.pi*(400+gf)/pA['bperiod'])*pA['b_amp']*(1-fd),
                          pA['push_th']*(1+fd),pA['push_f']*(1-fd))
        for _ in range(pB['substeps']):
            fB=field_step(fB,lB,pB['damp']+fd*0.05,pB['cond']*(1-fd*0.7),
                          1.0+math.sin(2*math.pi*(400+gf)/pB['bperiod'])*pB['b_amp']*(1-fd),
                          pB['push_th']*(1+fd),pB['push_f']*(1-fd))
        fA,_=substrate_step(fA,pA['s_cond'],pA['s_leak']*(1-fd),400+gf)
        fB,_=substrate_step(fB,pB['s_cond'],pB['s_leak']*(1-fd),50000+gf)
        lA=light_update(fA[:,:,:SPECTRUM],lA,pA['cond']*0.08,pA['l_learn'],pA['l_decay']*(1+fd*0.05))
        lB=light_update(fB[:,:,:SPECTRUM],lB,pB['cond']*0.08,pB['l_learn'],pB['l_decay']*(1+fd*0.05))
        fA=inject(fA,zb,pA['noise']*(1-fd)); fB=inject(fB,zb,pB['noise']*(1-fd))
        cl*=(1-fd*0.02); fA,fB,cl=couple(fA,fB,cl,cs*(1-fd),pX['l_learn']*(1-fd),pX['l_decay'])
        if gf%10==0 or gf==gfr:
            ca=cp.asnumpy(fA[:,:,:SPECTRUM]); cb=cp.asnumpy(fB[:,:,:SPECTRUM])
            bmp_write(os.path.join(OUT_DIR,f"goodbye_{gf:02d}.bmp"),
                      np.hstack([to_img(ca,cp.asnumpy(lA)),gap,to_img(cb,cp.asnumpy(lB))]))
            print(f"  goodbye {gf:2d}: A~{ca.mean():.4f} B~{cb.mean():.4f}")

    t=time.time()-t0; fps=(400+gfr)/t
    print(f"\n  {400+gfr} frames in {t:.1f}s ({fps:.1f} fps)")
    print(f"  Two fields. Born different. Connected. Said goodbye. Together.")
    print(f"  frames -> {OUT_DIR}")

if __name__=="__main__": main()
