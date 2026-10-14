"""Go-Explore route search for Level 1 (savestates; NOT RL - produces demonstrations).
usage: python tools/go_explore.py SEED SECONDS [TEMP=40] [REGION=sec,xmin,xmax or -] [SCORE_WEIGHT=0.25]
"""
"""Go-Explore with the training action set and the exact Shaped start timing (reset + 1 noop frame).
Cell = (section, x//16). Never accepts a life loss. Dumps archive every 60 s (resumable)."""
import numpy as np, stable_retro as retro, pickle, time, sys, os
from pv.env import GAME, COMBOS
seed=int(sys.argv[1]); budget=float(sys.argv[2]); TEMP=float(sys.argv[3]) if len(sys.argv)>3 else 40.0; REGION=tuple(int(v) for v in sys.argv[4].split(',')) if len(sys.argv)>4 and sys.argv[4]!='-' else None; QW=float(sys.argv[5]) if len(sys.argv)>5 else 0.25; os.makedirs("runs/search", exist_ok=True); out=f"runs/search/arch_{seed}.pkl"; src=out
rng=np.random.default_rng(seed)
e=retro.make(GAME,state="Level1",render_mode=None); e.reset(); names=e.unwrapped.buttons; em=e.unwrapped.em
ACTS=[]
for c in COMBOS:
    a=np.zeros(9,np.int8)
    for b in c: a[names.index(b)]=1
    ACTS.append(a)
# action prior: mostly forward motion / jumps, some shooting, ducking, waiting (for the moving stone)
P=np.array([.07,.22,.05,.12,.14,.10,.08,.15,.03,.04]); P/=P.sum()   # more duck (3) and shoot (4)
R=lambda: e.unwrapped.get_ram().astype(int); X=lambda r: r[86]+256*r[87]
if os.path.exists(out) or os.path.exists(src):
    arch=pickle.load(open(out if os.path.exists(out) else src,"rb")); print("resumed", len(arch))
    for v in arch.values(): v.setdefault("score",0)
    arch={(k[0],k[1],k[2],0) if len(k)==3 else k: v for k,v in arch.items()}
else:
    e.reset(); e.step(ACTS[0]); r=R()
    arch={(1,X(r)//16,0,0): dict(state=em.get_state(), x=X(r), sec=1, visits=0, t=0, traj=[], lives=r[247], score=0, hits=0, q=(int(r[647])*60+int(r[648])*10+int(r[649]))*15)}
def safe():
    """Survival check: from the current state, stand still 10 steps; must keep the life and stay out of the water."""
    st=em.get_state(); l0=e.unwrapped.get_ram()[247]; ok=True
    for _ in range(40):
        e.step(ACTS[0]); rr=e.unwrapped.get_ram()
        if rr[247]<l0 or (rr[88]>136 and rr[94]==0): ok=False; break
    em.set_state(st); return ok
prog=lambda k: k[0]*200+k[1]+0.5*k[2]+4*k[3]   # boss damage dominates inside the arena
t0=time.time(); tl=t0; it=0; done_boss=False; trans=[(k[0],0) for k in arch]
while time.time()-t0<budget:
    it+=1; keys=[k for k in arch if REGION is None or (k[0]==REGION[0] and REGION[1]<=arch[k]['x']<=REGION[2])] or list(arch)
    p=np.array([prog(k) for k in keys],float); w=np.exp((p-p.max())/TEMP)/(1+np.array([arch[k]["visits"] for k in keys])); w/=w.sum()   # softer frontier focus, punish dead ends
    c=arch[keys[rng.choice(len(keys),p=w)]]; c["visits"]+=1
    em.set_state(c["state"]); hits=c.get("hits",0); rr0=e.unwrapped.get_ram(); psc=int(rr0[244])*65536+int(rr0[245])*256+int(rr0[246]); traj=list(c["traj"]); t=c["t"]; lastx=c["x"]; lives=c["lives"]; a=rng.choice(10,p=P)
    macro=[]; vx=c["x"]; vdx=0
    for step in range(80):
        rr=e.unwrapped.get_ram(); cx=int(rr[86])+256*int(rr[87]); dx=cx-vx
        on_vine=(rr[102]==251) or (rr[94]>34 and rr[88]<100)
        if on_vine and rng.random()<0.9:                  # vine routine: RIGHT+A at the end of each swing, else hang on
            a = 7 if (dx==0 or (vdx>0 and dx<0) or (vdx<0 and dx>0)) else 0; macro=[]
        elif macro: a=macro.pop()
        else:
            u=rng.random()
            if u<0.06: macro=[0]*int(rng.integers(4,40))                       # wait, then time the next move
            elif u<0.11: macro=[3]*int(rng.integers(4,20))                     # duck under arrows ("duck and wait for the arrow")
            elif u<0.16: macro=[4]*int(rng.integers(2,6))+[3]*int(rng.integers(0,8))   # shoot burst, then duck again
            elif u<0.40: a=rng.choice(10,p=P)
            if macro: a=macro.pop()
        for _ in range(4): e.step(ACTS[a])
        traj.append(int(a)); t+=1; r=R(); vdx=dx if dx!=0 else vdx; vx=cx
        if r[247]<lives: break                 # no life loss: the 4-minute level timer cannot be refreshed usefully
        x=X(r); sec=max(r[75],1)
        if x>60000: continue                   # death animation: x reads 0xFFFF
        if sec==c["sec"] and x-lastx>64: continue
        if sec!=c['sec'] and not any(tr[0]==sec for tr in trans): trans.append((sec,t)); print('NEW SECTION',sec,'at t',t,flush=True)
        hang=int(r[94]>34 and r[88]<100)        # hanging on a vine: jump timer runs past a normal jump while airborne
        boss=(sec==3 and x>=1180)
        scn=int(r[244])*65536+int(r[245])*256+int(r[246])
        if boss and scn>psc: hits+=1                  # each hit on the Winged Demon scores points
        if boss and scn-psc>=0x1000:                  # killing blow on the Winged Demon (+1000, BCD) -> Level 1 cleared
            import json; json.dump([int(v) for v in traj], open(f"runs/search/level1_clear_seed{seed}_t{t}.json","w"))
            print('LEVEL 1 CLEARED at step',t,'-> runs/search/level1_clear_seed%d_t%d.json'%(seed,t),flush=True)
        psc=scn
        if c['sec']==3 and ((sec!=3) or (c['x']>1100 and x<200)):
            print('LEVEL END? sec',sec,'x',x,'t',t,'hits',hits,flush=True)
            pickle.dump(dict(traj=list(traj),state=em.get_state(),t=t,sec=sec,x=x),open(f'runs/search/level_end_{seed}_{t}.pkl','wb'))
        dmg=min(hits,60) if boss else 0
        lastx=x; kk=(sec,x//16,hang,dmg)
        sc=int(r[244])*65536+int(r[245])*256+int(r[246])   # score bytes; higher = more enemies cleared
        tl=int(r[647])*60+int(r[648])*10+int(r[649])
        q=tl*15+sc*QW                             # time LEFT on the game clock (T items = +60 s) + weighted score
        if (r[94]==0 or hang) and (kk not in arch or q>arch[kk]["q"]) and safe():   # only states the hero survives in
            arch[kk]=dict(state=em.get_state(), x=x, sec=sec, visits=0, t=t, traj=list(traj), lives=r[247], score=sc, hits=hits, q=q)
    if time.time()-tl>60:
        tl=time.time(); pickle.dump(arch,open(out,"wb")); b=max(arch,key=prog); print(round(tl-t0), "best sec",b[0],"x",arch[b]["x"],"t",arch[b]["t"],"cells",len(arch),flush=True)
pickle.dump(arch,open(out,"wb")); b=max(arch,key=prog); print("done best", b, arch[b]["t"])
