import io, math
import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile
from scipy import signal
import plotly.graph_objects as go

st.set_page_config(page_title='Heel Drop Floor Vibration Analyser', layout='wide')
st.title('Heel Drop Floor Vibration Analyser')
st.caption('Automatic multi-impact detection, event-by-event peak acceleration, free-decay spectrum, modal-frequency consistency and damping estimate.')

with st.sidebar:
    st.header('Calibration')
    sensitivity_mv = st.number_input('Sensor sensitivity [mV/(m/s²)]', 0.001, 10000.0, 10.2, 0.1)
    chain_gain = st.number_input('Recording-chain gain', 0.000001, 1000000.0, 1.0, 0.1)
    full_scale_v = st.number_input('WAV full-scale [Vpeak]', 0.000001, 1000.0, 1.0, 0.1)
    engineering_units = st.checkbox('WAV samples already represent m/s²', False)
    st.header('Heel-drop detection')
    hp_hz = st.number_input('High-pass [Hz], 0=off', 0.0, 100.0, 0.5, 0.1)
    threshold_sigma = st.number_input('Detection threshold [robust σ]', 2.0, 30.0, 7.0, 0.5)
    min_gap = st.number_input('Minimum gap between heel drops [s]', 0.3, 20.0, 2.0, 0.1)
    pre_time = st.number_input('Pre-trigger [s]', 0.0, 2.0, 0.15, 0.05)
    decay_delay = st.number_input('Exclude impact transient [s]', 0.0, 1.0, 0.08, 0.01)
    decay_time = st.number_input('Free-decay window [s]', 0.5, 20.0, 4.0, 0.25)
    st.header('Modal analysis')
    fmin = st.number_input('Search minimum [Hz]', 0.2, 100.0, 3.0, 0.5)
    fmax = st.number_input('Search maximum [Hz]', 1.0, 500.0, 40.0, 1.0)
    prominence_db = st.number_input('Minimum spectral prominence [dB]', 1.0, 30.0, 5.0, 0.5)
    cluster_tol = st.number_input('Cross-event frequency tolerance [Hz]', 0.1, 5.0, 0.5, 0.1)

files = st.file_uploader('Upload one or more heel-drop WAV files', type=['wav'], accept_multiple_files=True)

def read_wav(raw):
    fs, x = wavfile.read(io.BytesIO(raw))
    if x.ndim == 1: x = x[:, None]
    if np.issubdtype(x.dtype, np.integer):
        ii = np.iinfo(x.dtype); x = x.astype(float) / max(abs(ii.min), ii.max)
    else: x = x.astype(float)
    return fs, x

def to_acc(x):
    return x if engineering_units else x * full_scale_v / (sensitivity_mv / 1000.0 * chain_gain)

def condition(a, fs):
    a = signal.detrend(a, type='constant')
    if 0 < hp_hz < fs / 2:
        a = signal.sosfiltfilt(signal.butter(4, hp_hz, 'highpass', fs=fs, output='sos'), a)
    return a

def robust_sigma(y):
    med = np.median(y); return 1.4826 * np.median(np.abs(y - med)) + np.finfo(float).eps

def detect_impacts(a, fs):
    env = np.abs(signal.hilbert(a))
    smooth_n = max(1, int(0.02 * fs))
    env_s = np.convolve(env, np.ones(smooth_n)/smooth_n, mode='same')
    baseline = np.median(env_s); threshold = baseline + threshold_sigma * robust_sigma(env_s)
    peaks, props = signal.find_peaks(env_s, height=threshold, distance=max(1, int(min_gap*fs)), prominence=max(threshold*0.35, np.finfo(float).eps))
    # refine each event to the largest absolute acceleration nearby
    refined=[]
    radius=max(1,int(0.15*fs))
    for p in peaks:
        lo=max(0,p-radius); hi=min(len(a),p+radius)
        refined.append(lo+int(np.argmax(np.abs(a[lo:hi]))))
    return np.array(refined,dtype=int), env_s, threshold

def peak_spectrum(seg, fs):
    n=len(seg); w=signal.windows.hann(n, sym=False); cg=w.mean()
    y=np.fft.rfft((seg-np.mean(seg))*w); amp=np.abs(y)/(n*cg)
    if len(amp)>2: amp[1:-1]*=2
    return np.fft.rfftfreq(n,1/fs), amp

def modal_candidates(seg, fs):
    nper=min(len(seg), max(256, int(fs*2.0)))
    f,p=signal.welch(seg,fs=fs,window='hann',nperseg=nper,noverlap=nper//2,detrend='constant',scaling='density')
    db=10*np.log10(np.maximum(p,np.finfo(float).tiny)); mask=(f>=fmin)&(f<=min(fmax,fs/2))
    fm,dm=f[mask],db[mask]
    if len(fm)<3:return f,db,[]
    df=fm[1]-fm[0]; distance=max(1,int(0.4/max(df,1e-9)))
    ids,pr=signal.find_peaks(dm,prominence=prominence_db,distance=distance)
    c=[(float(fm[i]),float(dm[i]),float(q)) for i,q in zip(ids,pr.get('prominences',[]))]
    return f,db,sorted(c,key=lambda z:z[1],reverse=True)

def damping_from_decay(seg, fs, fn):
    if not fn or fn<=0:return np.nan,np.nan
    # narrow band around candidate, then logarithmic decrement from positive envelope peaks
    lo=max(0.5,fn*0.65); hi=min(fs/2*0.95,fn*1.35)
    if hi<=lo:return np.nan,np.nan
    y=signal.sosfiltfilt(signal.butter(3,[lo,hi],'bandpass',fs=fs,output='sos'),seg)
    env=np.abs(signal.hilbert(y)); start=int(0.05*fs); env=env[start:]
    pk,_=signal.find_peaks(env,distance=max(1,int(fs/(fn*1.5))))
    pk=pk[env[pk] > np.max(env)*0.08]
    if len(pk)<4:return np.nan,np.nan
    tt=pk/fs; yy=np.log(np.maximum(env[pk],np.finfo(float).tiny))
    slope,intercept=np.polyfit(tt,yy,1); zeta=-slope/(2*np.pi*fn)
    if not (0<zeta<0.3):return np.nan,np.nan
    return float(zeta), float(100*zeta)

def cluster_modes(event_rows):
    pts=[]
    for r in event_rows:
        for rank,c in enumerate(r['candidates'][:4]): pts.append((c[0],r['event'],rank,c[2]))
    pts.sort(); clusters=[]
    for p in pts:
        placed=False
        for cl in clusters:
            if abs(p[0]-np.median([x[0] for x in cl]))<=cluster_tol:
                cl.append(p);placed=True;break
        if not placed:clusters.append([p])
    out=[]
    n_events=max(1,len(event_rows))
    for cl in clusters:
        freqs=[x[0] for x in cl]; unique=len(set(x[1] for x in cl))
        out.append({'Frequency_Hz':float(np.median(freqs)),'Std_Hz':float(np.std(freqs)),'Events_detected':unique,'Consistency_pct':100*unique/n_events,'Median_prominence_dB':float(np.median([x[3] for x in cl]))})
    return sorted(out,key=lambda x:(x['Events_detected'],x['Median_prominence_dB']),reverse=True)

if not files:
    st.info('Upload a WAV containing repeated heel drops. Absolute acceleration requires correct sensitivity, chain gain and WAV full-scale voltage.')
else:
    all_events=[]; modal_tables=[]
    for uf in files:
        fs,x=read_wav(uf.getvalue())
        for ch in range(x.shape[1]):
            label=f'{uf.name} | Ch {ch+1}'; a=condition(to_acc(x[:,ch]),fs); t=np.arange(len(a))/fs
            impacts,env,thr=detect_impacts(a,fs)
            st.header(label)
            c1,c2,c3=st.columns(3);c1.metric('Detected heel drops',len(impacts));c2.metric('Sample rate',f'{fs:g} Hz');c3.metric('Duration',f'{len(a)/fs:.2f} s')
            fig=go.Figure();fig.add_trace(go.Scatter(x=t,y=a,name='Acceleration',line=dict(width=1)))
            if len(impacts): fig.add_trace(go.Scatter(x=t[impacts],y=a[impacts],mode='markers+text',text=[f'HD{i+1}' for i in range(len(impacts))],textposition='top center',name='Detected heel drops'))
            fig.update_layout(xaxis_title='Time [s]',yaxis_title='Acceleration [m/s²]',height=420);st.plotly_chart(fig,use_container_width=True)
            rows=[]
            for i,p in enumerate(impacts,1):
                peak_lo=max(0,p-int(pre_time*fs)); peak_hi=min(len(a),p+int(0.5*fs))
                ds=p+int(decay_delay*fs); de=min(len(a),ds+int(decay_time*fs), p+int((min_gap-0.1)*fs))
                if de-ds<max(128,int(0.5*fs)):continue
                decay=a[ds:de]; ff,aa=peak_spectrum(decay,fs); fw,pdb,cands=modal_candidates(decay,fs)
                fn=cands[0][0] if cands else np.nan; zeta,zeta_pct=damping_from_decay(decay,fs,fn if np.isfinite(fn) else None)
                peak=float(np.max(np.abs(a[peak_lo:peak_hi]))); rms=float(np.sqrt(np.mean(decay**2)))
                row={'file_channel':label,'event':i,'time_s':p/fs,'peak_acc_m_s2':peak,'peak_acc_mm_s2':peak*1000,'peak_acc_pct_g':peak/9.80665*100,'decay_rms_m_s2':rms,'dominant_Hz':fn,'damping_ratio':zeta,'damping_pct':zeta_pct,'candidates':cands}
                rows.append(row);all_events.append(row)
                with st.expander(f'Heel drop {i}: {p/fs:.2f} s | peak {peak*1000:.3f} mm/s² | candidate {fn:.2f} Hz' if np.isfinite(fn) else f'Heel drop {i}'):
                    td=np.arange(len(decay))/fs
                    q=go.Figure(go.Scatter(x=td,y=decay));q.update_layout(xaxis_title='Time after decay-window start [s]',yaxis_title='Acceleration [m/s²]',height=280);st.plotly_chart(q,use_container_width=True)
                    m=(ff>=fmin)&(ff<=min(fmax,fs/2));q=go.Figure(go.Scatter(x=ff[m],y=aa[m]));q.update_layout(xaxis_title='Frequency [Hz]',yaxis_title='Peak amplitude [m/s²]',height=280);st.plotly_chart(q,use_container_width=True)
                    st.dataframe(pd.DataFrame(cands[:6],columns=['Frequency_Hz','PSD_dB','Prominence_dB']),use_container_width=True)
            display=pd.DataFrame([{k:v for k,v in r.items() if k not in ['candidates','file_channel']} for r in rows])
            st.subheader('Event-by-event results');st.dataframe(display,use_container_width=True)
            modes=cluster_modes(rows); modal_tables.append((label,modes))
            st.subheader('Cross-event modal consistency')
            st.dataframe(pd.DataFrame(modes),use_container_width=True)
            if modes:
                best=modes[0]
                if best['Consistency_pct']>=70 and best['Std_Hz']<=cluster_tol:
                    st.success(f"Recommended reported dominant floor frequency: {best['Frequency_Hz']:.2f} Hz. Detected in {best['Events_detected']}/{len(rows)} valid heel drops; SD {best['Std_Hz']:.2f} Hz.")
                else:
                    st.warning('No highly consistent modal frequency was identified. Review event detection, decay window, search range and possible forced/noise peaks.')
    if all_events:
        st.header('Combined summary')
        export=pd.DataFrame([{k:v for k,v in r.items() if k!='candidates'} for r in all_events])
        st.dataframe(export,use_container_width=True)
        buf=io.BytesIO()
        with pd.ExcelWriter(buf,engine='openpyxl') as writer:
            export.to_excel(writer,index=False,sheet_name='Heel Drops')
            for i,(label,modes) in enumerate(modal_tables,1):pd.DataFrame(modes).to_excel(writer,index=False,sheet_name=f'Modes_{i:02d}')
        st.download_button('Download analysis Excel',buf.getvalue(),'heel_drop_floor_vibration_results.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.caption('Engineering note: the reported frequency is a repeatable free-decay spectral candidate, not an automatically proven structural mode. Confirm ambiguous results using repeated consistency, multiple sensor locations, impact-hammer/FRF testing, or mode-shape evidence.')
